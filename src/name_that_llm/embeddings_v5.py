"""Audited, local-only frozen LUAR-CRUD representations for experiment v5.

Each row is one episode containing every consecutive 126-token content window.
The upstream episode attention has no document-padding mask: only equal-length
episodes may share an inference batch. Competition labels never enter this API.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import types
from typing import Callable, Sequence

import numpy as np


MODEL_ID = "rrivera1849/LUAR-CRUD"
MODEL_REVISION = "8b36ce7f289ee91900a7dff0eaccf5cdca745ab1"
REVIEWED_SOURCE_SHA256 = {
    "model.py": "70d320a277fd26447bae9a57a8b1c4550097f9430666c3c3b106ff736e26c3b5",
    "config.py": "1425b832fbb977f3109b5e3429666a5d0d07be643b3853be53d942162def4b94",
    "config.json": "df2e852aabe2b4361ed9cca5b8112c3a7c336346d1928168b79839c0380d5927",
    "tokenizer.json": "7b62d797c95d1563d3467f48daad0eca3e150f2d587e7e687b557c9ef1b25473",
    "tokenizer_config.json": "4008a762b09edcbdd9b7583372e806ae65bf640ea419e34a6ba2b770e0e91a6b",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
}
REQUIRED_MODEL_FILES = (*REVIEWED_SOURCE_SHA256, "model.safetensors", "vocab.json", "merges.txt", "README.md")


def _json_native(value):
    """Transformers 5 loading diagnostics include sets; preserve them as lists."""
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_native(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Completion text must be an unchanged string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_row_manifest(ids: Sequence[str], texts: Sequence[str]) -> list[dict]:
    if len(ids) != len(texts):
        raise ValueError("IDs and texts have different lengths")
    if len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Row IDs must be unique nonempty strings")
    return [{"row_index": i, "id": row_id, "text_sha256": sha256_text(text)}
            for i, (row_id, text) in enumerate(zip(ids, texts, strict=True))]


def validate_model_manifest(model_dir: str | Path) -> dict:
    """Verify pinned identity and every model file before executing local source."""
    root = Path(model_dir).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Unexpected LUAR model identity or revision")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    for name in REQUIRED_MODEL_FILES:
        if name not in entries:
            raise ValueError(f"Missing model manifest entry: {name}")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a local regular model file: {name}")
        if path.stat().st_size != entries[name]["size_bytes"] or sha256_file(path) != entries[name]["sha256"]:
            raise ValueError(f"Model manifest mismatch: {name}")
    for name, expected in REVIEWED_SOURCE_SHA256.items():
        if entries[name]["sha256"] != expected:
            raise ValueError(f"Source differs from the reviewed revision: {name}")
    return manifest


def load_embedding_cache(cache_dir: str | Path, split: str, ids: Sequence[str],
                         texts: Sequence[str], *, model_dir: str | Path,
                         plan_path: str | Path, source_path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """Reject stale, reordered, edited or mismatched cached vectors before fitting."""
    if split not in ("train", "test"):
        raise ValueError("Expected train or test cache")
    root = Path(cache_dir)
    metadata = json.loads((root / f"{split}-embeddings.json").read_text())
    model_manifest = validate_model_manifest(model_dir)
    provenance = metadata["provenance"]
    expected_rows = build_row_manifest(ids, texts)
    plan = json.loads(Path(plan_path).read_text())
    checks = {
        "status": metadata["status"] == "complete",
        "split": metadata["split"] == split,
        "rows": metadata["rows"] == expected_rows,
        "shape": metadata["shape"] == [len(ids), 512],
        "dtype": metadata["dtype"] == "float32",
        "plan": metadata["plan_sha256"] == sha256_file(plan_path),
        "model_identity": provenance["model_id"] == MODEL_ID and provenance["revision"] == MODEL_REVISION,
        "model_manifest": provenance["manifest_sha256"] == sha256_file(Path(model_dir) / "manifest.json"),
        "model_files": provenance["model_files"] == model_manifest["files"],
        "settings": provenance["runtime"]["settings"] == plan["encoder"],
        "encoder_source": provenance["encoder_source_sha256"] == sha256_file(__file__),
        "extraction_source": metadata["script_sha256"] == sha256_file(Path(__file__).resolve().parents[2] / "scripts" / "extract_embeddings_v5.py"),
    }
    if source_path is not None:
        checks["source_csv"] = metadata["source_csv_sha256"] == sha256_file(source_path)
    if not all(checks.values()):
        raise ValueError(f"Embedding cache provenance mismatch: {[key for key, valid in checks.items() if not valid]}")
    artifact = metadata["artifact"]
    path = root / f"{split}-embeddings.npz"
    if artifact["path"] != path.name or path.stat().st_size != artifact["size_bytes"] or sha256_file(path) != artifact["sha256"]:
        raise ValueError("Embedding cache artifact checksum mismatch")
    with np.load(path, allow_pickle=False) as cache:
        embeddings = cache["embeddings"]
        if cache["ids"].tolist() != list(ids) or cache["text_sha256"].tolist() != [row["text_sha256"] for row in expected_rows]:
            raise ValueError("Embedding cache row/text mismatch")
        tokens, windows = cache["content_tokens"], cache["window_counts"]
        if tokens.shape != (len(ids),) or windows.shape != (len(ids),) or (tokens < 0).any():
            raise ValueError("Embedding cache token counts are invalid")
        if not np.array_equal(windows, np.maximum(1, (tokens + 125) // 126)):
            raise ValueError("Embedding cache window coverage is inconsistent")
        if tokens.tolist() != metadata["report"]["content_tokens"] or windows.tolist() != metadata["report"]["window_counts"]:
            raise ValueError("Embedding cache count metadata mismatch")
        if embeddings.shape != (len(ids), 512) or embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
            raise ValueError("Embedding cache vectors are invalid")
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-6, rtol=0):
            raise ValueError("Embedding cache vectors are not unit normalized")
    return embeddings, metadata


@dataclass(frozen=True)
class EncoderSettings:
    content_tokens: int = 126
    sequence_length: int = 128
    max_batch_windows: int = 32
    document_batch_size: int = 16
    cpu_threads: int = 2
    device: str = "cpu"

    def __post_init__(self):
        if self.content_tokens != 126 or self.sequence_length != 128:
            raise ValueError("V5 fixes 126 content tokens plus two special tokens")
        if min(self.max_batch_windows, self.document_batch_size, self.cpu_threads) < 1:
            raise ValueError("Batch bounds and CPU thread count must be positive")
        if self.device != "cpu":
            raise ValueError("V5 uses the audited CPU float32 path")


@dataclass(frozen=True)
class TokenizedEpisode:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    content_tokens: int

    @property
    def windows(self) -> int:
        return len(self.input_ids)


class FrozenLuarEncoder:
    """Load the reviewed local custom classes and compute normalized vectors."""

    def __init__(self, model_dir: str | Path, settings: EncoderSettings | None = None):
        self.settings = settings or EncoderSettings()
        self.model_dir = Path(model_dir).resolve(strict=True)
        self.manifest = validate_model_manifest(self.model_dir)
        # Set before importing HF. The executable also installs a socket guard.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        torch.set_num_threads(self.settings.cpu_threads)
        torch.use_deterministic_algorithms(True)
        config_class, model_class = self._load_local_classes()
        config = config_class.from_pretrained(str(self.model_dir), local_files_only=True)
        self.model, loading_info = model_class.from_pretrained(
            str(self.model_dir), config=config, local_files_only=True,
            use_safetensors=True, output_loading_info=True,
        )
        loading_info = _json_native(loading_info)
        bad = {key: value for key, value in loading_info.items() if value}
        if bad:
            raise ValueError(f"Checkpoint did not load exactly: {bad}")
        self.model.to(device="cpu", dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        if self.tokenizer.num_special_tokens_to_add(pair=False) != 2:
            raise ValueError("Expected the reviewed RoBERTa two-token template")
        if (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id) != (0, 2, 1):
            raise ValueError("Special token IDs differ from the reviewed RoBERTa template")
        self.report: dict = {}
        self.runtime = {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in
               ("torch", "transformers", "einops", "huggingface_hub", "tokenizers", "safetensors", "numpy")},
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "parameter_dtype": str(next(self.model.parameters()).dtype),
            "backbone_attention": self.model.transformer.config._attn_implementation,
            "torch_num_threads": torch.get_num_threads(),
            "settings": asdict(self.settings), "loading_info": loading_info,
            "loaded_custom_source": {
                name: {"path": str(self.model_dir / name), "sha256": sha256_file(self.model_dir / name)}
                for name in ("model.py", "config.py")
            },
        }

    def _load_local_classes(self):
        # No dynamic HF modules cache, trust_remote_code or remote-code resolution.
        package_name = "_reviewed_luar_crud_" + hashlib.sha256(str(self.model_dir).encode()).hexdigest()[:16]
        package = types.ModuleType(package_name)
        package.__path__ = [str(self.model_dir)]
        sys.modules[package_name] = package
        modules = {}
        old_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            for name in ("config", "model"):
                qualified = f"{package_name}.{name}"
                spec = importlib.util.spec_from_file_location(qualified, self.model_dir / f"{name}.py")
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load reviewed local module {name}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = module
                spec.loader.exec_module(module)
                modules[name] = module
        finally:
            sys.dont_write_bytecode = old_bytecode
        return modules["config"].LUARConfig, modules["model"].LUAR

    def tokenize(self, text: str) -> TokenizedEpisode:
        if not isinstance(text, str):
            raise TypeError("Completion text must be an unchanged string")
        content = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                 return_attention_mask=False, verbose=False)["input_ids"]
        windows = max(1, (len(content) + self.settings.content_tokens - 1) // self.settings.content_tokens)
        input_ids = np.full((windows, self.settings.sequence_length), self.tokenizer.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for index in range(windows):
            start = index * self.settings.content_tokens
            chunk = content[start:start + self.settings.content_tokens]
            # Exact single-sequence RobertaProcessing template from reviewed
            # tokenizer.json. Transformers 5 removed the old build_inputs helper.
            prepared = [self.tokenizer.cls_token_id, *chunk, self.tokenizer.sep_token_id]
            if len(prepared) != len(chunk) + 2 or len(prepared) > self.settings.sequence_length:
                raise ValueError("Unexpected special token construction")
            input_ids[index, :len(prepared)] = prepared
            attention_mask[index, :len(prepared)] = 1
        return TokenizedEpisode(input_ids, attention_mask, len(content))

    def encode(self, texts: Sequence[str], progress: Callable[[dict], None] | None = None) -> np.ndarray:
        started = time.perf_counter()
        episodes = [self.tokenize(text) for text in texts]
        tokenization_seconds = time.perf_counter() - started
        grouped: dict[int, list[int]] = defaultdict(list)
        for index, episode in enumerate(episodes):
            grouped[episode.windows].append(index)
        result = np.empty((len(episodes), 512), dtype=np.float32)
        completed = 0
        batch_records = []
        with self.torch.inference_mode():
            for windows, indices in sorted(grouped.items()):
                # One long row is never truncated. document_batch_size still
                # bounds transformer activation memory when windows > the cap.
                rows_per_batch = max(1, self.settings.max_batch_windows // windows)
                for start in range(0, len(indices), rows_per_batch):
                    selected = indices[start:start + rows_per_batch]
                    batch_started = time.perf_counter()
                    inputs = {
                        name: self.torch.from_numpy(np.stack([getattr(episodes[i], name) for i in selected]))
                        for name in ("input_ids", "attention_mask")
                    }
                    raw = self.model(**inputs, document_batch_size=self.settings.document_batch_size)
                    norms = self.torch.linalg.vector_norm(raw, ord=2, dim=1, keepdim=True)
                    if tuple(raw.shape) != (len(selected), 512) or not self.torch.isfinite(raw).all() or (norms <= 0).any():
                        raise ValueError("Invalid LUAR output shape, value or norm")
                    result[selected] = (raw / norms).cpu().numpy()
                    completed += len(selected)
                    record = {"batch": len(batch_records), "rows": len(selected), "windows_per_row": windows,
                              "completed_rows": completed, "total_rows": len(episodes),
                              "elapsed_seconds": time.perf_counter() - batch_started}
                    batch_records.append(record)
                    if progress:
                        progress(record)
        self.report = {"rows": len(episodes), "content_tokens": [e.content_tokens for e in episodes],
                       "window_counts": [e.windows for e in episodes],
                       "total_windows": sum(e.windows for e in episodes),
                       "tokenization_seconds": tokenization_seconds,
                       "elapsed_seconds": time.perf_counter() - started, "batches": batch_records}
        return result

    def provenance(self) -> dict:
        return {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "manifest_sha256": sha256_file(self.model_dir / "manifest.json"),
                "model_files": self.manifest["files"], "runtime": self.runtime,
                "encoder_source_sha256": sha256_file(__file__)}
