"""Frozen local StyleDistance encoder: raw text, complete windows, masked mean pooling."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from typing import Callable, Sequence

import numpy as np

MODEL_ID = "StyleDistance/styledistance"
MODEL_REVISION = "b7df5f0b0480773c097ba3121d83ca32b71015ca"
REVIEWED_SOURCE_SHA256 = {'config.json': 'b7bcc672f41211775356b1bb7b614453b91edc8425f58eb4ba2c71d6d8b1875a', 'modules.json': '8f4b264b80206c830bebbdcae377e137925650a433b689343a63bdc9b3145460', '1_Pooling/config.json': '1c8bc7bd750c5c20d8707f1c5c578f5d69bb3d1d5ebcf4b2fde5128de154ec1c', 'sentence_bert_config.json': 'ec8e29d6dcb61b611b7d3fdd2982c4524e6ad985959fa7194eacfb655a8d0d51', 'config_sentence_transformers.json': 'fe35b3fc2768f364c610e91305d45aa600eda73a2613635a47db835f9cc4fb74', 'tokenizer.json': '6e6a06a662f8bce9823abd4c2cda07dcb7e07a84153a2ed190b8d901516b3162', 'tokenizer_config.json': '56f306b23b9233c94016f1b0b8aea3bc120676b378f70933908c9e62ab2daa36', 'special_tokens_map.json': '06e405a36dfe4b9604f484f6a1e619af1a7f7d09e34a8555eb0b77b66318067f'}
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
    """Verify pinned identity and every model file before loading the local checkpoint."""
    root = Path(model_dir).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Unexpected StyleDistance model identity or revision")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    if len(entries) != len(manifest["files"]) or not set(REQUIRED_MODEL_FILES).issubset(entries):
        raise ValueError("Missing or duplicate model manifest entries")
    for name in entries:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe model manifest path")
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
        "shape": metadata["shape"] == [len(ids), 768],
        "dtype": metadata["dtype"] == "float32",
        "plan": metadata["plan_sha256"] == sha256_file(plan_path),
        "model_identity": provenance["model_id"] == MODEL_ID and provenance["revision"] == MODEL_REVISION,
        "model_manifest": provenance["manifest_sha256"] == sha256_file(Path(model_dir) / "manifest.json"),
        "model_files": provenance["model_files"] == model_manifest["files"],
        "settings": provenance["runtime"]["settings"] == plan["encoder"],
        "encoder_source": provenance["encoder_source_sha256"] == sha256_file(__file__),
        "extraction_source": metadata["script_sha256"] == sha256_file(Path(__file__).resolve().parents[2] / "scripts" / "extract_embeddings_v7.py"),
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
        if not np.array_equal(windows, np.maximum(1, (tokens + 509) // 510)):
            raise ValueError("Embedding cache window coverage is inconsistent")
        if tokens.tolist() != metadata["report"]["content_tokens"] or windows.tolist() != metadata["report"]["window_counts"]:
            raise ValueError("Embedding cache count metadata mismatch")
        if embeddings.shape != (len(ids), 768) or embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
            raise ValueError("Embedding cache vectors are invalid")
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-6, rtol=0):
            raise ValueError("Embedding cache vectors are not unit normalized")
    return embeddings, metadata


@dataclass(frozen=True)
class EncoderSettings:
    content_tokens: int = 510
    sequence_length: int = 512
    max_batch_windows: int = 8
    cpu_threads: int = 2
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self):
        if self.content_tokens != 510 or self.sequence_length != 512:
            raise ValueError("V7 fixes 510 content tokens plus two special tokens")
        if min(self.max_batch_windows, self.cpu_threads) < 1:
            raise ValueError("Batch bounds and thread count must be positive")
        if self.device != "cpu" or self.dtype != "float32":
            raise ValueError("V7 fixes CPU float32 inference")


@dataclass(frozen=True)
class TokenizedPassage:
    input_ids: list[list[int]]
    content_tokens: int

    @property
    def windows(self) -> int:
        return len(self.input_ids)


def pool_window_embeddings(hidden, attention_mask):
    """Released ST 2.7 mean pooling; mask includes the two special tokens."""
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class FrozenStyleDistanceEncoder:
    """Standard RoBERTa loaded from verified files without remote custom code."""

    def __init__(self, model_dir: str | Path, settings: EncoderSettings | None = None):
        self.settings = settings or EncoderSettings()
        self.model_dir = Path(model_dir).resolve(strict=True)
        self.manifest = validate_model_manifest(self.model_dir)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ[name] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        torch.set_num_threads(self.settings.cpu_threads)
        torch.use_deterministic_algorithms(True)
        self.model, loading_info = AutoModel.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32, output_loading_info=True,
        )
        loading_info = _json_native(loading_info)
        if any(loading_info.values()):
            raise ValueError(f"Checkpoint did not load exactly: {loading_info}")
        self.model.to(device="cpu", dtype=torch.float32).eval()
        self.model.requires_grad_(False)
        if self.model.config.model_type != "roberta" or self.model.config.hidden_size != 768:
            raise ValueError("Unexpected StyleDistance architecture")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False, use_fast=True,
        )
        if self.tokenizer.num_special_tokens_to_add(pair=False) != 2:
            raise ValueError("Expected the reviewed two-token RoBERTa template")
        if (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id) != (0, 2, 1):
            raise ValueError("Unexpected special-token IDs")
        self.report: dict = {}
        self.runtime = {
            "python": platform.python_version(), "platform": platform.platform(),
            **{name: importlib.metadata.version(name) for name in
               ("torch", "transformers", "huggingface_hub", "tokenizers", "safetensors", "numpy")},
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "parameter_dtype": str(next(self.model.parameters()).dtype),
            "backbone_attention": self.model.config._attn_implementation,
            "torch_num_threads": torch.get_num_threads(),
            "settings": asdict(self.settings), "loading_info": loading_info,
            "custom_remote_code": False,
            "pooling_accumulation": "float32 window mean; float64 token-count-weighted passage sum; final float32 unit vector",
        }

    def tokenize(self, text: str) -> TokenizedPassage:
        if not isinstance(text, str):
            raise TypeError("Completion text must be an unchanged string")
        content = self.tokenizer(text, add_special_tokens=False, truncation=False,
                                 return_attention_mask=False, verbose=False)["input_ids"]
        chunks = [content[i:i + self.settings.content_tokens]
                  for i in range(0, len(content), self.settings.content_tokens)] or [[]]
        return TokenizedPassage(
            [[self.tokenizer.cls_token_id, *chunk, self.tokenizer.sep_token_id] for chunk in chunks],
            len(content),
        )

    def encode(self, texts: Sequence[str], progress: Callable[[dict], None] | None = None) -> np.ndarray:
        started = time.perf_counter()
        passages = [self.tokenize(text) for text in texts]
        tokenization_seconds = time.perf_counter() - started
        # Only token IDs are retained for all passages. Transformer states are
        # released each bounded window batch, including for arbitrarily long rows.
        windows = [(row, ids) for row, passage in enumerate(passages) for ids in passage.input_ids]
        sums = np.zeros((len(passages), 768), dtype=np.float64)
        weights = np.zeros(len(passages), dtype=np.int64)
        records = []
        with self.torch.inference_mode():
            for start in range(0, len(windows), self.settings.max_batch_windows):
                selected = windows[start:start + self.settings.max_batch_windows]
                batch_started = time.perf_counter()
                width = max(len(ids) for _, ids in selected)
                ids_array = np.full((len(selected), width), self.tokenizer.pad_token_id, dtype=np.int64)
                masks = np.zeros_like(ids_array)
                for index, (_, ids) in enumerate(selected):
                    ids_array[index, :len(ids)] = ids
                    masks[index, :len(ids)] = 1
                inputs = {"input_ids": self.torch.from_numpy(ids_array),
                          "attention_mask": self.torch.from_numpy(masks)}
                hidden = self.model(**inputs).last_hidden_state
                pooled = pool_window_embeddings(hidden, inputs["attention_mask"]).cpu().numpy()
                if pooled.shape != (len(selected), 768) or not np.isfinite(pooled).all():
                    raise ValueError("Invalid StyleDistance pooled output")
                for index, (row, ids) in enumerate(selected):
                    sums[row] += pooled[index].astype(np.float64) * len(ids)
                    weights[row] += len(ids)
                del hidden, pooled, inputs
                record = {"batch": len(records), "windows": len(selected), "padded_tokens": width,
                          "completed_windows": start + len(selected), "total_windows": len(windows),
                          "elapsed_seconds": time.perf_counter() - batch_started}
                records.append(record)
                if progress:
                    progress(record)
        means = sums / np.maximum(weights, 1)[:, None]
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        if not np.isfinite(means).all() or (norms <= 0).any():
            raise ValueError("Invalid StyleDistance passage mean or norm")
        result = (means / norms).astype(np.float32)
        self.report = {"rows": len(passages), "content_tokens": [p.content_tokens for p in passages],
                       "window_counts": [p.windows for p in passages], "total_windows": len(windows),
                       "tokenization_seconds": tokenization_seconds,
                       "elapsed_seconds": time.perf_counter() - started, "batches": records}
        return result

    def provenance(self) -> dict:
        return {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                "manifest_sha256": sha256_file(self.model_dir / "manifest.json"),
                "model_files": self.manifest["files"], "runtime": self.runtime,
                "encoder_source_sha256": sha256_file(__file__)}


