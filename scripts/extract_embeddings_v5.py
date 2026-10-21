"""Offline LUAR smoke checks and extraction; never reads a label column."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.embeddings_v5 import (
    EncoderSettings, FrozenLuarEncoder, MODEL_ID, MODEL_REVISION, build_row_manifest, sha256_file,
)


def write_json(path: Path, data: dict | list):
    serialized = json.dumps(data, indent=2, allow_nan=False) + "\n"
    with path.open("x") as handle:
        handle.write(serialized)


def block_network():
    """Guard Python internet socket connect/connect_ex/sendto operations."""
    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError("V5 offline proof blocked a network connection")
            return super().connect(address)

        def connect_ex(self, address):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError("V5 offline proof blocked a network connection")
            return super().connect_ex(address)

        def sendto(self, *args, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError("V5 offline proof blocked a network datagram")
            return super().sendto(*args, **kwargs)

    socket.socket = OfflineSocket


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def read_source(path: Path):
    return pd.read_csv(path, usecols=["id", "text"], dtype=str, keep_default_na=False)


def smoke(encoder, train, context, output):
    samples = ["", "Raw Case, punctuation!\n  Spaces stay.", "Another short passage.",
               "A long passage with repeated tokens and\nnew lines. " * 70,
               "A long passage with distinct tokens and\nnew lines. " * 70]
    for text in samples[:3]:
        episode = encoder.tokenize(text)
        reference = encoder.tokenizer(text, add_special_tokens=True, padding="max_length", max_length=128, truncation=False)
        if episode.input_ids[0].tolist() != reference["input_ids"] or episode.attention_mask[0].tolist() != reference["attention_mask"]:
            raise AssertionError("Reviewed special-token template differs from tokenizer postprocessor")
    batched = encoder.encode(samples)
    synthetic_report = encoder.report
    repeat = encoder.encode(samples)
    individual = np.concatenate([encoder.encode([text]) for text in samples])
    repeat_error = float(np.max(np.abs(batched - repeat)))
    batch_error = float(np.max(np.abs(batched - individual)))
    if repeat_error != 0 or not np.allclose(batched, individual, atol=2e-6, rtol=2e-5):
        raise AssertionError(f"LUAR deterministic/batch smoke failed: {repeat_error=}, {batch_error=}")
    if not np.allclose(np.linalg.norm(batched, axis=1), 1, atol=2e-6):
        raise AssertionError("LUAR vectors must have unit norm")
    # Fixed first 32 unlabeled rows; no timing-driven batch-size search.
    benchmark_texts = train.text.iloc[:32].tolist()
    encoder.encode(benchmark_texts)
    benchmark = encoder.report
    windows = [encoder.tokenize(text).windows for text in train.text]
    longest_index = int(np.argmax(windows))
    longest_vector = encoder.encode([train.text.iloc[longest_index]])
    longest_report = encoder.report
    if not np.isfinite(longest_vector).all() or not np.allclose(np.linalg.norm(longest_vector, axis=1), 1, atol=2e-6):
        raise AssertionError("Longest-row inference produced an invalid vector")
    seconds_per_window = benchmark["elapsed_seconds"] / benchmark["total_windows"]
    hidden_windows = 4000 * float(np.mean(windows))
    hidden_seconds = seconds_per_window * hidden_windows
    conservative_seconds = 4 * hidden_seconds + 90
    report = {**context, "status": "passed", "special_token_template_matches_backend": True, "synthetic_samples": samples,
              "synthetic_report": synthetic_report, "repeat_max_abs_difference": repeat_error,
              "batch_individual_max_abs_difference": batch_error,
              "batch_individual_tolerance": {"atol": 2e-6, "rtol": 2e-5},
              "fixed_first_32_rows_benchmark": benchmark,
              "longest_row_smoke": {"id": train.id.iloc[longest_index], "row_index": longest_index,
                                    "report": longest_report, "finite_unit_vector": True},
              "train_window_distribution": {"total": sum(windows), "min": min(windows), "max": max(windows),
                                             "mean": float(np.mean(windows)),
                                             "quantiles": np.quantile(windows, [0, .5, .9, .95, .99, 1]).tolist()},
              "hidden_4000_rows_estimate_seconds": hidden_seconds,
              "hidden_estimate_with_4x_margin_and_90s_startup": conservative_seconds,
              "nine_hour_runtime_gate": conservative_seconds < 9 * 3600,
              "runtime_limit": "Estimate from this host CPU and observed train lengths; Kaggle CPU and hidden lengths are not measured.",
              "peak_rss_bytes": peak_rss_bytes(), "provenance": encoder.provenance()}
    write_json(output / "smoke.json", report)
    print(json.dumps({k: report[k] for k in ("status", "repeat_max_abs_difference", "batch_individual_max_abs_difference",
                                            "hidden_4000_rows_estimate_seconds", "hidden_estimate_with_4x_margin_and_90s_startup",
                                            "nine_hour_runtime_gate", "peak_rss_bytes")}), flush=True)


def extract(encoder, sources, context, output):
    evidence = json.loads((output / "smoke.json").read_text())
    if evidence["status"] != "passed" or not evidence["nine_hour_runtime_gate"]:
        raise ValueError("A passing smoke test and runtime estimate are required")
    for key in ("plan_sha256", "script_sha256"):
        if evidence[key] != context[key]:
            raise ValueError(f"Smoke evidence does not match extraction: {key}")
    if evidence["provenance"] != encoder.provenance():
        raise ValueError("Smoke evidence does not match encoder/model/runtime provenance")
    for split, (path, frame) in sources.items():
        rows = build_row_manifest(frame.id.tolist(), frame.text.tolist())
        for suffix in ("npz", "json"):
            if (output / f"{split}-embeddings.{suffix}").exists():
                raise FileExistsError(f"Refusing to overwrite the frozen {split} cache")
        log_path = output / f"{split}-progress.jsonl"
        with log_path.open("x") as log:
            def progress(record):
                line = json.dumps(record)
                log.write(line + "\n")
                log.flush()
                print(line, flush=True)
            vectors = encoder.encode(frame.text.tolist(), progress=progress)
        result_path = output / f"{split}-embeddings.npz"
        np.savez_compressed(result_path, embeddings=vectors,
                            ids=np.asarray(frame.id.tolist(), dtype=str),
                            text_sha256=np.asarray([row["text_sha256"] for row in rows], dtype=str),
                            content_tokens=np.asarray(encoder.report["content_tokens"], dtype=np.int64),
                            window_counts=np.asarray(encoder.report["window_counts"], dtype=np.int64))
        metadata = {**context, "status": "complete", "split": split, "shape": list(vectors.shape),
                    "dtype": str(vectors.dtype), "source_csv_sha256": sha256_file(path),
                    "source_path": str(path), "rows": rows, "report": encoder.report,
                    "peak_rss_bytes": peak_rss_bytes(), "provenance": encoder.provenance(),
                    "artifact": {"path": result_path.name, "size_bytes": result_path.stat().st_size,
                                 "sha256": sha256_file(result_path)}}
        write_json(output / f"{split}-embeddings.json", metadata)
        print(json.dumps({"split": split, "shape": list(vectors.shape), "seconds": encoder.report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "extract"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/improvement-v5/model"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/improvement-v5/encoder"))
    parser.add_argument("--train", type=Path, default=Path("data/name-that-llm/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/name-that-llm/test.csv"))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["model"]["id"] != MODEL_ID or plan["model"]["revision"] != MODEL_REVISION or plan["model"]["fine_tuning"]:
        raise ValueError("Plan does not describe the pinned frozen LUAR model")
    settings = EncoderSettings(**plan["encoder"])
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="empty-offline-cache-", dir=args.output) as cache:
        assert not list(Path(cache).iterdir())
        for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_MODULES_CACHE"):
            os.environ[name] = str(Path(cache) / name.lower())
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        block_network()
        encoder = FrozenLuarEncoder(args.model_dir, settings=settings)
        context = {"schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
                   "plan_sha256": sha256_file(args.plan), "script_sha256": sha256_file(__file__),
                   "offline": {"empty_cache_at_start": True, "socket_guard": True,
                               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "local_files_only": True}}
        train = read_source(args.train)
        if args.mode == "smoke":
            smoke(encoder, train, context, args.output)
        else:
            extract(encoder, {"train": (args.train, train), "test": (args.test, read_source(args.test))}, context, args.output)
        print(json.dumps({"mode": args.mode, "wall_seconds": time.perf_counter() - started,
                          "cache_files_at_end": [str(p.relative_to(cache)) for p in Path(cache).rglob("*") if p.is_file()]}), flush=True)


if __name__ == "__main__":
    main()
