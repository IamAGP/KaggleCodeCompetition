"""Evaluate three predeclared topic-control models using saved prompt folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
import warnings

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ[name] = "2"

import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, file_sha256, load_data, runtime_versions
from name_that_llm.topic_v3 import (
    TOPIC_CANDIDATES, MaskContent, PromptCenteredRidge,
    build_topic_pipeline, center_by_training_groups,
)

LOGGER = logging.getLogger("topic_v3")
ROOT = Path(__file__).resolve().parents[1]


def sanity_checks():
    """Check the nonstandard algebra against a direct dense primal solution."""
    rng = np.random.default_rng(91)
    matrix = rng.normal(size=(12, 8))
    targets = np.tile(np.eye(3), (4, 1))
    targets -= targets.mean(axis=0)
    group_indices = [np.arange(i * 3, (i + 1) * 3) for i in range(4)]
    errors = []
    for strength in (0.5, 1.0):
        residual = center_by_training_groups(matrix, group_indices, strength)
        gram = matrix @ matrix.T
        centered = center_by_training_groups(gram, group_indices, strength)
        centered = center_by_training_groups(centered.T, group_indices, strength).T
        np.testing.assert_allclose(centered, residual @ residual.T, atol=1e-12)
        dual = solve(centered + np.eye(len(matrix)), targets, assume_a="pos")
        computed = matrix.T @ center_by_training_groups(dual, group_indices, strength)
        primal = solve(residual.T @ residual + np.eye(matrix.shape[1]), residual.T @ targets, assume_a="pos")
        np.testing.assert_allclose(computed, primal, atol=1e-12)
        errors.append(float(np.abs(computed - primal).max()))
    assert MaskContent.mask("The cobalt—21, isn't here.") == "The ******—##, ***'* here."
    assert MaskContent.mask("The cobalt—21, isn't here.") == MaskContent.mask("The cobalt—21, isn't here.")
    # A tiny synthetic vocabulary permits a real fit while keeping validation
    # labels and held-out tokens completely outside the learned transforms.
    texts = [
        "The red kite is here. Yet it is calm.",
        "The blue kite is here; indeed it is calm.",
        "The red boat is here. Yet it is calm.",
        "The blue boat is here; indeed it is calm.",
        "The red cloud is here. Yet it is calm.",
        "The blue cloud is here; indeed it is calm.",
    ]
    labels = np.asarray(["a", "b"] * 3)
    groups = np.repeat(np.arange(3), 2)
    model = PromptCenteredRidge().fit(texts, labels, groups)
    a, b = "The uniqueqz boat is here.", "Something unrelated!"
    alone = model.decision_function([a])
    np.testing.assert_allclose(alone, model.decision_function([a, b])[:1], atol=1e-12)
    np.testing.assert_allclose(alone, model.decision_function([b, a])[1:], atol=1e-12)
    vocabulary = model.features_.named_steps["features"].transformer_list[0][1].vocabulary_
    assert "uqz" not in vocabulary
    permuted = PromptCenteredRidge().fit(texts[::-1], labels[::-1], groups[::-1])
    np.testing.assert_allclose(alone, permuted.decision_function([a]), atol=1e-10)
    for name in TOPIC_CANDIDATES:
        candidate = build_topic_pipeline(name)
        kwargs = {"groups": groups} if getattr(candidate, "requires_groups", False) else {}
        candidate.fit(texts, labels, **kwargs)
        assert np.isfinite(candidate.decision_function(["", "🙂", a])).all()
    try:
        PromptCenteredRidge().fit(texts, labels)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing groups accepted.")
    return {
        "passed": True,
        "dense_primal_max_errors": errors,
        "checks": [
            "Gram centering equals dense centered feature product",
            "dual coefficients equal independently solved dense primal ridge",
            "fixed masking retains punctuation, function words and word lengths",
            "real estimator prediction invariant to other test passages and order",
            "unseen vocabulary excluded; training permutation invariant",
            "all three factories fit and return finite empty/Unicode passage scores",
            "required training groups rejected when missing",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/name-that-llm"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/baseline-v1/folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement-v3/topic"))
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log")],
    )
    with threadpool_limits(limits=2):
        checks = sanity_checks()
    (args.output_dir / "sanity_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    LOGGER.info("Sanity checks passed.")
    if args.sanity_only:
        return
    definition_path = args.output_dir / "experiment_definition.json"
    if definition_path.exists():
        raise FileExistsError("Experiment already defined; use a new output directory.")
    train, _ = load_data(args.data_dir)
    assignments = pd.read_csv(args.folds)
    if not assignments["id"].equals(train["id"]) or not assignments["prompt_id"].equals(train["prompt_id"]):
        raise ValueError("Saved folds do not align with data IDs and prompt IDs.")
    if assignments.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("Prompt leakage in saved folds.")
    if set(assignments["fold"]) != set(range(5)):
        raise ValueError("Expected five saved folds.")
    definition = {
        "experiment": "improvement-v3-topic",
        "defined_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": TOPIC_CANDIDATES,
        "labels": list(LABELS),
        "selection_metric": "pooled_oof_balanced_accuracy",
        "train_sha256": file_sha256(args.data_dir / "train.csv"),
        "folds_sha256": file_sha256(args.folds),
        "source_sha256": file_sha256(ROOT / "src/name_that_llm/topic_v3.py"),
        "style_source_sha256": file_sha256(ROOT / "src/name_that_llm/stylometry.py"),
        "runner_sha256": file_sha256(Path(__file__)),
        "versions": runtime_versions(),
        "validation": "Exact saved baseline grouped folds; transforms fit only training rows.",
        "inference": "Only text. Prompt groups used for training only. Never testbatch centering.",
        "limitations": [
            "Development folds reused from earlier model selection, not an untouched holdout.",
            "Prompt centering training/inference distributions differ; this is an experimental hypothesis.",
            "Group means do not exactly balance all labels because one Fable passage is absent.",
            "Masking uses a fixed function-word list, not the paper's BNC frequencies or POS tags.",
            "No claim these older human-authorship results establish current LLM-attribution SOTA.",
        ],
    }
    definition_path.write_text(json.dumps(definition, indent=2) + "\n")
    results = {}
    with threadpool_limits(limits=2):
        for name in TOPIC_CANDIDATES:
            destination = args.output_dir / name
            destination.mkdir()
            scores = np.full((len(train), len(LABELS)), np.nan)
            fold_metrics = []
            for fold in range(5):
                valid = np.flatnonzero(assignments["fold"].to_numpy() == fold)
                fitting = np.flatnonzero(assignments["fold"].to_numpy() != fold)
                assert not set(train.iloc[fitting]["prompt_id"]) & set(train.iloc[valid]["prompt_id"])
                model = build_topic_pipeline(name)
                kwargs = {"groups": train.iloc[fitting]["prompt_id"]} if getattr(model, "requires_groups", False) else {}
                started = time.perf_counter()
                LOGGER.info("Start %s fold=%d", name, fold)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(train.iloc[fitting]["text"], train.iloc[fitting]["model"], **kwargs)
                for warning in caught:
                    LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError("Classifier did not converge.")
                if tuple(model.classes_) != LABELS:
                    raise ValueError("Class order mismatch.")
                scores[valid] = model.decision_function(train.iloc[valid]["text"])
                pred = np.asarray(LABELS)[scores[valid].argmax(axis=1)]
                metric = {
                    "fold": fold,
                    "balanced_accuracy": float(balanced_accuracy_score(train.iloc[valid]["model"], pred)),
                    "seconds": time.perf_counter() - started,
                }
                fold_metrics.append(metric)
                LOGGER.info("Done %s %s", name, json.dumps(metric))
            if not np.isfinite(scores).all():
                raise ValueError("Incomplete/nonfinite OOF scores.")
            pred = np.asarray(LABELS)[scores.argmax(axis=1)]
            predictions = train[["id", "prompt_id", "model"]].copy()
            predictions["fold"] = assignments["fold"]
            predictions["prediction"] = pred
            predictions.to_csv(destination / "oof_predictions.csv", index=False)
            np.save(destination / "oof_scores.npy", scores)
            schema = {"labels": list(LABELS), "scores_kind": "decision_function", "row_order": "train.csv"}
            (destination / "score_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
            recalls = recall_score(train["model"], pred, labels=LABELS, average=None)
            pd.DataFrame({"model": LABELS, "recall": recalls}).to_csv(destination / "per_class_recall.csv", index=False)
            pd.DataFrame(confusion_matrix(train["model"], pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(
                destination / "confusion_matrix.csv", index_label="true_model"
            )
            results[name] = {
                "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(train["model"], pred)),
                "folds": fold_metrics, **schema,
            }
            (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
            LOGGER.info("Pooled %s %.6f", name, results[name]["pooled_oof_balanced_accuracy"])


if __name__ == "__main__":
    main()
