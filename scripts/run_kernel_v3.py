"""Evaluate two predeclared nonlinear kernels using saved prompt folds."""

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, file_sha256, load_data, runtime_versions
from name_that_llm.kernel_v3 import (
    KERNEL_CANDIDATES, bandwidth_sample_indices, build_kernel_pipeline,
    combine_style_char_kernel, median_style_gamma,
)

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("kernel_v3")


def sanity_checks():
    styles = np.asarray([[0., 0.], [1., 0.], [0., 2.]])
    chars = np.asarray([[1., 0.], [0., 1.], [0.6, 0.8]])
    expected_distances = np.asarray([[0., 1., 4.], [1., 0., 5.], [4., 5., 0.]])
    gamma = median_style_gamma(styles, np.arange(3))
    assert gamma == 0.25
    expected = 0.5 * np.exp(-0.25 * expected_distances) + 0.5 * (chars @ chars.T)
    actual = combine_style_char_kernel(
        styles, styles, gamma=gamma, char_left=chars, char_right=chars, char_weight=0.5,
    )
    np.testing.assert_allclose(expected, actual, atol=1e-12)
    np.testing.assert_allclose(actual, actual.T, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() > -1e-12
    np.testing.assert_allclose(np.diag(actual), 1.0, atol=1e-12)
    assert median_style_gamma(np.zeros((4, 3)), np.arange(4)) == 1.0
    texts = [f"synthetic passage {i}" for i in range(400)]
    perm = np.random.default_rng(101).permutation(len(texts))
    a = [texts[i] for i in bandwidth_sample_indices(texts)]
    reordered = [texts[i] for i in perm]
    b = [reordered[i] for i in bandwidth_sample_indices(reordered)]
    assert a == b and len(a) == 256
    texts = [
        "The red kite is here. Yet it is calm.",
        "The blue kite is here; indeed it is calm.",
        "The green kite is here — it is calm!",
        "The red boat is here. Yet it is calm.",
        "The blue boat is here; indeed it is calm.",
        "The green boat is here — it is calm!",
        "The red cloud is here. Yet it is calm.",
        "The blue cloud is here; indeed it is calm.",
        "The green cloud is here — it is calm!",
    ]
    labels = ["a", "b", "c"] * 3
    max_permutation_error = 0.0
    for name in KERNEL_CANDIDATES:
        model = build_kernel_pipeline(name).fit(texts, labels)
        query = "The uniqueqz boat is here."
        first = model.decision_function([query])
        np.testing.assert_allclose(first, model.decision_function([query, "Unrelated!"])[:1], atol=1e-12)
        np.testing.assert_allclose(first, model.decision_function(["Unrelated!", query])[1:], atol=1e-12)
        model.batch_size = 1
        np.testing.assert_allclose(first, model.decision_function([query, "Unrelated!"])[:1], atol=1e-12)
        gamma_before = model.gamma_
        means_before = model.scaler_.mean_.copy()
        assert np.isfinite(model.decision_function(["", "🙂", query])).all()
        np.testing.assert_array_equal(model.scaler_.mean_, means_before)
        assert model.gamma_ == gamma_before
        if model.char_weight:
            assert "uqz" not in model.vectorizer_.vocabulary_
        reversed_model = build_kernel_pipeline(name).fit(texts[::-1], labels[::-1])
        np.testing.assert_allclose(model.gamma_, reversed_model.gamma_, atol=1e-12)
        error = float(np.max(np.abs(first - reversed_model.decision_function([query]))))
        # LIBSVM stopping tolerance allows small changes under training reordering.
        assert error < 0.005, error
        max_permutation_error = max(max_permutation_error, error)
        assert model.decision_function([]).shape == (0, 3)
    return {
        "passed": True,
        "maximum_training_permutation_margin_difference": max_permutation_error,
        "checks": [
            "RBF bandwidth and additive kernel equal hand-calculated values",
            "kernel is symmetric, PSD and unit diagonal on nonempty normalized examples",
            "constant-style bandwidth falls back finitely",
            "256 text-hash bandwidth samples invariant under 400-row permutation",
            "real fit scores invariant to inference batch composition, order and chunk size",
            "inference cannot alter scaler or bandwidth, unseen vocabulary absent",
            "training permutation stable within LIBSVM tolerance",
            "finite empty/Unicode passage scores and valid empty batch shape",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/name-that-llm"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/baseline-v1/folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement-v3/kernel"))
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
        raise ValueError("Saved folds do not align with data.")
    if assignments.groupby("prompt_id")["fold"].nunique().max() != 1 or set(assignments.fold) != set(range(5)):
        raise ValueError("Expected five saved prompt-isolated folds.")
    definition = {
        "experiment": "improvement-v3-kernel-second-stage",
        "defined_utc": datetime.now(timezone.utc).isoformat(), "candidates": KERNEL_CANDIDATES,
        "labels": list(LABELS), "selection_metric": "pooled_oof_balanced_accuracy",
        "train_sha256": file_sha256(args.data_dir / "train.csv"),
        "folds_sha256": file_sha256(args.folds),
        "source_sha256": file_sha256(ROOT / "src/name_that_llm/kernel_v3.py"),
        "style_source_sha256": file_sha256(ROOT / "src/name_that_llm/stylometry.py"),
        "runner_sha256": file_sha256(Path(__file__)), "versions": runtime_versions(),
        "validation": "All vocabularies, scales and bandwidths fitted only fold-training rows.",
        "inference": "Text only, scores against fixed training features in batches of 256.",
        "limitations": [
            "Same development folds reused after earlier selection; optimism remains.",
            "Second-stage hypothesis followed a narrowly failing first-stage selection gate.",
            "Median bandwidth is an unvalidated heuristic, not a known optimum.",
            "Fixed kernel mixture is not learned multiple-kernel optimization.",
            "No current-state-of-the-art or leaderboard improvement claim.",
        ],
        "sources": [
            "https://arxiv.org/pdf/0912.3309",
            "https://arxiv.org/html/2507.00838v2",
            "https://github.com/kprzystalski/stylometry-llm",
            "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
            "https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html",
        ],
    }
    definition_path.write_text(json.dumps(definition, indent=2) + "\n")
    results = {}
    with threadpool_limits(limits=2):
        for name in KERNEL_CANDIDATES:
            destination = args.output_dir / name
            destination.mkdir()
            scores = np.full((len(train), len(LABELS)), np.nan)
            fold_metrics = []
            for fold in range(5):
                fitting = np.flatnonzero(assignments.fold.to_numpy() != fold)
                valid = np.flatnonzero(assignments.fold.to_numpy() == fold)
                assert not set(train.iloc[fitting].prompt_id) & set(train.iloc[valid].prompt_id)
                model = build_kernel_pipeline(name)
                started = time.perf_counter()
                LOGGER.info("Start %s fold=%d", name, fold)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(train.iloc[fitting].text, train.iloc[fitting].model)
                for warning in caught:
                    LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError("Classifier did not converge.")
                if tuple(model.classes_) != LABELS:
                    raise ValueError("Class order mismatch.")
                scores[valid] = model.decision_function(train.iloc[valid].text)
                pred = np.asarray(LABELS)[scores[valid].argmax(axis=1)]
                metric = {
                    "fold": fold, "balanced_accuracy": float(balanced_accuracy_score(train.iloc[valid].model, pred)),
                    "seconds": time.perf_counter() - started, "gamma": model.gamma_,
                    "bandwidth_training_samples": model.n_bandwidth_samples_,
                }
                fold_metrics.append(metric)
                LOGGER.info("Done %s %s", name, json.dumps(metric))
            if not np.isfinite(scores).all():
                raise ValueError("Incomplete/nonfinite OOF scores.")
            pred = np.asarray(LABELS)[scores.argmax(axis=1)]
            predictions = train[["id", "prompt_id", "model"]].copy()
            predictions["fold"] = assignments.fold
            predictions["prediction"] = pred
            predictions.to_csv(destination / "oof_predictions.csv", index=False)
            np.save(destination / "oof_scores.npy", scores)
            schema = {"labels": list(LABELS), "scores_kind": "decision_function", "row_order": "train.csv"}
            (destination / "score_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
            recalls = recall_score(train.model, pred, labels=LABELS, average=None)
            pd.DataFrame({"model": LABELS, "recall": recalls}).to_csv(destination / "per_class_recall.csv", index=False)
            pd.DataFrame(confusion_matrix(train.model, pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(
                destination / "confusion_matrix.csv", index_label="true_model",
            )
            results[name] = {
                "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(train.model, pred)),
                "folds": fold_metrics, **schema,
            }
            (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
            LOGGER.info("Pooled %s %.6f", name, results[name]["pooled_oof_balanced_accuracy"])


if __name__ == "__main__":
    main()
