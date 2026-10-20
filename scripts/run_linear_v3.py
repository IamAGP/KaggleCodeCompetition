"""Evaluate exactly two frozen joint multinomial classifiers on saved folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, file_sha256, load_data, runtime_versions
from name_that_llm.linear_v3 import LINEAR_CANDIDATES, LINEAR_COMMON, build_linear_pipeline
from name_that_llm.stylometry import StyleFeatures

LOGGER = logging.getLogger("linear_v3")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def fixed_folds(train, path):
    assignments = pd.read_csv(path)
    for column in ("id", "prompt_id"):
        if assignments[column].tolist() != train[column].tolist():
            raise ValueError(f"Baseline {column} order mismatch.")
    if assignments.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("Prompt leakage.")
    if set(assignments["fold"]) != set(range(5)):
        raise ValueError("Expected five saved folds.")
    folds = []
    for fold in range(5):
        valid = np.flatnonzero(assignments["fold"].to_numpy() == fold)
        fitting = np.flatnonzero(assignments["fold"].to_numpy() != fold)
        if set(train.iloc[fitting]["model"]) != set(LABELS):
            raise ValueError("Missing training class.")
        if set(train.iloc[valid]["prompt_id"]) & set(train.iloc[fitting]["prompt_id"]):
            raise ValueError("Prompt leakage.")
        folds.append((fitting, valid))
    return assignments, folds


def check_fold_pipeline(pipeline, fitting_texts, validation_texts):
    features = dict(pipeline.named_steps["features"].transformer_list)
    vectorizer = features["char"]
    # Independent train-only reference catches full-data vocab/IDF fitting.
    reference = clone(vectorizer).fit(fitting_texts)
    if reference.vocabulary_ != vectorizer.vocabulary_:
        raise AssertionError("Vocabulary was not fitted on exactly the training rows.")
    np.testing.assert_array_equal(reference.idf_, vectorizer.idf_)
    scaler = features["style"].named_steps["scale"]
    expected = StyleFeatures().transform(fitting_texts)
    assert scaler.n_samples_seen_ == len(fitting_texts)
    np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(scaler.var_, expected.var(axis=0), rtol=1e-12, atol=1e-12)
    classifier = pipeline.named_steps["classifier"]
    if set(classifier.classes_) != set(LABELS):
        raise ValueError("Classifier label mismatch.")
    # One iteration count for the joint fit, rather than 18 independent fits.
    if classifier.n_iter_.shape != (1,):
        raise AssertionError("Expected one joint multinomial optimization.")
    probe = validation_texts.iloc[:3]
    probabilities = pipeline.predict_proba(probe)
    np.testing.assert_allclose(
        probabilities, softmax(pipeline.decision_function(probe), axis=1),
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        probabilities,
        np.vstack([pipeline.predict_proba([text]) for text in probe]),
        rtol=1e-12, atol=1e-12,
    )
    # Prediction must not learn a vocabulary or alter scaling statistics.
    unseen = ["\U0001fae0\U0001fae0\U0001fae0 validation-only symbols " * 2]
    pipeline.predict_proba(unseen)
    if vectorizer.vocabulary_ != reference.vocabulary_:
        raise AssertionError("Prediction changed vocabulary.")
    np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0), rtol=1e-13, atol=1e-13)
    return {
        "vocabulary_and_idf_match_independent_train_only_fit": True,
        "scaler_matches_training_only_statistics": True,
        "joint_multinomial_probabilities_equal_softmax": True,
        "single_joint_optimization_iterations": int(classifier.n_iter_[0]),
        "prediction_batch_invariance": True,
        "prediction_does_not_refit_features": True,
        "character_vocabulary_size": len(vectorizer.vocabulary_),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/name-that-llm"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/baseline-v1/folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement-v3/linear"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    definition_path = args.output_dir / "experiment_definition.json"
    if definition_path.exists():
        raise FileExistsError("Experiment already frozen; use a new output directory.")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log")],
    )
    train, _ = load_data(args.data_dir)
    assignments, folds = fixed_folds(train, args.folds)
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__), root / "src/name_that_llm/linear_v3.py",
        root / "src/name_that_llm/stylometry.py",
    ]
    source_hashes = {str(p.relative_to(root)): file_sha256(p) for p in sources}
    definition = {
        "experiment": "improvement-v3-linear-second-stage",
        "defined_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": LINEAR_CANDIDATES,
        "common_parameters": LINEAR_COMMON,
        "selection_metric": "pooled_oof_balanced_accuracy",
        "folds_sha256": file_sha256(args.folds),
        "train_sha256": file_sha256(args.data_dir / "train.csv"),
        "source_hashes": source_hashes,
        "labels": list(LABELS), "versions": runtime_versions(),
        "hypothesis": "Joint multinomial loss over character and style features can differ usefully from one-versus-rest SVM margins.",
        "validation": "Exact saved baseline prompt folds, fresh train-only pipeline per fold, reject ConvergenceWarning.",
        "budget": "Exactly two models, five folds each, two CPU threads; no adaptive tuning.",
        "limitations": [
            "Second-stage hypothesis follows a first stage with no candidate passing its improvement gate.",
            "Original prompt folds are reused development data, not an untouched holdout.",
            "Selection optimism remains; a separate frozen robustness check is needed before submission.",
        ],
    }
    write_json(definition_path, definition)
    LOGGER.info("Experiment frozen before fitting/scoring.")
    results = {}
    all_checks = []
    with threadpool_limits(limits=2):
        for name in LINEAR_CANDIDATES:
            destination = args.output_dir / name
            destination.mkdir()
            scores = np.full((len(train), len(LABELS)), np.nan)
            fold_metrics = []
            for fold, (fitting, valid) in enumerate(folds):
                started = time.perf_counter()
                pipeline = build_linear_pipeline(name)
                LOGGER.info("Start %s fold=%d training=%d validation=%d", name, fold, len(fitting), len(valid))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(train.iloc[fitting]["text"], train.iloc[fitting]["model"])
                fit_seconds = time.perf_counter() - started
                for warning in caught:
                    LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError("Joint multinomial optimizer did not converge.")
                checks = check_fold_pipeline(
                    pipeline, train.iloc[fitting]["text"], train.iloc[valid]["text"],
                )
                classes = list(pipeline.named_steps["classifier"].classes_)
                scores[valid] = pipeline.predict_proba(train.iloc[valid]["text"])[:, [classes.index(label) for label in LABELS]]
                np.testing.assert_allclose(scores[valid].sum(axis=1), 1., atol=1e-12)
                if not np.isfinite(scores[valid]).all() or (scores[valid] < 0).any():
                    raise ValueError("Invalid probabilities.")
                pred = np.asarray(LABELS)[np.argmax(scores[valid], axis=1)]
                fm = {
                    "fold": fold,
                    "balanced_accuracy": float(balanced_accuracy_score(train.iloc[valid]["model"], pred)),
                    "fit_seconds": fit_seconds,
                    "seconds_including_checks": time.perf_counter() - started,
                    "fitting_rows": len(fitting), "validation_rows": len(valid),
                    "optimizer_iterations": checks["single_joint_optimization_iterations"],
                }
                fold_metrics.append(fm)
                all_checks.append({"candidate": name, "fold": fold, **checks})
                write_json(args.output_dir / "checks.json", all_checks)
                LOGGER.info("Done %s %s", name, json.dumps(fm))
            if not np.isfinite(scores).all():
                raise ValueError("Incomplete or non-finite OOF scores.")
            pred = np.asarray(LABELS)[np.argmax(scores, axis=1)]
            predictions = train[["id", "prompt_id", "model"]].copy()
            predictions["fold"] = assignments["fold"]
            predictions["prediction"] = pred
            predictions.to_csv(destination / "oof_predictions.csv", index=False)
            np.save(destination / "oof_scores.npy", scores)
            write_json(destination / "score_schema.json", {
                "labels": list(LABELS), "scores_kind": "predict_proba",
                "row_order": "Exact train.csv order, aligned to oof_predictions.csv id",
                "shape": list(scores.shape), "source_hashes": source_hashes,
                "folds_sha256": definition["folds_sha256"],
            })
            pd.DataFrame({
                "model": LABELS,
                "recall": recall_score(train["model"], pred, labels=LABELS, average=None),
            }).to_csv(destination / "per_class_recall.csv", index=False)
            pd.DataFrame(
                confusion_matrix(train["model"], pred, labels=LABELS),
                index=LABELS, columns=LABELS,
            ).to_csv(destination / "confusion_matrix.csv", index_label="true_model")
            results[name] = {
                "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(train["model"], pred)),
                "folds": fold_metrics, "scores_kind": "predict_proba", "labels": list(LABELS),
                "total_fit_seconds": sum(fm["fit_seconds"] for fm in fold_metrics),
                "total_seconds_including_checks": sum(fm["seconds_including_checks"] for fm in fold_metrics),
            }
            write_json(destination / "metrics.json", results[name])
            write_json(args.output_dir / "metrics.json", results)
            LOGGER.info("Pooled %s %.6f", name, results[name]["pooled_oof_balanced_accuracy"])
    if any(file_sha256(root / path) != digest for path, digest in source_hashes.items()):
        raise AssertionError("Experiment source changed while training.")
    write_json(args.output_dir / "completion.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "both_recipes_complete": True, "source_hashes": source_hashes,
        "source_unchanged_during_run": True, "all_checks_passed": True,
    })


if __name__ == "__main__":
    main()
