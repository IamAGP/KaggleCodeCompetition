"""Run three predeclared LightGBM recipes on the immutable baseline folds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
import warnings

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, file_sha256, load_data, runtime_versions
from name_that_llm.boosting_v3 import (
    BOOSTING_CANDIDATES, BOOSTING_COMMON, build_boosting_pipeline,
)
from name_that_llm.stylometry import StyleFeatures

LOGGER = logging.getLogger("boosting_v3")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def fixed_folds(train, path):
    assignments = pd.read_csv(path)
    if assignments["id"].tolist() != train["id"].tolist():
        raise ValueError("Baseline fold IDs/order mismatch.")
    if assignments["prompt_id"].tolist() != train["prompt_id"].tolist():
        raise ValueError("Baseline prompt IDs/order mismatch.")
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


def check_style_features():
    """Check actual feature values under label/batch changes."""
    passages = ["", "One sentence. Another—different sentence!", "UPPER 'quotes' 2.0\nabc"]
    other = ["Unseen prose with totally different punctuation?!", "東京"]
    first = StyleFeatures().fit(passages, ["a", "b", "c"])
    second = StyleFeatures().fit(other, ["x", "y"])
    reference = first.transform(passages)
    np.testing.assert_array_equal(reference, second.transform(passages))
    np.testing.assert_array_equal(
        reference, np.vstack([first.transform([text]) for text in passages]),
    )
    np.testing.assert_array_equal(
        reference, first.transform(passages + other)[:len(passages)],
    )
    assert reference.shape == (3, 465) and np.isfinite(reference).all()
    assert not vars(first) and not vars(second), "Style fit unexpectedly learned state."
    return {
        "label_and_fit_corpus_independence": True,
        "feature_batch_invariance": True,
        "finite_empty_and_unicode_inputs": True,
        "feature_count": 465,
    }


def ordered_probabilities(pipeline, texts):
    classes = list(pipeline.named_steps["classifier"].classes_)
    if set(classes) != set(LABELS):
        raise ValueError("Classifier label mismatch.")
    with warnings.catch_warnings():
        # LightGBM 4.6 attaches auto names to ndarray features; sklearn 1.7
        # then warns about the same ndarray input during prediction.
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        probabilities = pipeline.predict_proba(texts)[:, [classes.index(label) for label in LABELS]]
    if not np.isfinite(probabilities).all() or (probabilities < 0.).any():
        raise ValueError("Invalid probabilities.")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-12)
    return probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/name-that-llm"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/baseline-v1/folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement-v3/boosting"))
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
        Path(__file__), root / "src/name_that_llm/boosting_v3.py",
        root / "src/name_that_llm/stylometry.py",
    ]
    source_hashes = {str(p.relative_to(root)): file_sha256(p) for p in sources}
    definition = {
        "experiment": "improvement-v3-boosting",
        "defined_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": BOOSTING_CANDIDATES,
        "common_classifier_parameters": BOOSTING_COMMON,
        "selection_metric": "pooled_oof_balanced_accuracy",
        "folds_sha256": file_sha256(args.folds),
        "train_sha256": file_sha256(args.data_dir / "train.csv"),
        "source_hashes": source_hashes,
        "labels": list(LABELS),
        "feature_count": len(StyleFeatures().get_feature_names_out()),
        "versions": {**runtime_versions(), "lightgbm": lightgbm.__version__},
        "validation": "Exact saved baseline prompt folds; fresh pipeline per fold; no validation early stopping.",
        "features": "Existing 465 deterministic text-only StyleFeatures; no IDs, prompt IDs or target encodings.",
        "budget": "Exactly three recipes, five folds each, two CPU threads; no adaptive search.",
        "limitations": [
            "Candidate selection reuses development data and is optimistic.",
            "Related topics can cross the original prompt folds.",
            "Style features were motivated by earlier label-visible dataset review.",
            "LightGBM version/platform changes can change numerical results.",
        ],
    }
    write_json(definition_path, definition)
    LOGGER.info("Experiment frozen before scoring; LightGBM %s", lightgbm.__version__)
    checks = check_style_features()
    checks["folds"] = []
    write_json(args.output_dir / "checks.json", checks)
    reference = pd.read_csv(root / "artifacts/improvement-v2/style/style_hist/oof_predictions.csv")
    if reference["id"].tolist() != train["id"].tolist() or reference["fold"].tolist() != assignments["fold"].tolist():
        raise ValueError("Stored v2 baseline alignment mismatch.")
    baseline_score = float(balanced_accuracy_score(train["model"], reference["prediction"]))
    results = {}
    with threadpool_limits(limits=2):
        for name in BOOSTING_CANDIDATES:
            destination = args.output_dir / name
            destination.mkdir()
            scores = np.full((len(train), len(LABELS)), np.nan)
            fold_metrics = []
            for fold, (fitting, valid) in enumerate(folds):
                started = time.perf_counter()
                pipeline = build_boosting_pipeline(name)
                LOGGER.info("Start %s fold=%d training=%d validation=%d", name, fold, len(fitting), len(valid))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(train.iloc[fitting]["text"], train.iloc[fitting]["model"])
                for warning in caught:
                    LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
                # With row bagging disabled, the first fitted tree must contain
                # exactly the training rows; an accidental full-data fit fails.
                first_tree = pipeline.named_steps["classifier"].booster_.dump_model(
                    start_iteration=0, num_iteration=1,
                )["tree_info"][0]["tree_structure"]
                root_count = first_tree.get("internal_count", first_tree.get("leaf_count"))
                if root_count != len(fitting):
                    raise AssertionError(f"Tree saw {root_count} rows; expected {len(fitting)}.")
                scores[valid] = ordered_probabilities(pipeline, train.iloc[valid]["text"])
                probe = train.iloc[valid[:3]]["text"]
                individual = np.vstack([ordered_probabilities(pipeline, [text]) for text in probe])
                np.testing.assert_allclose(scores[valid[:3]], individual, rtol=1e-12, atol=1e-12)
                pred = np.asarray(LABELS)[np.argmax(scores[valid], axis=1)]
                fm = {
                    "fold": fold,
                    "balanced_accuracy": float(balanced_accuracy_score(train.iloc[valid]["model"], pred)),
                    "seconds": time.perf_counter() - started,
                    "fitting_rows": len(fitting),
                    "validation_rows": len(valid),
                }
                fold_metrics.append(fm)
                checks["folds"].append({
                    "candidate": name, "fold": fold,
                    "tree_root_training_count": int(root_count),
                    "expected_training_count": len(fitting),
                    "probability_batch_invariance": True,
                    "normalized_finite_probabilities": True,
                })
                write_json(args.output_dir / "checks.json", checks)
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
                "shape": list(scores.shape),
                "source_hashes": source_hashes,
                "folds_sha256": definition["folds_sha256"],
                "lightgbm_version": lightgbm.__version__,
            })
            pd.DataFrame({
                "model": LABELS,
                "recall": recall_score(train["model"], pred, labels=LABELS, average=None),
            }).to_csv(destination / "per_class_recall.csv", index=False)
            pd.DataFrame(
                confusion_matrix(train["model"], pred, labels=LABELS),
                index=LABELS, columns=LABELS,
            ).to_csv(destination / "confusion_matrix.csv", index_label="true_model")
            pooled = float(balanced_accuracy_score(train["model"], pred))
            results[name] = {
                "pooled_oof_balanced_accuracy": pooled,
                "folds": fold_metrics, "scores_kind": "predict_proba", "labels": list(LABELS),
                "stored_v2_style_hist_balanced_accuracy": baseline_score,
                "difference_from_v2_style_hist": pooled - baseline_score,
                "total_fit_and_validation_seconds": sum(fm["seconds"] for fm in fold_metrics),
            }
            write_json(destination / "metrics.json", results[name])
            write_json(args.output_dir / "metrics.json", results)
            LOGGER.info("Pooled %s %.6f; stored style_hist %.6f", name, pooled, baseline_score)
    if any(file_sha256(root / path) != digest for path, digest in source_hashes.items()):
        raise AssertionError("Experiment source changed while training.")
    write_json(args.output_dir / "completion.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "all_three_recipes_complete": True,
        "source_hashes": source_hashes,
        "source_unchanged_during_run": True,
        "all_checks_passed": True,
    })


if __name__ == "__main__":
    main()
