"""Run the three predeclared style-v2 candidates on immutable baseline folds."""

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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, file_sha256, load_data, runtime_versions
from name_that_llm.stylometry import CANDIDATES, StyleFeatures, build_style_pipeline

LOGGER = logging.getLogger("style_v2")


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
            raise ValueError("Missing class.")
        folds.append((fitting, valid))
    return assignments, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/name-that-llm"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/baseline-v1/folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement-v2/style"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    definition_path = args.output_dir / "experiment_definition.json"
    if definition_path.exists():
        raise FileExistsError("Experiment already defined; use a new output directory.")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log")],
    )
    train, _ = load_data(args.data_dir)
    assignments, folds = fixed_folds(train, args.folds)
    definition = {
        "experiment": "improvement-v2-style", "defined_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": CANDIDATES, "selection_metric": "pooled_oof_balanced_accuracy",
        "folds_sha256": file_sha256(args.folds),
        "train_sha256": file_sha256(args.data_dir / "train.csv"),
        "style_source_sha256": file_sha256(Path(__file__).resolve().parents[1] / "src/name_that_llm/stylometry.py"),
        "script_sha256": file_sha256(Path(__file__)), "labels": list(LABELS),
        "feature_count": len(StyleFeatures().get_feature_names_out()),
        "versions": runtime_versions(),
        "validation": "Exact saved baseline prompt folds; all learned transforms fitted inside each fold.",
        "limitations": [
            "Small candidate selection introduces optimism; no untouched final holdout.",
            "Handcrafted features motivated by previous label-visible dataset review.",
            "Related topics may still cross prompt folds.",
            "Sentence boundaries are regex approximations.",
        ],
        "sources": [
            "https://arxiv.org/html/2104.08530v2",
            "https://github.com/malikaltakrori/Topic-Confusion-for-authorship-attribution-EMNLP-2021-",
            "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html",
            "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        ],
    }
    definition_path.write_text(json.dumps(definition, indent=2) + "\n")
    (args.output_dir / "feature_names.json").write_text(
        json.dumps(StyleFeatures().get_feature_names_out().tolist(), indent=2) + "\n"
    )
    results = {}
    # Small CPU budget and avoid BLAS oversubscription against other branches.
    with threadpool_limits(limits=2):
        for name in CANDIDATES:
            destination = args.output_dir / name
            destination.mkdir()
            scores = np.full((len(train), len(LABELS)), np.nan)
            fold_metrics = []
            for fold, (fitting, valid) in enumerate(folds):
                started = time.perf_counter()
                pipeline = build_style_pipeline(name)
                LOGGER.info("Start %s fold=%d", name, fold)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pipeline.fit(train.iloc[fitting]["text"], train.iloc[fitting]["model"])
                for warning in caught:
                    LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError("Classifier did not converge.")
                classes = pipeline.named_steps["classifier"].classes_
                order = [list(classes).index(label) for label in LABELS]
                kind = "decision_function" if name == "char_style_svm" else "predict_proba"
                scores[valid] = getattr(pipeline, kind)(train.iloc[valid]["text"])[:, order]
                pred = np.asarray(LABELS)[np.argmax(scores[valid], axis=1)]
                fm = {
                    "fold": fold, "balanced_accuracy": float(balanced_accuracy_score(train.iloc[valid]["model"], pred)),
                    "seconds": time.perf_counter() - started,
                }
                fold_metrics.append(fm)
                LOGGER.info("Done %s %s", name, json.dumps(fm))
            if not np.isfinite(scores).all():
                raise ValueError("Incomplete or non-finite OOF scores.")
            pred = np.asarray(LABELS)[np.argmax(scores, axis=1)]
            predictions = train[["id", "prompt_id", "model"]].copy()
            predictions["fold"] = assignments["fold"]
            predictions["prediction"] = pred
            predictions.to_csv(destination / "oof_predictions.csv", index=False)
            score_frame = pd.DataFrame(scores, columns=LABELS)
            score_frame.insert(0, "id", train["id"].to_numpy())
            score_frame.to_csv(destination / "oof_scores.csv", index=False)
            np.save(destination / "oof_scores.npy", scores)
            recalls = recall_score(train["model"], pred, labels=LABELS, average=None)
            pd.DataFrame({"model": LABELS, "recall": recalls}).to_csv(destination / "per_class_recall.csv", index=False)
            pd.DataFrame(confusion_matrix(train["model"], pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(
                destination / "confusion_matrix.csv", index_label="true_model"
            )
            results[name] = {
                "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(train["model"], pred)),
                "folds": fold_metrics, "scores_kind": kind, "labels": list(LABELS),
                "score_order": "train.csv order; CSV includes IDs",
                "feature_count": definition["feature_count"],
            }
            (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
            LOGGER.info("Pooled %s %.6f", name, results[name]["pooled_oof_balanced_accuracy"])


if __name__ == "__main__":
    main()
