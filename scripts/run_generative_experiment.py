"""Evaluate three predeclared generative candidates on baseline-v1's saved folds."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, file_sha256, runtime_versions, validate_frame
from name_that_llm.generative import GENERATIVE_CONFIGS, build_generative


LOGGER = logging.getLogger("name_that_llm.generative")
SOURCES = [
    {
        "url": "https://arxiv.org/abs/1412.5335",
        "title": "Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews",
        "use": "Primary motivation for complementary generative/discriminative scores and NB-SVM weighting; read full paper.",
        "repository": "https://github.com/mesnilgr/iclr15",
        "nbsvm_repository": "https://github.com/mesnilgr/nbsvm",
        "limitation": "Historical sentiment research, not current LLM attribution performance evidence.",
    },
    {
        "url": "https://aclanthology.org/P12-2018/",
        "title": "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification",
        "use": "Equations for binarized MNB and NB log-count ratios. Our multiclass character variant omits optional interpolation.",
        "repository": "https://github.com/sidaw/nbsvm",
        "maintenance": "Author-maintained historical reference, not a current runtime dependency; implementation is original code using sklearn.",
    },
    {
        "url": "https://scikit-learn.org/stable/modules/naive_bayes.html",
        "title": "scikit-learn Naive Bayes documentation",
        "use": "MNB smoothing and normalized ComplementNB; maintained implementation used locally.",
    },
]
SCORE_TYPES = {
    "char_mnb": "Uncalibrated average log joint: (sum binary_ngram * class_log_probability + uniform_log_prior) / present_feature_count; larger is better.",
    "char_cnb": "Uncalibrated normalized complement weight sum / present_feature_count; larger is better. Not log probabilities.",
    "char_nbsvm": "Raw one-versus-rest LinearSVC decision margins after NB log-count-ratio feature weighting; larger is better.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/name-that-llm"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/baseline-v1/folds.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement-v2/generative"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "plan.json").exists():
        raise FileExistsError("An experiment already exists; choose a separate output directory.")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log")],
    )
    train = pd.read_csv(args.data_dir / "train.csv", dtype=str, keep_default_na=False)
    validate_frame(train, training=True)
    folds = pd.read_csv(args.folds, dtype={"id": str, "prompt_id": str, "fold": int})
    if not train[["id", "prompt_id"]].equals(folds[["id", "prompt_id"]]):
        raise ValueError("Saved folds must exactly match training IDs, prompt IDs and row order.")
    if sorted(folds["fold"].unique().tolist()) != list(range(5)):
        raise ValueError("Expected exactly the five saved baseline folds.")
    if folds.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("Saved folds split prompt groups.")
    summary = {
        "experiment": "improvement-v2-generative",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configs": [asdict(c) for c in GENERATIVE_CONFIGS],
        "labels": list(LABELS),
        "train_sha256": file_sha256(args.data_dir / "train.csv"),
        "folds_sha256": file_sha256(args.folds),
        "sources": SOURCES,
        "versions": runtime_versions(),
        "selection_metric": "pooled_oof_balanced_accuracy",
        "score_semantics": "Uncalibrated per-class decision scores in LABELS order; NB score divided by number of present features.",
        "score_types": SCORE_TYPES,
        "limitations": [
            "Three candidates selected on fixed folds reused from baseline-v1; not an unbiased final estimate.",
            "Character features motivated by label-visible reviews of all training data.",
            "Related topics can appear under different prompt IDs.",
            "No probability calibration, hidden data, quotas or prompt/id features.",
        ],
    }
    (args.output_dir / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["candidates"] = {}
    for config in GENERATIVE_CONFIGS:
        candidate_dir = args.output_dir / config.variant
        candidate_dir.mkdir()
        oof_scores = np.full((len(train), len(LABELS)), np.nan)
        fold_metrics = []
        for fold in range(5):
            fitting = np.flatnonzero(folds["fold"].to_numpy() != fold)
            held_out = np.flatnonzero(folds["fold"].to_numpy() == fold)
            if set(train.iloc[fitting]["prompt_id"]) & set(train.iloc[held_out]["prompt_id"]):
                raise ValueError("Prompt leakage.")
            LOGGER.info("Start %s fold=%s", config.variant, fold)
            started = time.perf_counter()
            model = build_generative(config).fit(
                train.iloc[fitting]["text"], train.iloc[fitting]["model"],
            )
            if tuple(model.classes_) != LABELS:
                raise ValueError("Class order mismatch.")
            scores = model.decision_function(train.iloc[held_out]["text"])
            oof_scores[held_out] = scores
            prediction = np.asarray(LABELS)[scores.argmax(axis=1)]
            metric = {
                "fold": fold,
                "balanced_accuracy": float(balanced_accuracy_score(
                    train.iloc[held_out]["model"], prediction,
                )),
                "train_rows": len(fitting), "validation_rows": len(held_out),
                "features": model.n_features_in_,
                "solver_iterations": model.solver_iterations_,
                "seconds": time.perf_counter() - started,
            }
            fold_metrics.append(metric)
            LOGGER.info("Done %s %s", config.variant, json.dumps(metric))
            (candidate_dir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2) + "\n")
        if not np.isfinite(oof_scores).all():
            raise ValueError("OOF scores incomplete.")
        predictions = np.asarray(LABELS)[oof_scores.argmax(axis=1)]
        result = train[["id", "prompt_id", "model"]].copy()
        result["fold"] = folds["fold"]
        result["prediction"] = predictions
        result.to_csv(candidate_dir / "oof_predictions.csv", index=False)
        # NPY rows match train.csv exactly; JSON explicitly retains columns.
        np.save(candidate_dir / "oof_scores.npy", oof_scores)
        pd.DataFrame(oof_scores, columns=LABELS).to_csv(candidate_dir / "oof_scores.csv", index=False)
        (candidate_dir / "score_schema.json").write_text(json.dumps({
            "row_order": "data/name-that-llm/train.csv",
            "row_id_file": "oof_predictions.csv",
            "columns": list(LABELS), "shape": list(oof_scores.shape),
            "semantics": SCORE_TYPES[config.variant],
        }, indent=2) + "\n")
        matrix = confusion_matrix(train["model"], predictions, labels=LABELS)
        pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis("true_model").to_csv(
            candidate_dir / "confusion_matrix.csv",
        )
        recalls = recall_score(train["model"], predictions, labels=LABELS, average=None)
        pd.DataFrame({"model": LABELS, "recall": recalls, "support": matrix.sum(axis=1)}).to_csv(
            candidate_dir / "per_class_recall.csv", index=False,
        )
        metrics = {
            "config": asdict(config),
            "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(train["model"], predictions)),
            "fold_mean": float(np.mean([r["balanced_accuracy"] for r in fold_metrics])),
            "folds": fold_metrics,
        }
        summary["candidates"][config.variant] = metrics
        (candidate_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
        LOGGER.info("COMPLETE %s pooled_oof_balanced_accuracy=%.8f",
                    config.variant, metrics["pooled_oof_balanced_accuracy"])
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    summary["selected_variant"] = max(
        summary["candidates"], key=lambda k: summary["candidates"][k]["pooled_oof_balanced_accuracy"],
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
