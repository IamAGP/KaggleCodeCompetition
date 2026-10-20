"""Evaluate four predeclared sparse candidates on the frozen baseline folds."""

from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from name_that_llm.baseline import LABELS, file_sha256, fit_checked, load_data, runtime_versions
from name_that_llm.sparse_v2 import SPARSE_CANDIDATES, build_sparse_pipeline


def main():
    output = Path("artifacts/improvement-v2/sparse")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "run.log")],
    )
    train, _ = load_data(Path("data/name-that-llm"))
    folds_path = Path("artifacts/baseline-v1/folds.csv")
    folds = pd.read_csv(folds_path, dtype={"id": str, "prompt_id": str})
    if not folds["id"].equals(train["id"]) or not folds["prompt_id"].equals(train["prompt_id"]):
        raise ValueError("Frozen fold rows do not match training data.")
    plan = {
        "candidates": [asdict(c) for c in SPARSE_CANDIDATES],
        "folds_sha256": file_sha256(folds_path),
        "train_sha256": file_sha256(Path("data/name-that-llm/train.csv")),
        "versions": runtime_versions(),
        "selection_metric": "pooled_oof_balanced_accuracy",
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    results = {}
    for config in SPARSE_CANDIDATES:
        predictions = np.empty(len(train), dtype=object)
        scores = np.zeros((len(train), len(LABELS)))
        fold_metrics = []
        for fold in sorted(folds["fold"].unique()):
            held_out = np.flatnonzero(folds["fold"].to_numpy() == fold)
            fitting = np.flatnonzero(folds["fold"].to_numpy() != fold)
            assert set(train.iloc[fitting]["prompt_id"]).isdisjoint(train.iloc[held_out]["prompt_id"])
            started = time.perf_counter()
            logging.info("Start %s fold=%s", config.name, fold)
            model = fit_checked(
                build_sparse_pipeline(config), train.iloc[fitting]["text"], train.iloc[fitting]["model"],
            )
            assert tuple(model.classes_) == LABELS
            scores[held_out] = model.decision_function(train.iloc[held_out]["text"])
            predictions[held_out] = np.asarray(LABELS)[scores[held_out].argmax(axis=1)]
            result = {
                "fold": int(fold),
                "balanced_accuracy": float(balanced_accuracy_score(train.iloc[held_out]["model"], predictions[held_out])),
                "seconds": time.perf_counter() - started,
                "solver_iterations": int(model.named_steps["classifier"].n_iter_),
                "feature_count": len(model.named_steps["features"].vocabulary_),
            }
            fold_metrics.append(result)
            logging.info("Done %s %s", config.name, json.dumps(result))
        np.savez_compressed(
            output / f"{config.name}_oof.npz",
            ids=train["id"].to_numpy(dtype=str),
            labels=np.asarray(LABELS), scores=scores,
            predictions=predictions.astype(str),
        )
        pd.DataFrame(confusion_matrix(train["model"], predictions, labels=LABELS),
                     index=LABELS, columns=LABELS).to_csv(output / f"{config.name}_confusion.csv")
        results[config.name] = {
            "config": asdict(config),
            "balanced_accuracy": float(balanced_accuracy_score(train["model"], predictions)),
            "folds": fold_metrics,
            "per_class_recall": dict(zip(LABELS, recall_score(train["model"], predictions, labels=LABELS, average=None))),
        }
        (output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
        logging.info("Overall %s: %.6f", config.name, results[config.name]["balanced_accuracy"])


if __name__ == "__main__":
    main()
