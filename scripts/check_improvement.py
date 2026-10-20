"""One frozen comparison on a second, stricter prompt-group split."""

import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from name_that_llm.baseline import LABELS, BaselineConfig, build_pipeline, fit_checked, file_sha256, load_data
from name_that_llm.ensemble_v2 import EvidenceEnsemble


RELATED_PROMPTS = (
    ("p010", "p082"), ("p100", "p146"), ("p017", "p148"),
    ("p128", "p136"), ("p008", "p131"), ("p055", "p058"),
    ("p004", "p111"),
)


def main():
    root = Path("artifacts/improvement-v2")
    out = root / "robustness"
    out.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")],
    )
    train, _ = load_data(Path("data/name-that-llm"))
    selected_path = root / "selection/selected.json"
    selected = json.loads(selected_path.read_text())
    mapping = {prompt: prompt for prompt in train["prompt_id"].unique()}
    for first, second in RELATED_PROMPTS:
        if first not in mapping or second not in mapping:
            raise ValueError("Related prompt not found.")
        mapping[second] = first
    groups = train["prompt_id"].map(mapping)
    splits = list(GroupKFold(n_splits=5, shuffle=True, random_state=1729).split(
        train["text"], train["model"], groups=groups,
    ))
    plan = {
        "seed": 1729, "n_splits": 5,
        "related_pairs": RELATED_PROMPTS, "group_count": groups.nunique(),
        "selected_recipe": selected["recipe"],
        "selected_file_sha256": file_sha256(selected_path),
        "baseline": "Unmodified baseline-v1 char3-5 TFIDF LinearSVC C1",
        "purpose": "Post-selection robustness check, not a never-inspected holdout.",
        "acceptance": "Frozen v2 must beat baseline pooled balanced accuracy and in at least 4/5 folds.",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    oof = train[["id", "prompt_id", "model"]].copy()
    oof["group"] = groups
    oof["fold"] = -1
    predictions = {name: np.empty(len(train), dtype=object) for name in ("baseline", "improved")}
    results = {"folds": []}
    for fold, (fitting, valid) in enumerate(splits):
        if not set(groups.iloc[fitting]).isdisjoint(groups.iloc[valid]):
            raise ValueError("Related prompt group crosses a fold.")
        oof.loc[valid, "fold"] = fold
        metrics = {"fold": fold}
        for name in predictions:
            start = time.perf_counter()
            model = build_pipeline(BaselineConfig()) if name == "baseline" else EvidenceEnsemble(**selected["recipe"])
            logging.info("Start fold=%d %s", fold, name)
            fit_checked(model, train.iloc[fitting]["text"], train.iloc[fitting]["model"])
            predictions[name][valid] = model.predict(train.iloc[valid]["text"])
            metrics[name] = float(balanced_accuracy_score(train.iloc[valid]["model"], predictions[name][valid]))
            metrics[name + "_seconds"] = time.perf_counter() - start
            logging.info("Done fold=%d %s score=%.6f seconds=%.2f",
                         fold, name, metrics[name], metrics[name + "_seconds"])
        results["folds"].append(metrics)
    for name in predictions:
        oof[name] = predictions[name]
        results[name] = float(balanced_accuracy_score(train["model"], predictions[name]))
    oof.to_csv(out / "oof_predictions.csv", index=False)
    results["delta"] = results["improved"] - results["baseline"]
    results["fold_wins"] = sum(r["improved"] > r["baseline"] for r in results["folds"])
    results["passes"] = results["delta"] > 0 and results["fold_wins"] >= 4
    # Resample entire related-prompt groups and retain both models' paired results.
    group_names = sorted(groups.unique())
    supports = np.zeros((len(group_names), len(LABELS)))
    correct = {name: np.zeros_like(supports) for name in predictions}
    for gi, group in enumerate(group_names):
        for ci, label in enumerate(LABELS):
            subset = (groups == group) & (train["model"] == label)
            supports[gi, ci] = subset.sum()
            for name in predictions:
                correct[name][gi, ci] = np.sum(predictions[name][subset] == label)
    rng = np.random.default_rng(1729)
    draws = rng.integers(0, len(group_names), size=(5000, len(group_names)))
    totals = supports[draws].sum(axis=1)
    deltas = (
        correct["improved"][draws].sum(axis=1) / totals
        - correct["baseline"][draws].sum(axis=1) / totals
    ).mean(axis=1)
    results["paired_group_bootstrap_delta_95pct"] = np.quantile(deltas, [.025, .975]).tolist()
    results["uncertainty_note"] = (
        "Descriptive paired group-bootstrap interval; overlapping training folds and "
        "prior development on the full labeled dataset limit independence."
    )
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
