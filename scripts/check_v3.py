"""Compare a frozen v3 recipe with v2 on a new related-prompt-group split."""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from name_that_llm.baseline import LABELS, file_sha256, load_data
from name_that_llm.ensemble_v2 import EvidenceEnsemble
from name_that_llm.ensemble_v3 import V3Ensemble
from check_improvement import RELATED_PROMPTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("artifacts/improvement-v3/plan.json"))
    parser.add_argument("--selected", type=Path, default=Path("artifacts/improvement-v3/selection/selected.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement-v3/robustness"))
    args = parser.parse_args()
    root = Path("artifacts/improvement-v3")
    plan = json.loads(args.plan.read_text())
    selected_path = args.selected
    selected = json.loads(selected_path.read_text())
    if selected["plan_sha256"] != file_sha256(args.plan):
        raise RuntimeError("Selected recipe was evaluated under a different plan.")
    if not selected["passes_development_gate"]:
        raise RuntimeError("Candidate did not pass the predeclared development gate.")
    out = args.output_dir
    out.mkdir(exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    train, _ = load_data(Path("data/name-that-llm"))
    baseline = json.loads(Path("artifacts/improvement-v2/selection/selected.json").read_text())["recipe"]
    mapping = {p: p for p in train["prompt_id"].unique()}
    for first, second in RELATED_PROMPTS:
        mapping[second] = first
    groups = train["prompt_id"].map(mapping)
    settings = plan["robustness"]
    splits = list(GroupKFold(n_splits=settings["n_splits"], shuffle=True,
                            random_state=settings["seed"]).split(train, train["model"], groups))
    frozen = {
        "defined_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings, "recipe": selected["recipe"], "baseline_recipe": baseline,
        "selected_sha256": file_sha256(selected_path), "plan_sha256": file_sha256(args.plan),
        "source_sha256": {str(p): file_sha256(p) for p in Path("src/name_that_llm").glob("*.py")},
        "related_pairs": RELATED_PROMPTS, "group_count": groups.nunique(),
    }
    (out / "plan.json").write_text(json.dumps(frozen, indent=2) + "\n")
    oof = train[["id", "prompt_id", "model"]].copy()
    oof["group"], oof["fold"] = groups, -1
    predictions = {name: np.empty(len(train), dtype=object) for name in ("v2", "v3")}
    results = {"folds": []}
    with threadpool_limits(limits=2):
        for fold, (fit, valid) in enumerate(splits):
            if not set(groups.iloc[fit]).isdisjoint(groups.iloc[valid]):
                raise ValueError("Related groups overlap.")
            oof.loc[valid, "fold"] = fold
            metrics = {"fold": fold}
            for name in predictions:
                start = time.perf_counter()
                logging.info("Start fold=%s candidate=%s", fold, name)
                model = EvidenceEnsemble(**baseline) if name == "v2" else V3Ensemble(**selected["recipe"])
                kwargs = {} if name == "v2" else {"groups": train.iloc[fit]["prompt_id"].to_numpy()}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(train.iloc[fit]["text"], train.iloc[fit]["model"], **kwargs)
                for warning in caught:
                    logging.warning("%s: %s", warning.category.__name__, warning.message)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError("Unconverged candidate.")
                predictions[name][valid] = model.predict(train.iloc[valid]["text"])
                metrics[name] = float(balanced_accuracy_score(train.iloc[valid]["model"], predictions[name][valid]))
                metrics[name + "_seconds"] = time.perf_counter() - start
                logging.info("Done fold=%s candidate=%s balanced_accuracy=%.6f seconds=%.2f",
                             fold, name, metrics[name], metrics[name + "_seconds"])
            results["folds"].append(metrics)
    for name in predictions:
        oof[name] = predictions[name]
        results[name] = float(balanced_accuracy_score(train["model"], predictions[name]))
    oof.to_csv(out / "oof_predictions.csv", index=False)
    results["delta"] = results["v3"] - results["v2"]
    results["fold_wins"] = sum(f["v3"] > f["v2"] for f in results["folds"])
    results["passes"] = (results["delta"] >= settings["min_pooled_gain"]
                         and results["fold_wins"] >= settings["min_fold_wins"])
    unique_groups = sorted(groups.unique())
    supports = np.zeros((len(unique_groups), len(LABELS)))
    correct = {name: np.zeros_like(supports) for name in predictions}
    for gi, group in enumerate(unique_groups):
        for ci, label in enumerate(LABELS):
            subset = (groups == group) & (train["model"] == label)
            supports[gi, ci] = subset.sum()
            for name in predictions:
                correct[name][gi, ci] = np.sum(predictions[name][subset] == label)
    draws = np.random.default_rng(settings["seed"]).integers(0, len(unique_groups), (5000, len(unique_groups)))
    totals = supports[draws].sum(axis=1)
    delta = ((correct["v3"][draws].sum(axis=1) - correct["v2"][draws].sum(axis=1)) / totals).mean(axis=1)
    results["paired_group_bootstrap_delta_95pct"] = np.quantile(delta, [.025, .975]).tolist()
    results["uncertainty_note"] = (
        "Descriptive interval; development data is reused and training folds overlap. "
        "The recipe was frozen before this comparison; no retuning on these results."
    )
    (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
