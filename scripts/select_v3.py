"""Evaluate the small frozen v3 blend plan using strictly aligned OOF scores."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from name_that_llm.baseline import LABELS, file_sha256, load_data
from name_that_llm.ensemble_v2 import normalize_evidence
from name_that_llm.ensemble_v3 import combine_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("artifacts/improvement-v3/plan.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/improvement-v3/selection"))
    args = parser.parse_args(argv)
    root = Path("artifacts/improvement-v3")
    plan = json.loads(args.plan.read_text())
    out = args.output_dir
    out.mkdir(exist_ok=False)
    train, _ = load_data(Path("data/name-that-llm"))
    folds_path = Path("artifacts/baseline-v1/folds.csv")
    if plan["folds_sha256"] != file_sha256(folds_path):
        raise ValueError("Fold file differs from the frozen plan.")
    folds = pd.read_csv(folds_path)
    if folds["id"].tolist() != train["id"].tolist():
        raise ValueError("Fold IDs differ from training data.")
    if folds["prompt_id"].tolist() != train["prompt_id"].tolist():
        raise ValueError("Fold prompt IDs differ from training data.")
    if folds.groupby("prompt_id")["fold"].nunique().max() != 1:
        raise ValueError("A training prompt spans validation folds.")
    if set(folds["fold"]) != set(range(5)):
        raise ValueError("Expected the five original validation folds.")
    labels = np.asarray(LABELS)
    evidence = {}
    individual = {}
    sources = {}
    old = Path("artifacts/improvement-v2")
    payload = np.load(old / "sparse/char15_c4_oof.npz", allow_pickle=False)
    if payload["ids"].tolist() != train["id"].tolist() or tuple(payload["labels"]) != LABELS:
        raise ValueError("Saved sparse scores are misaligned.")
    evidence["sparse:char15_c4"] = normalize_evidence(payload["scores"])
    sources["sparse:char15_c4"] = file_sha256(old / "sparse/char15_c4_oof.npz")
    style_metrics = json.loads((old / "style/metrics.json").read_text())
    families = plan.get("branch_families", ["boosting", "topic"])
    entries = [("style", ["style_hist", "char_style_svm"])] + [
        (family, plan[family + "_candidates"]) for family in families
    ]
    for family, names in entries:
        for variant in names:
            folder = (old if family == "style" else root) / family / variant
            identity = pd.read_csv(folder / "oof_predictions.csv")
            if identity["id"].tolist() != train["id"].tolist():
                raise ValueError(f"Misaligned IDs: {folder}")
            if identity["fold"].tolist() != folds["fold"].tolist():
                raise ValueError(f"Misaligned folds: {folder}")
            if identity["model"].tolist() != train["model"].tolist():
                raise ValueError(f"Misaligned labels: {folder}")
            schema = (
                style_metrics[variant] if family == "style"
                else json.loads((folder / "score_schema.json").read_text())
            )
            if tuple(schema["labels"]) != LABELS:
                raise ValueError(f"Misaligned score columns: {folder}")
            raw = np.load(folder / "oof_scores.npy", allow_pickle=False)
            if raw.shape != (len(train), len(LABELS)):
                raise ValueError(f"Wrong score shape: {folder}")
            name = family + ":" + variant
            evidence[name] = normalize_evidence(raw, probabilities=schema["scores_kind"] == "predict_proba")
            sources[name] = file_sha256(folder / "oof_scores.npy")
    for name, values in evidence.items():
        individual[name] = float(balanced_accuracy_score(train["model"], labels[values.argmax(axis=1)]))
    best = {
        family: max((n for n in evidence if n.startswith(family + ":")), key=individual.get)
        for family in families
    }
    recipes = {}
    for name, definition in plan["blend_recipes"].items():
        recipes[name] = {
            "components": [best.get(component, component) for component in definition["components"]],
            "weights": definition["weights"],
            "pooling": definition["pooling"],
        }
    # Include the predeclared new individual models and the current v2.
    for name in evidence:
        if name.split(":", 1)[0] in best:
            recipes[name] = {"components": [name], "weights": [1], "pooling": "evidence"}
    results, combined = {}, {}
    for name, recipe in recipes.items():
        values = combine_evidence([evidence[c] for c in recipe["components"]],
                                  recipe["weights"], recipe["pooling"])
        pred = labels[values.argmax(axis=1)]
        combined[name] = values
        results[name] = {
            "recipe": recipe,
            "balanced_accuracy": float(balanced_accuracy_score(train["model"], pred)),
            "folds": [float(balanced_accuracy_score(
                train.loc[folds["fold"] == f, "model"], pred[folds["fold"] == f],
            )) for f in range(5)],
        }
    winner = max(results, key=lambda name: results[name]["balanced_accuracy"])
    selected = results[winner]
    pred = labels[combined[winner].argmax(axis=1)]
    result = {
        "selected_name": winner, **selected,
        "delta_from_v2": selected["balanced_accuracy"] - results["v2"]["balanced_accuracy"],
        "passes_development_gate": selected["balanced_accuracy"] - results["v2"]["balanced_accuracy"]
                                  >= plan["development_min_gain"],
        "individual_results": individual, "all_results": results,
        "plan_sha256": file_sha256(args.plan), "score_source_sha256": sources,
        "limitations": "Development selection on reused labeled data; not an unbiased holdout estimate.",
    }
    (out / "selected.json").write_text(json.dumps(result, indent=2) + "\n")
    np.save(out / "oof_scores.npy", combined[winner])
    oof = train[["id", "prompt_id", "model"]].copy()
    oof["fold"], oof["prediction"] = folds["fold"], pred
    oof.to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(train["model"], pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(
        out / "confusion_matrix.csv")
    pd.DataFrame({"model": LABELS, "recall": recall_score(train["model"], pred, labels=LABELS, average=None)}).to_csv(
        out / "per_class_recall.csv", index=False)
    print(json.dumps({k: result[k] for k in (
        "selected_name", "recipe", "balanced_accuracy", "delta_from_v2", "passes_development_gate",
    )}, indent=2))


if __name__ == "__main__":
    main()
