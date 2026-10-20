"""Choose from predeclared individual candidates and a small fixed blend set."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from name_that_llm.baseline import LABELS, load_data
from name_that_llm.ensemble_v2 import normalize_evidence


def main():
    root = Path("artifacts/improvement-v2")
    output = root / "selection"
    output.mkdir(exist_ok=True)
    train, _ = load_data(Path("data/name-that-llm"))
    y = train["model"]
    evidence = {}
    individual_scores = {}
    for path in sorted((root / "sparse").glob("*_oof.npz")):
        payload = np.load(path, allow_pickle=False)
        assert payload["ids"].tolist() == train["id"].tolist()
        assert tuple(payload["labels"]) == LABELS
        name = "sparse:" + path.stem.removesuffix("_oof")
        raw = payload["scores"]
        evidence[name] = normalize_evidence(raw)
        individual_scores[name] = float(balanced_accuracy_score(y, np.asarray(LABELS)[raw.argmax(axis=1)]))
    for family in ("style", "generative"):
        for path in sorted((root / family).glob("*/oof_scores.npy")):
            folder = path.parent
            name = family + ":" + folder.name
            identity = pd.read_csv(folder / "oof_predictions.csv", dtype={"id": str})
            assert identity["id"].tolist() == train["id"].tolist()
            metadata = (
                json.loads((root / family / "metrics.json").read_text())[folder.name]
                if family == "style" else json.loads((folder / "metrics.json").read_text())
            )
            schema_labels = (
                metadata["labels"] if family == "style"
                else json.loads((folder / "score_schema.json").read_text())["columns"]
            )
            if tuple(schema_labels) != LABELS:
                raise ValueError(f"Score columns are misaligned for {name}.")
            raw = np.load(path, allow_pickle=False)
            probabilities = metadata.get("scores_kind") == "predict_proba"
            evidence[name] = normalize_evidence(raw, probabilities=probabilities)
            individual_scores[name] = float(balanced_accuracy_score(y, np.asarray(LABELS)[raw.argmax(axis=1)]))
    if len(evidence) != 10:
        raise ValueError(f"Expected all 10 predeclared individual candidates; got {len(evidence)}.")
    best = {
        family: max((name for name in evidence if name.startswith(family + ":")),
                    key=lambda name: individual_scores[name])
        for family in ("sparse", "style", "generative")
    }
    recipes = {
        name: {"components": [name], "weights": [1.0]}
        for name in evidence
    }
    # These weights are fixed; there is no trained meta-model or per-class adjustment.
    for other in [name for name in evidence if name.startswith("style:")] + [best["generative"]]:
        for weight in (1, 2):
            name = f"{best['sparse']}+{other}@{weight}:1"
            recipes[name] = {"components": [best["sparse"], other], "weights": [weight, 1]}
    recipes["three_branch_equal"] = {
        "components": [best[family] for family in ("sparse", "style", "generative")],
        "weights": [1, 1, 1],
    }
    (output / "blend_plan.json").write_text(json.dumps({
        "normalization": "Within each passage: log probabilities if applicable, then center and divide by class-score SD.",
        "recipes": recipes,
        "selection_metric": "pooled_oof_balanced_accuracy",
        "secondary_robustness_check": "Freeze winner before seed1729 related-prompt-grouped evaluation on reused development data.",
    }, indent=2) + "\n")
    results = {}
    for name, recipe in recipes.items():
        weights = np.asarray(recipe["weights"], dtype=float)
        combined = sum(evidence[c] * w for c, w in zip(recipe["components"], weights)) / weights.sum()
        prediction = np.asarray(LABELS)[combined.argmax(axis=1)]
        results[name] = {"recipe": recipe, "balanced_accuracy": float(balanced_accuracy_score(y, prediction))}
    selected = max(results, key=lambda name: results[name]["balanced_accuracy"])
    recipe = results[selected]["recipe"]
    weights = np.asarray(recipe["weights"], dtype=float)
    scores = sum(evidence[c] * w for c, w in zip(recipe["components"], weights)) / weights.sum()
    pred = np.asarray(LABELS)[scores.argmax(axis=1)]
    np.save(output / "oof_scores.npy", scores)
    oof = train[["id", "prompt_id", "model"]].copy()
    oof["prediction"] = pred
    oof.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y, pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(output / "confusion_matrix.csv")
    pd.DataFrame({"model": LABELS, "recall": recall_score(y, pred, labels=LABELS, average=None)}).to_csv(
        output / "per_class_recall.csv", index=False)
    final = {
        "selected_name": selected, "recipe": recipe,
        "balanced_accuracy": results[selected]["balanced_accuracy"],
        "individual_results": individual_scores, "all_results": results,
        "limitation": "Development-set model selection across 10 individual candidates and 9 fixed blends; score is not an unbiased test estimate.",
    }
    (output / "selected.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps({"selected": selected, "score": final["balanced_accuracy"], "recipe": recipe}, indent=2))


if __name__ == "__main__":
    main()
