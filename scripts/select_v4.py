"""Select two frozen stacking candidates and check the winner on new folds."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from name_that_llm.baseline import LABELS, file_sha256, load_data
from name_that_llm.ensemble_v2 import normalize_evidence
from name_that_llm.stacking_v4 import (
    COMPONENTS, META_PARAMETERS, source_hashes, validate_fold_assignment,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def verify_producer(run, folds_path, plan_path):
    files = {row["path"]: row for row in read_json(run / "artifacts.json")["files"]}
    required = (
        "stack_oof_probabilities.npy", "outer_oof_evidence.npy", "oof_predictions.csv",
        "experiment_definition.json", "completion.json", "score_schema.json", "metrics.json",
    )
    for name in required:
        record, path = files[name], run / name
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Producer output changed: {name}")
    definition = read_json(run / "experiment_definition.json")
    expected = {
        "plan_sha256": file_sha256(plan_path),
        "source_hashes": source_hashes(),
        "script_sha256": file_sha256(Path(__file__).with_name("run_nested_v4.py")),
        "folds_sha256": file_sha256(folds_path),
        "train_sha256": file_sha256(Path("data/name-that-llm/train.csv")),
        "labels": list(LABELS), "meta_parameters": META_PARAMETERS,
    }
    for key, value in expected.items():
        if definition[key] != value:
            raise ValueError(f"Producer provenance mismatch: {key}")
    if definition["recipe"]["components"] != list(COMPONENTS):
        raise ValueError("Producer component order differs.")
    schema = read_json(run / "score_schema.json")
    for key, value in {
        "labels": list(LABELS), "components": list(COMPONENTS),
        "source_hashes": expected["source_hashes"], "folds_sha256": expected["folds_sha256"],
    }.items():
        if schema[key] != value:
            raise ValueError(f"Output schema mismatch: {key}")
    completed = read_json(run / "completion.json")
    if not completed["all_checks_passed"] or not completed["source_unchanged_during_run"]:
        raise ValueError("Nested evaluation did not complete successfully.")
    if completed["metrics"] != read_json(run / "metrics.json"):
        raise ValueError("Completion metrics differ.")
    return definition


def load_outputs(run, folds_path):
    train, _ = load_data(Path("data/name-that-llm"))
    folds = pd.read_csv(folds_path)
    rows = pd.read_csv(run / "oof_predictions.csv")
    for key in ("id", "prompt_id", "model"):
        if not np.array_equal(rows[key], train[key]):
            raise ValueError(f"Evaluation row alignment mismatch: {key}")
    for key in ("id", "prompt_id", "fold"):
        if not np.array_equal(rows[key], folds[key]):
            raise ValueError(f"Evaluation fold alignment mismatch: {key}")
    if "group" in folds and ("group" not in rows or not np.array_equal(rows["group"], folds["group"])):
        raise ValueError("Evaluation related-group alignment mismatch.")
    groups = folds["group"] if "group" in folds else folds["prompt_id"]
    assignment = validate_fold_assignment(folds["fold"], train["model"], groups)
    probabilities = np.load(run / "stack_oof_probabilities.npy", allow_pickle=False)
    evidence = np.load(run / "outer_oof_evidence.npy", allow_pickle=False)
    if probabilities.shape != (len(train), len(LABELS)):
        raise ValueError("Stack probability shape mismatch.")
    if evidence.shape != (len(train), len(COMPONENTS), len(LABELS)):
        raise ValueError("Outer evidence shape mismatch.")
    if not np.isfinite(evidence).all():
        raise ValueError("Non-finite outer evidence.")
    stack = normalize_evidence(probabilities, probabilities=True)
    if not np.array_equal(rows["prediction"], np.asarray(LABELS)[stack.argmax(axis=1)]):
        raise ValueError("Saved stack predictions differ from probabilities.")
    # The v3 score is an equal mean of its three normalized base outputs.
    baseline = evidence[:, [0, 1, 3], :].mean(axis=1)
    return train, assignment, np.asarray(groups), stack, baseline


def evaluate(evidence, labels, assignment):
    predictions = np.asarray(LABELS)[evidence.argmax(axis=1)]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "folds": [
            float(balanced_accuracy_score(labels[assignment == f], predictions[assignment == f]))
            for f in range(5)
        ],
    }, predictions


def prepare_robustness(plan_path, selected_path, out):
    plan, selected = read_json(plan_path), read_json(selected_path)
    if selected["plan_sha256"] != file_sha256(plan_path):
        raise ValueError("Selection belongs to a different plan.")
    if not selected["passes_development_gate"]:
        raise ValueError("The development gate failed.")
    if selected["source_hashes"] != source_hashes():
        raise ValueError("Model source changed after development.")
    for key, path in (
        ("selector_sha256", Path(__file__)),
        ("evaluator_sha256", Path(__file__).with_name("run_nested_v4.py")),
    ):
        if selected[key] != file_sha256(path):
            raise ValueError(f"Validation code changed after development: {key}")
    train, _ = load_data(Path("data/name-that-llm"))
    settings = plan["robustness"]
    mapping = {p: p for p in train["prompt_id"].unique()}
    for first, second in settings["related_prompt_pairs"]:
        mapping[second] = first
    groups = train["prompt_id"].map(mapping)
    rows = train[["id", "prompt_id", "model"]].copy()
    rows["group"], rows["fold"] = groups, -1
    splitter = GroupKFold(n_splits=5, shuffle=True, random_state=settings["seed"])
    for fold, (_, valid) in enumerate(splitter.split(train, train["model"], groups)):
        rows.loc[valid, "fold"] = fold
    validate_fold_assignment(rows["fold"], train["model"], groups)
    out.mkdir(parents=True, exist_ok=False)
    rows.to_csv(out / "folds.csv", index=False)
    save_json(out / "frozen.json", {
        "plan_sha256": file_sha256(plan_path),
        "selected_sha256": file_sha256(selected_path),
        "source_hashes": source_hashes(),
        "selector_sha256": file_sha256(Path(__file__)),
        "evaluator_sha256": file_sha256(Path(__file__).with_name("run_nested_v4.py")),
        "folds_sha256": file_sha256(out / "folds.csv"),
        "settings": settings,
        "group_count": int(groups.nunique()),
    })


def select(plan_path, run, folds_path, out, selected_path=None):
    plan = read_json(plan_path)
    producer = verify_producer(run, folds_path, plan_path)
    train, folds, groups, stack, baseline = load_outputs(run, folds_path)
    baseline_result, baseline_predictions = evaluate(baseline, train["model"].to_numpy(), folds)
    baseline_result["name"] = "v3"
    if selected_path is None:
        if producer["outer_prediction_source"] != "audited_original_v2_v3_oof":
            raise ValueError("Development must use the audited original outer predictions.")
        if file_sha256(folds_path) != plan["folds_sha256"]:
            raise ValueError("Development folds changed.")
        if not np.isclose(baseline_result["balanced_accuracy"],
                          plan["baseline"]["development_balanced_accuracy"], atol=1e-12, rtol=0):
            raise ValueError("Frozen v3 baseline did not reproduce.")
        candidates = plan["candidates"]
    else:
        if producer["outer_prediction_source"] != "fresh_outer_fits":
            raise ValueError("Robustness must refit all outer components.")
        selected = read_json(selected_path)
        frozen = read_json(folds_path.parent / "frozen.json")
        expected = {
            "plan_sha256": file_sha256(plan_path),
            "selected_sha256": file_sha256(selected_path),
            "source_hashes": source_hashes(),
            "folds_sha256": file_sha256(folds_path),
            "selector_sha256": file_sha256(Path(__file__)),
            "evaluator_sha256": file_sha256(Path(__file__).with_name("run_nested_v4.py")),
        }
        for key, value in expected.items():
            if frozen[key] != value:
                raise ValueError(f"Frozen robustness comparison changed: {key}")
        if not selected["passes_development_gate"]:
            raise ValueError("Development gate failed.")
        candidates = {selected["selected_name"]: selected["recipe"]}
    results, predictions = {}, {}
    for name, recipe in candidates.items():
        scores = recipe["stack_weight"] * stack + recipe["v3_weight"] * baseline
        results[name], predictions[name] = evaluate(scores, train["model"].to_numpy(), folds)
    chosen = max(results, key=lambda name: (results[name]["balanced_accuracy"], name == "stack"))
    metric = results[chosen]
    delta = metric["balanced_accuracy"] - baseline_result["balanced_accuracy"]
    wins = sum(a > b for a, b in zip(metric["folds"], baseline_result["folds"]))
    result = {
        "selected_name": chosen, "recipe": candidates[chosen],
        **metric, "baseline": baseline_result, "delta_from_v3": delta,
        "fold_wins": wins, "all_results": results,
        "plan_sha256": file_sha256(plan_path), "source_hashes": source_hashes(),
        "selector_sha256": file_sha256(Path(__file__)),
        "evaluator_sha256": producer["script_sha256"],
        "input_sha256": {str(p): file_sha256(p) for p in (
            folds_path, run / "stack_oof_probabilities.npy",
            run / "outer_oof_evidence.npy", run / "oof_predictions.csv",
            run / "artifacts.json", run / "experiment_definition.json",
            run / "completion.json", run / "score_schema.json", run / "metrics.json",
        )},
        "limitations": plan["limitations"],
    }
    if selected_path is None:
        result["passes_development_gate"] = delta >= plan["development_min_gain"]
        filename = "selected.json"
    else:
        settings = plan["robustness"]
        result["passes"] = (delta >= settings["min_pooled_gain"]
                            and wins >= settings["min_fold_wins"])
        result["selected_sha256"] = file_sha256(selected_path)
        unique = np.unique(groups)
        support = np.zeros((len(unique), len(LABELS)))
        correct_delta = np.zeros_like(support)
        labels = train["model"].to_numpy()
        for gi, group in enumerate(unique):
            for ci, label in enumerate(LABELS):
                mask = (groups == group) & (labels == label)
                support[gi, ci] = mask.sum()
                correct_delta[gi, ci] = (
                    np.sum(predictions[chosen][mask] == label)
                    - np.sum(baseline_predictions[mask] == label)
                )
        draws = np.random.default_rng(settings["seed"]).integers(0, len(unique), (5000, len(unique)))
        bootstrap = (correct_delta[draws].sum(axis=1) / support[draws].sum(axis=1)).mean(axis=1)
        result["descriptive_paired_group_bootstrap_delta_95pct"] = np.quantile(
            bootstrap, [.025, .975]
        ).tolist()
        filename = "metrics.json"
    out.mkdir(parents=True, exist_ok=False)
    saved_rows = train[["id", "prompt_id", "model"]].copy()
    saved_rows["fold"], saved_rows["group"] = folds, groups
    saved_rows["v3"], saved_rows["prediction"] = baseline_predictions, predictions[chosen]
    saved_rows.to_csv(out / "oof_predictions.csv", index=False)
    save_json(out / filename, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("select", "prepare-robustness", "check"))
    parser.add_argument("--plan", type=Path, default=Path("configs/name-that-llm/v4.json"))
    parser.add_argument("--run", type=Path, default=Path("artifacts/improvement-v4/nested"))
    parser.add_argument("--folds", type=Path, default=Path("artifacts/baseline-v1/folds.csv"))
    parser.add_argument("--selected", type=Path, default=Path("artifacts/improvement-v4/selection/selected.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare-robustness":
        prepare_robustness(args.plan, args.selected, args.output_dir)
    else:
        select(args.plan, args.run, args.folds, args.output_dir,
               args.selected if args.mode == "check" else None)


if __name__ == "__main__":
    main()
