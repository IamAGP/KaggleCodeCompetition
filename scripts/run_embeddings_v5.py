"""Evaluate the two frozen v5 recipes with audited embeddings and v4 OOF."""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, runtime_versions
from name_that_llm.classifier_v5 import (
    CLASSIFIER_PARAMETERS, RECIPES, combine_evidence, evaluate_oof, source_hashes,
)
from name_that_llm.embeddings_v5 import load_embedding_cache, sha256_file as file_sha256
from select_v4 import evaluate, load_outputs, verify_producer


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def validate_plan(path):
    plan = read_json(path)
    if plan["classifier"] != CLASSIFIER_PARAMETERS or plan["candidates"] != RECIPES:
        raise ValueError("Plan and implemented classifier/candidates differ")
    if plan["baseline"]["plan_sha256"] != file_sha256("configs/name-that-llm/v4.json"):
        raise ValueError("Incumbent plan changed")
    frozen = Path("artifacts/improvement-v5/frozen-plan.json")
    if file_sha256(path) != file_sha256(frozen):
        raise ValueError("The frozen v5 plan changed")
    return plan


def load_incumbent(plan, mode):
    settings = plan[mode]
    run, folds_path = Path(settings["incumbent_run"]), Path(settings["folds"])
    if file_sha256(folds_path) != settings["folds_sha256"]:
        raise ValueError("Frozen evaluation folds changed")
    selection_path = Path("artifacts/improvement-v4/selection/selected.json")
    robustness_path = Path("artifacts/improvement-v4/robustness/check/metrics.json")
    selection, robustness = read_json(selection_path), read_json(robustness_path)
    if (file_sha256(selection_path) != "757452101b388879b9f284215fbe62a06300cb28c016f7eb2dda0575f43dffa3"
            or robustness["selected_sha256"] != file_sha256(selection_path)
            or not selection["passes_development_gate"] or not robustness["passes"]):
        raise ValueError("Scored incumbent selection is not the validated v4 winner")
    from name_that_llm.stacking_v4 import source_hashes as incumbent_hashes
    for result in (selection, robustness):
        expected = {"selected_name": "stack_v3_half", "recipe": {"stack_weight": .5, "v3_weight": .5},
                    "plan_sha256": plan["baseline"]["plan_sha256"], "source_hashes": incumbent_hashes(),
                    "selector_sha256": file_sha256(Path(__file__).with_name("select_v4.py")),
                    "evaluator_sha256": file_sha256(Path(__file__).with_name("run_nested_v4.py"))}
        for key, value in expected.items():
            if result[key] != value:
                raise ValueError(f"Scored incumbent recipe/provenance changed: {key}")
        for path, expected_hash in result["input_sha256"].items():
            if file_sha256(path) != expected_hash:
                raise ValueError(f"Incumbent selection input changed: {path}")
    producer = verify_producer(run, folds_path, Path("configs/name-that-llm/v4.json"))
    expected_kind = "audited_original_v2_v3_oof" if mode == "development" else "fresh_outer_fits"
    if producer["outer_prediction_source"] != expected_kind:
        raise ValueError("Unexpected incumbent OOF producer")
    train, folds, groups, stack, v3 = load_outputs(run, folds_path)
    incumbent = .5 * stack + .5 * v3
    metric, prediction = evaluate(incumbent, train.model.to_numpy(), folds)
    if not np.isclose(metric["balanced_accuracy"], plan["baseline"][f"{mode}_balanced_accuracy"], atol=1e-12, rtol=0):
        raise ValueError("Scored v4 validation baseline failed to reproduce")
    paths = [selection_path, robustness_path, folds_path, *[run / name for name in (
        "artifacts.json", "experiment_definition.json", "completion.json", "score_schema.json",
        "metrics.json", "stack_oof_probabilities.npy", "outer_oof_evidence.npy", "oof_predictions.csv")]]
    return train, folds, groups, incumbent, metric, prediction, paths


def verify_selection(selected_path):
    root = selected_path.parent
    completion = read_json(root / "completion.json")
    if (not completion["all_checks_passed"] or not completion["source_unchanged_during_run"]
            or completion["result_sha256"] != file_sha256(selected_path)):
        raise ValueError("Development selection does not match its completed evaluation")
    manifest = read_json(root / "artifacts.json")
    files = {entry["path"]: entry for entry in manifest["files"]}
    for name in (selected_path.name, "luar_oof_probabilities.npy", "fit_audit.json", "oof_predictions.csv",
                 "experiment_definition.json", "completion.json"):
        entry, path = files[name], root / name
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Development selection artifact changed: {name}")
    selected = read_json(selected_path)
    if selected["recipe"] != RECIPES.get(selected["selected_name"]):
        raise ValueError("Selected recipe differs from its frozen name")
    winner = max(selected["all_results"], key=lambda n: (selected["all_results"][n]["balanced_accuracy"], n == "luar"))
    if winner != selected["selected_name"]:
        raise ValueError("Selected recipe is not the frozen development winner")
    return selected


def diagnostics(labels, candidate, incumbent):
    return {"labels": list(LABELS),
            "per_class_recall": recall_score(labels, candidate, labels=LABELS, average=None).tolist(),
            "incumbent_per_class_recall": recall_score(labels, incumbent, labels=LABELS, average=None).tolist(),
            "confusion_matrix": confusion_matrix(labels, candidate, labels=LABELS).tolist(),
            "disagreed_rows": int(np.sum(candidate != incumbent)),
            "corrected_rows": int(np.sum((candidate == labels) & (incumbent != labels))),
            "new_error_rows": int(np.sum((candidate != labels) & (incumbent == labels)))}


def bootstrap_delta(labels, candidate, incumbent, groups, settings):
    unique = np.unique(groups)
    support = np.zeros((len(unique), len(LABELS)))
    difference = np.zeros_like(support)
    for gi, group in enumerate(unique):
        for ci, label in enumerate(LABELS):
            mask = (groups == group) & (labels == label)
            support[gi, ci] = mask.sum()
            difference[gi, ci] = np.sum(candidate[mask] == label) - np.sum(incumbent[mask] == label)
    draws = np.random.default_rng(settings["seed"]).integers(
        0, len(unique), (settings["bootstrap_draws"], len(unique)))
    gain = (difference[draws].sum(axis=1) / support[draws].sum(axis=1)).mean(axis=1)
    return np.quantile(gain, [.025, .975]).tolist()


def run(mode, plan_path, cache, model_dir, out, selected_path):
    plan = validate_plan(plan_path)
    sources_before = source_hashes()
    script_before = file_sha256(__file__)
    selected = None
    if mode == "robustness":
        selected = verify_selection(selected_path)
        if not selected["passes_development_gate"]:
            raise ValueError("Development gate failed; no robustness selection allowed")
        for key, expected in {"source_hashes": sources_before, "plan_sha256": file_sha256(plan_path),
                              "evaluator_sha256": script_before}.items():
            if selected[key] != expected:
                raise ValueError(f"Selected candidate provenance changed: {key}")
        for path, expected in selected["input_sha256"].items():
            if file_sha256(path) != expected:
                raise ValueError(f"Selected evaluation input changed: {path}")
    train, folds, groups, incumbent, baseline, baseline_prediction, input_paths = load_incumbent(plan, mode)
    embeddings, metadata = load_embedding_cache(
        cache, "train", train.id.tolist(), train.text.tolist(), model_dir=model_dir,
        plan_path=plan_path, source_path="data/name-that-llm/train.csv")
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "debug.log"), logging.StreamHandler()])
    input_paths.extend([plan_path, cache / "train-embeddings.npz", cache / "train-embeddings.json",
                        Path("data/name-that-llm/train.csv"), model_dir / "manifest.json",
                        Path("scripts/select_v4.py"), Path("scripts/extract_embeddings_v5.py")])
    input_hashes = {str(p): file_sha256(p) for p in input_paths}
    definition = {"mode": mode, "plan_sha256": file_sha256(plan_path), "source_hashes": sources_before,
                  "evaluator_sha256": script_before, "input_sha256": input_hashes,
                  "labels": list(LABELS), "classifier": CLASSIFIER_PARAMETERS,
                  "versions": runtime_versions(), "encoder_runtime": metadata["provenance"]["runtime"]}
    save_json(out / "experiment_definition.json", definition)
    probabilities, audit = evaluate_oof(embeddings, train.model.to_numpy(), folds, groups)
    np.save(out / "luar_oof_probabilities.npy", probabilities, allow_pickle=False)
    save_json(out / "fit_audit.json", audit)
    candidates = RECIPES if selected is None else {selected["selected_name"]: selected["recipe"]}
    results, predictions = {}, {}
    for name, recipe in candidates.items():
        scores = combine_evidence(probabilities, incumbent, recipe)
        results[name], predictions[name] = evaluate(scores, train.model.to_numpy(), folds)
    save_json(out / "diagnostics.json", {name: diagnostics(train.model.to_numpy(), prediction, baseline_prediction)
                                        for name, prediction in predictions.items()})
    chosen = max(results, key=lambda n: (results[n]["balanced_accuracy"], n == "luar"))
    metric = results[chosen]
    gain = metric["balanced_accuracy"] - baseline["balanced_accuracy"]
    wins = sum(a > b for a, b in zip(metric["folds"], baseline["folds"]))
    result = {**definition, "selected_name": chosen, "recipe": candidates[chosen], **metric,
              "baseline": baseline, "delta_from_v4": gain, "fold_wins": wins,
              "all_results": results, "limitations": plan["limitations"],
              "descriptive_paired_group_bootstrap_delta_95pct": bootstrap_delta(
                  train.model.to_numpy(), predictions[chosen], baseline_prediction, groups, plan["robustness"])}
    if selected is None:
        result["passes_development_gate"] = gain >= plan["development"]["min_gain"]
        filename = "selected.json"
    else:
        result["passes"] = gain >= plan["robustness"]["min_gain"] and wins >= plan["robustness"]["min_fold_wins"]
        result["selected_sha256"] = file_sha256(selected_path)
        filename = "metrics.json"
    if source_hashes() != sources_before or file_sha256(__file__) != script_before:
        raise ValueError("Evaluation source changed during the run")
    for path, expected in input_hashes.items():
        if file_sha256(path) != expected:
            raise ValueError(f"Evaluation input changed during the run: {path}")
    rows = train[["id", "prompt_id", "model"]].copy()
    rows["fold"], rows["group"], rows["v4"], rows["prediction"] = folds, groups, baseline_prediction, predictions[chosen]
    rows.to_csv(out / "oof_predictions.csv", index=False)
    save_json(out / filename, result)
    save_json(out / "completion.json", {"all_checks_passed": True, "source_unchanged_during_run": True,
                                        "result_sha256": file_sha256(out / filename)})
    save_json(out / "artifacts.json", {"files": [{"path": p.name, "bytes": p.stat().st_size,
                                                 "sha256": file_sha256(p)} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({key: result[key] for key in ("selected_name", "balanced_accuracy", "baseline", "delta_from_v4",
                                                 "fold_wins", "all_results", "descriptive_paired_group_bootstrap_delta_95pct")}, indent=2))
    print("PASS" if result.get("passes_development_gate", result.get("passes")) else "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("development", "robustness"))
    parser.add_argument("--plan", type=Path, default=Path("configs/name-that-llm/v5.json"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/improvement-v5/encoder"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/improvement-v5/model"))
    parser.add_argument("--selected", type=Path, default=Path("artifacts/improvement-v5/development/selected.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.plan, args.cache, args.model_dir, args.output_dir, args.selected)


if __name__ == "__main__":
    main()
