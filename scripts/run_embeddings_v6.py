"""Run the separately frozen v6 probes on audited, unchanged v5 vectors."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, runtime_versions
from name_that_llm.classifier_v6 import (
    MODEL_DEFINITIONS, RECIPES, choose_recipe, combine_evidence, evaluate_oof, source_hashes,
)
from name_that_llm.embeddings_v5 import load_embedding_cache, sha256_file as file_sha256
from run_embeddings_v5 import (
    bootstrap_delta, diagnostics, load_incumbent, read_json,
    validate_plan as validate_embedding_plan, verify_selection as verify_predecessor_selection,
)
from select_v4 import evaluate

FROZEN_PLAN_SHA256 = "3c05b07738659cb17c2f6a4d23d198dc281abb8ebf782d59bddcfd23a2df9ec0"


def save_json(path, value):
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(serialized)


def validate_plan(path):
    frozen = Path("artifacts/improvement-v6/frozen-plan.json")
    if file_sha256(path) != FROZEN_PLAN_SHA256 or file_sha256(frozen) != FROZEN_PLAN_SHA256:
        raise ValueError("The separately frozen v6 plan changed")
    plan = read_json(path)
    if plan["models"] != MODEL_DEFINITIONS or plan["candidates"] != RECIPES:
        raise ValueError("Plan and implemented models/recipes differ")
    if list(plan["candidates"]) != list(RECIPES):
        raise ValueError("Frozen recipe declaration order changed")
    embedding_plan_path = Path(plan["embedding_plan"])
    if file_sha256(embedding_plan_path) != plan["embedding_plan_sha256"]:
        raise ValueError("Frozen v5 embedding plan changed")
    embedding_plan = validate_embedding_plan(embedding_plan_path)
    for key in ("baseline", "development", "robustness"):
        if plan[key] != embedding_plan[key]:
            raise ValueError(f"V6 changed the frozen evaluation contract: {key}")
    predecessor = Path(plan["predecessor_result"]["path"])
    if file_sha256(predecessor) != plan["predecessor_result"]["sha256"]:
        raise ValueError("Recorded v5 predecessor result changed")
    previous = verify_predecessor_selection(predecessor)
    if previous["passes_development_gate"] or previous["balanced_accuracy"] != plan["predecessor_result"]["development_balanced_accuracy"]:
        raise ValueError("V6 no longer follows the recorded failed v5 result")
    return plan


def verify_selection(selected_path):
    """Verify all development artifacts before allowing the selected robustness fit."""
    root = selected_path.parent
    manifest = read_json(root / "artifacts.json")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    required = {selected_path.name, "fit_audit.json", "oof_predictions.csv", "diagnostics.json",
                "experiment_definition.json", "completion.json", "score_schema.json",
                *[f"{name}_oof_evidence.npy" for name in MODEL_DEFINITIONS]}
    if not required <= set(entries) or len(entries) != len(manifest["files"]):
        raise ValueError("Development manifest lacks required artifacts or has duplicates")
    for name, entry in entries.items():
        if Path(name).name != name:
            raise ValueError("Invalid development artifact path")
        path = root / name
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Development artifact changed: {name}")
    schema = read_json(root / "score_schema.json")
    if schema["labels"] != list(LABELS) or set(schema["models"]) != set(MODEL_DEFINITIONS):
        raise ValueError("Development evidence class/model schema changed")
    for name in MODEL_DEFINITIONS:
        scores = np.load(root / f"{name}_oof_evidence.npy", allow_pickle=False)
        expected_shape = (len(schema["row_ids"]), len(LABELS))
        if scores.shape != expected_shape or not np.isfinite(scores).all():
            raise ValueError("Development OOF evidence shape/values are invalid")
        std = scores.std(axis=1)
        if not np.allclose(scores.mean(axis=1), 0, atol=1e-8) or not np.all(np.isclose(std, 1, atol=1e-8) | (std == 0)):
            raise ValueError("Development OOF evidence is not normalized per row")
    completion = read_json(root / "completion.json")
    if (not completion["all_checks_passed"] or not completion["source_unchanged_during_run"]
            or not completion["inputs_unchanged_during_run"] or completion["result_sha256"] != file_sha256(selected_path)):
        raise ValueError("Development selection does not match its completed evaluation")
    selected = read_json(selected_path)
    if selected["mode"] != "development" or set(selected["all_results"]) != set(RECIPES):
        raise ValueError("Development did not evaluate exactly the four frozen recipes")
    if selected["recipe"] != RECIPES.get(selected["selected_name"]):
        raise ValueError("Selected recipe differs from its frozen name")
    if choose_recipe(selected["all_results"]) != selected["selected_name"]:
        raise ValueError("Selected recipe is not the frozen development winner")
    if selected["balanced_accuracy"] != selected["all_results"][selected["selected_name"]]["balanced_accuracy"]:
        raise ValueError("Selected score differs from candidate results")
    gain = selected["balanced_accuracy"] - selected["baseline"]["balanced_accuracy"]
    if not np.isclose(gain, selected["delta_from_v4"], atol=1e-12, rtol=0):
        raise ValueError("Selected development gain is inconsistent")
    return selected


def run(mode, plan_path, cache, model_dir, out, selected_path):
    if mode not in ("development", "robustness"):
        raise ValueError("Unknown evaluation mode")
    plan = validate_plan(plan_path)
    sources_before, script_before = source_hashes(), file_sha256(__file__)
    selected = None
    selection_inputs = []
    if mode == "robustness":
        selected = verify_selection(selected_path)
        if not selected["passes_development_gate"] or selected["delta_from_v4"] < plan["development"]["min_gain"]:
            raise ValueError("Development gate failed; no robustness evaluation allowed")
        for key, expected in {"source_hashes": sources_before, "plan_sha256": file_sha256(plan_path),
                              "evaluator_sha256": script_before}.items():
            if selected[key] != expected:
                raise ValueError(f"Selected development provenance changed: {key}")
        for path, expected in selected["input_sha256"].items():
            if file_sha256(path) != expected:
                raise ValueError(f"Selected evaluation input changed: {path}")
        selection_manifest = selected_path.parent / "artifacts.json"
        selection_inputs = [selection_manifest, *[selected_path.parent / entry["path"]
                            for entry in read_json(selection_manifest)["files"]]]
    train, folds, groups, incumbent, baseline, baseline_prediction, input_paths = load_incumbent(plan, mode)
    if file_sha256(cache / "train-embeddings.npz") != plan["train_embedding_sha256"]:
        raise ValueError("V6 must reuse exactly the recorded v5 training embeddings")
    embeddings, metadata = load_embedding_cache(
        cache, "train", train.id.tolist(), train.text.tolist(), model_dir=model_dir,
        plan_path=Path(plan["embedding_plan"]), source_path="data/name-that-llm/train.csv")
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "debug.log"), logging.StreamHandler()])
    input_paths.extend([plan_path, Path("artifacts/improvement-v6/frozen-plan.json"), Path(plan["embedding_plan"]),
                        Path(plan["predecessor_result"]["path"]), cache / "train-embeddings.npz", cache / "train-embeddings.json",
                        Path("data/name-that-llm/train.csv"), model_dir / "manifest.json",
                        *[model_dir / item["path"] for item in metadata["provenance"]["model_files"]],
                        *selection_inputs])
    input_hashes = {str(path): file_sha256(path) for path in input_paths}
    candidates = RECIPES if selected is None else {selected["selected_name"]: selected["recipe"]}
    model_names = list(dict.fromkeys(recipe["model"] for recipe in candidates.values()))
    definition = {"mode": mode, "plan_sha256": file_sha256(plan_path), "source_hashes": sources_before,
                  "evaluator_sha256": script_before, "input_sha256": input_hashes, "labels": list(LABELS),
                  "models_fitted": {name: MODEL_DEFINITIONS[name] for name in model_names},
                  "candidate_recipes": candidates, "versions": runtime_versions(),
                  "embedding_provenance": metadata["provenance"]}
    save_json(out / "experiment_definition.json", definition)
    save_json(out / "score_schema.json", {
        "labels": list(LABELS), "row_ids": train.id.tolist(),
        "row_order": "exact source train CSV order", "normalization": "per-row mean zero and unit standard deviation",
        "models": {name: {"score_kind": "normalized_log_probabilities" if MODEL_DEFINITIONS[name]["kind"] == "logistic"
                          else "normalized_ovr_decision_function", "shape": [len(train), len(LABELS)]} for name in model_names}})
    evidence, audit = {}, {}
    for name in model_names:
        evidence[name], audit[name] = evaluate_oof(embeddings, train.model.to_numpy(), folds, groups, name)
        np.save(out / f"{name}_oof_evidence.npy", evidence[name], allow_pickle=False)
    save_json(out / "fit_audit.json", audit)
    results, predictions = {}, {}
    for name, recipe in candidates.items():
        scores = combine_evidence(evidence[recipe["model"]], incumbent, recipe)
        results[name], predictions[name] = evaluate(scores, train.model.to_numpy(), folds)
    save_json(out / "diagnostics.json", {name: diagnostics(train.model.to_numpy(), prediction, baseline_prediction)
                                        for name, prediction in predictions.items()})
    chosen = choose_recipe(results)
    metric = results[chosen]
    gain = metric["balanced_accuracy"] - baseline["balanced_accuracy"]
    wins = sum(a > b for a, b in zip(metric["folds"], baseline["folds"]))
    result = {**definition, "selected_name": chosen, "recipe": candidates[chosen], **metric,
              "baseline": baseline, "delta_from_v4": gain, "fold_wins": wins, "all_results": results,
              "limitations": plan["limitations"], "descriptive_paired_group_bootstrap_delta_95pct": bootstrap_delta(
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
    for name, prediction in predictions.items():
        rows[name] = prediction
    rows.to_csv(out / "oof_predictions.csv", index=False)
    save_json(out / filename, result)
    save_json(out / "completion.json", {"all_checks_passed": True, "source_unchanged_during_run": True,
                                        "inputs_unchanged_during_run": True, "result_sha256": file_sha256(out / filename)})
    save_json(out / "artifacts.json", {"files": [{"path": path.name, "bytes": path.stat().st_size,
                                                 "sha256": file_sha256(path)} for path in sorted(out.iterdir()) if path.is_file()]})
    print(json.dumps({key: result[key] for key in ("selected_name", "balanced_accuracy", "delta_from_v4", "fold_wins",
                                                 "all_results", "descriptive_paired_group_bootstrap_delta_95pct")}, indent=2))
    print("PASS" if result.get("passes_development_gate", result.get("passes")) else "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("development", "robustness"))
    parser.add_argument("--plan", type=Path, default=Path("configs/name-that-llm/v6.json"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/improvement-v5/encoder"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/improvement-v5/model"))
    parser.add_argument("--selected", type=Path, default=Path("artifacts/improvement-v6/development/selected.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.plan, args.cache, args.model_dir, args.output_dir, args.selected)


if __name__ == "__main__":
    main()
