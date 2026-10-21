"""Evaluate the frozen StyleDistance probe and its one fixed v4 blend."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.baseline import LABELS, runtime_versions
from name_that_llm.classifier_v7 import (
    CLASSIFIER_PARAMETERS, RECIPES, choose_recipe, combine_evidence, evaluate_oof, source_hashes,
)
from name_that_llm.encoder_v7 import load_embedding_cache
from name_that_llm.embeddings_v5 import sha256_file as file_sha256
from run_embeddings_v5 import bootstrap_delta, diagnostics, load_incumbent, read_json
from select_v4 import evaluate

FROZEN_PLAN_SHA256 = "fdce1e0af45dab45031877dd8fef56272d06cedbd4882f230712c78204eed2dd"


def save_json(path, value):
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x") as handle:
        handle.write(serialized)


def validate_plan(path):
    if file_sha256(path) != FROZEN_PLAN_SHA256 or file_sha256(Path("artifacts/improvement-v7/frozen-plan.json")) != FROZEN_PLAN_SHA256:
        raise ValueError("Frozen v7 plan changed")
    plan = read_json(path)
    if plan["classifier"] != CLASSIFIER_PARAMETERS or plan["candidates"] != RECIPES or list(plan["candidates"]) != list(RECIPES):
        raise ValueError("Implemented classifier or candidate order differs from plan")
    for p, expected in plan["predecessors"].items():
        if file_sha256(p) != expected:
            raise ValueError(f"Recorded predecessor changed: {p}")
    return plan


def verify_selection(path):
    root = path.parent
    records = read_json(root / "artifacts.json")["files"]
    entries = {r["path"]: r for r in records}
    required = {path.name, "embedding_oof_evidence.npy", "fit_audit.json", "oof_predictions.csv",
                "experiment_definition.json", "completion.json", "score_schema.json", "diagnostics.json"}
    if len(entries) != len(records) or not required <= set(entries):
        raise ValueError("Development manifest incomplete or duplicate")
    for name, record in entries.items():
        if Path(name).name != name:
            raise ValueError("Invalid development artifact path")
        item = root / name
        if item.stat().st_size != record["bytes"] or file_sha256(item) != record["sha256"]:
            raise ValueError(f"Development artifact changed: {name}")
    done = read_json(root / "completion.json")
    if (not done["all_checks_passed"] or not done["source_unchanged_during_run"]
            or not done["inputs_unchanged_during_run"] or done["result_sha256"] != file_sha256(path)):
        raise ValueError("Development selection was not completed unchanged")
    selected = read_json(path)
    schema = read_json(root / "score_schema.json")
    rows = pd.read_csv(root / "oof_predictions.csv")
    evidence = np.load(root / "embedding_oof_evidence.npy", allow_pickle=False)
    expected_shape = (len(rows), len(LABELS))
    if (schema["labels"] != list(LABELS) or schema["row_ids"] != rows["id"].tolist()
            or schema["shape"] != list(expected_shape) or schema["score_kind"] != "normalized_log_probabilities"
            or evidence.shape != expected_shape or not np.isfinite(evidence).all()):
        raise ValueError("Development score schema, row alignment or evidence is invalid")
    standard_deviations = evidence.std(axis=1)
    if (not np.allclose(evidence.mean(axis=1), 0, atol=1e-8)
            or not np.all(np.isclose(standard_deviations, 1, atol=1e-7) | np.isclose(standard_deviations, 0, atol=1e-8))):
        raise ValueError("Development evidence is not normalized per row")
    if selected["mode"] != "development" or set(selected["all_results"]) != set(RECIPES):
        raise ValueError("Development did not evaluate the two frozen recipes")
    name = selected["selected_name"]
    if name != choose_recipe(selected["all_results"]) or selected["recipe"] != RECIPES[name]:
        raise ValueError("Selected recipe differs from the frozen winner")
    if selected["balanced_accuracy"] != selected["all_results"][name]["balanced_accuracy"]:
        raise ValueError("Selected score differs from candidate results")
    gain = selected["balanced_accuracy"] - selected["baseline"]["balanced_accuracy"]
    if not np.isclose(gain, selected["delta_from_v4"], atol=1e-12, rtol=0):
        raise ValueError("Development gain is inconsistent")
    return selected


def run(mode, plan_path, cache, model_dir, out, selected_path):
    if mode not in ("development", "robustness"):
        raise ValueError("Unknown evaluation mode")
    plan = validate_plan(plan_path)
    before, evaluator_hash = source_hashes(), file_sha256(__file__)
    selected, extra_inputs = None, []
    if mode == "robustness":
        selected = verify_selection(selected_path)
        if not selected["passes_development_gate"] or selected["delta_from_v4"] < plan["development"]["min_gain"]:
            raise ValueError("Development gate failed")
        for key, expected in {"source_hashes": before, "evaluator_sha256": evaluator_hash,
                              "plan_sha256": file_sha256(plan_path)}.items():
            if selected[key] != expected:
                raise ValueError(f"Selected provenance changed: {key}")
        for p, expected in selected["input_sha256"].items():
            if file_sha256(p) != expected:
                raise ValueError(f"Development input changed: {p}")
        manifest_path = selected_path.parent / "artifacts.json"
        extra_inputs = [manifest_path, *[selected_path.parent / r["path"] for r in read_json(manifest_path)["files"]]]
    train, folds, groups, incumbent, baseline, baseline_prediction, paths = load_incumbent(plan, mode)
    embeddings, metadata = load_embedding_cache(cache, "train", train.id.tolist(), train.text.tolist(),
                                              model_dir=model_dir, plan_path=plan_path,
                                              source_path="data/name-that-llm/train.csv")
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(out / "debug.log"), logging.StreamHandler()])
    paths.extend([plan_path, Path("artifacts/improvement-v7/frozen-plan.json"),
                  cache / "train-embeddings.npz", cache / "train-embeddings.json",
                  Path("data/name-that-llm/train.csv"), model_dir / "manifest.json",
                  *[Path(p) for p in plan["predecessors"]],
                  *[model_dir / r["path"] for r in metadata["provenance"]["model_files"]], *extra_inputs])
    inputs = {str(p): file_sha256(p) for p in paths}
    candidates = RECIPES if selected is None else {selected["selected_name"]: selected["recipe"]}
    definition = {"mode": mode, "plan_sha256": file_sha256(plan_path), "source_hashes": before,
                  "evaluator_sha256": evaluator_hash, "input_sha256": inputs, "labels": list(LABELS),
                  "classifier": CLASSIFIER_PARAMETERS, "candidate_recipes": candidates,
                  "versions": runtime_versions(), "embedding_provenance": metadata["provenance"]}
    save_json(out / "experiment_definition.json", definition)
    save_json(out / "score_schema.json", {"labels": list(LABELS), "row_ids": train.id.tolist(),
                                          "score_kind": "normalized_log_probabilities", "shape": [len(train), len(LABELS)]})
    evidence, audit = evaluate_oof(embeddings, train.model.to_numpy(), folds, groups)
    np.save(out / "embedding_oof_evidence.npy", evidence, allow_pickle=False)
    save_json(out / "fit_audit.json", audit)
    results, predictions = {}, {}
    for name, recipe in candidates.items():
        results[name], predictions[name] = evaluate(combine_evidence(evidence, incumbent, recipe), train.model.to_numpy(), folds)
    save_json(out / "diagnostics.json", {name: diagnostics(train.model.to_numpy(), pred, baseline_prediction)
                                        for name, pred in predictions.items()})
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
    if source_hashes() != before or file_sha256(__file__) != evaluator_hash:
        raise ValueError("Evaluation source changed during run")
    for p, expected in inputs.items():
        if file_sha256(p) != expected:
            raise ValueError(f"Evaluation input changed during run: {p}")
    rows = train[["id", "prompt_id", "model"]].copy()
    rows["fold"], rows["group"], rows["v4"], rows["prediction"] = folds, groups, baseline_prediction, predictions[chosen]
    for name, pred in predictions.items():
        rows[name] = pred
    rows.to_csv(out / "oof_predictions.csv", index=False)
    save_json(out / filename, result)
    save_json(out / "completion.json", {"all_checks_passed": True, "source_unchanged_during_run": True,
                                        "inputs_unchanged_during_run": True, "result_sha256": file_sha256(out / filename)})
    save_json(out / "artifacts.json", {"files": [{"path": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)}
                                                 for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps({key: result[key] for key in ("selected_name", "balanced_accuracy", "delta_from_v4", "fold_wins", "all_results")}, indent=2))
    print("PASS" if result.get("passes_development_gate", result.get("passes")) else "FAIL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("development", "robustness"))
    parser.add_argument("--plan", type=Path, default=Path("configs/name-that-llm/v7.json"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/improvement-v7/encoder"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/improvement-v7/model"))
    parser.add_argument("--selected", type=Path, default=Path("artifacts/improvement-v7/development/selected.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.plan, args.cache, args.model_dir, args.output_dir, args.selected)


if __name__ == "__main__":
    main()
