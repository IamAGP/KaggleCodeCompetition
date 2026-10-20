"""Evaluate the frozen four-base stack with exact double-exclusion cross-fitting."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import logging
import multiprocessing
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from name_that_llm.baseline import LABELS, file_sha256, load_data, runtime_versions
from name_that_llm.linear_v3 import LINEAR_CANDIDATES, LINEAR_COMMON
from name_that_llm.sparse_v2 import SPARSE_CANDIDATES
from name_that_llm.stylometry import CANDIDATES
from name_that_llm.stacking_v4 import (
    COMPONENTS, META_PARAMETERS, array_hash, audited_oof_bundle,
    component_evidence, evidence_features, fit_component, fit_meta, json_hash,
    recipe_definition, source_hashes, training_fingerprint, validate_fold_assignment,
)
from name_that_llm.ensemble_v2 import normalize_evidence

LOGGER = logging.getLogger("nested_v4")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_json(path):
    return json.loads(path.read_text())


def fixed_folds(train, path):
    assignments = pd.read_csv(path, dtype={"id": str, "prompt_id": str, "group": str})
    for column in ("id", "prompt_id"):
        if assignments[column].tolist() != train[column].tolist():
            raise ValueError(f"Fold CSV {column} order mismatch.")
    folds = validate_fold_assignment(
        assignments["fold"].to_numpy(), train["model"], train["prompt_id"],
    )
    if "group" in assignments:
        if assignments["group"].isna().any():
            raise ValueError("Explicit groups cannot be missing.")
        validate_fold_assignment(folds, train["model"], assignments["group"])
    assignments["fold"] = folds
    return assignments


def fold_groups(train, assignments):
    return assignments["group"] if "group" in assignments else train["prompt_id"]


def assert_equal(actual, expected, message):
    if actual != expected:
        raise ValueError(message)


def verify_manifest_file(experiment_root, relative_path):
    records = load_json(experiment_root / "run-manifest.json")["files"]
    record = next(r for r in records if r["path"] == relative_path)
    path = experiment_root / relative_path
    assert_equal(path.stat().st_size, record["bytes"], f"Size mismatch: {path}")
    assert_equal(file_sha256(path), record["sha256"], f"Hash mismatch: {path}")
    return {"path": str(path.relative_to(ROOT)), "sha256": record["sha256"]}


def archive_source_match(experiment_root, source_paths):
    audit = verify_manifest_file(experiment_root, "source.zip")
    with zipfile.ZipFile(experiment_root / "source.zip") as archive:
        for relative in source_paths:
            archived = hashlib.sha256(archive.read(relative)).hexdigest()
            assert_equal(file_sha256(ROOT / relative), archived, f"Archived source mismatch: {relative}")
    return audit


def ast_definitions(path):
    definitions = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            definitions[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = ast.dump(node, include_attributes=False)
    return definitions


def audit_legacy_outer(train, assignments, data_dir, folds_path):
    """Read only original OOF; prove exact row/class/fold/config/source provenance."""
    common = {
        "train_sha256": file_sha256(data_dir / "train.csv"),
        "folds_sha256": file_sha256(folds_path),
        "versions": runtime_versions(),
    }
    evidence = np.empty((len(train), len(COMPONENTS), len(LABELS)))
    audit = []
    v2, v3 = ROOT / "artifacts/improvement-v2", ROOT / "artifacts/improvement-v3"
    source_archive = archive_source_match(v2, [
        "src/name_that_llm/baseline.py", "src/name_that_llm/sparse_v2.py",
        "scripts/run_sparse_experiment.py",
    ])
    sparse_plan = load_json(v2 / "sparse/plan.json")
    for key, expected in common.items():
        assert_equal(sparse_plan[key], expected, f"Sparse provenance mismatch: {key}")
    config = next(c for c in SPARSE_CANDIDATES if c.name == "char15_c4")
    assert_equal(sparse_plan["candidates"][0], asdict(config), "Sparse recipe changed.")
    sparse_record = verify_manifest_file(v2, "sparse/char15_c4_oof.npz")
    metrics = load_json(v2 / "sparse/metrics.json")["char15_c4"]
    with np.load(v2 / "sparse/char15_c4_oof.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["ids"], train["id"].to_numpy())
        np.testing.assert_array_equal(saved["labels"], np.asarray(LABELS))
        scores = saved["scores"]
        predictions = np.asarray(LABELS)[scores.argmax(axis=1)]
        np.testing.assert_array_equal(saved["predictions"], predictions)
    assert_equal(scores.shape, (len(train), len(LABELS)), "Sparse shape mismatch.")
    assert_equal(metrics["config"], asdict(config), "Sparse metrics config mismatch.")
    # The historical sparse NPZ does not include folds. The immutable producer
    # reads the hashed fold CSV; recompute all five recorded held-out metrics.
    for record in metrics["folds"]:
        mask = assignments["fold"].to_numpy() == record["fold"]
        actual = float(balanced_accuracy_score(train.loc[mask, "model"], predictions[mask]))
        np.testing.assert_allclose(actual, record["balanced_accuracy"], atol=1e-15, rtol=0)
    evidence[:, 0] = normalize_evidence(scores)
    audit.append({
        "component": COMPONENTS[0], "score_artifact": sparse_record,
        "source_archive": source_archive, "plan_sha256": file_sha256(v2 / "sparse/plan.json"),
        "fold_proof": "Exact producer source, plan CSV hash and all five recorded fold metrics",
        "row_ids_sha256": json_hash(train["id"].tolist()), "classes_sha256": json_hash(list(LABELS)),
    })

    definition = load_json(v2 / "style/experiment_definition.json")
    for key, expected in common.items():
        assert_equal(definition[key], expected, f"Style provenance mismatch: {key}")
    assert_equal(definition["labels"], list(LABELS), "Style class order mismatch.")
    assert_equal(definition["candidates"], CANDIDATES, "Style recipes changed.")
    refactor = load_json(v2 / "style/refactor_record.json")
    original_source = v2 / "style/stylometry_at_definition.py"
    original_runner = v2 / "style/runner_at_definition.py"
    assert_equal(file_sha256(original_source), definition["style_source_sha256"], "Style original hash mismatch.")
    assert_equal(file_sha256(original_runner), definition["script_sha256"], "Style runner original hash mismatch.")
    assert_equal(file_sha256(ROOT / "src/name_that_llm/stylometry.py"), refactor["new_source_sha256"], "Style source changed.")
    assert_equal(file_sha256(ROOT / "scripts/run_style_experiment.py"), refactor["new_runner_sha256"], "Style runner changed.")
    current_defs = ast_definitions(ROOT / "src/name_that_llm/stylometry.py")
    for key, value in ast_definitions(original_source).items():
        if key in current_defs:
            assert_equal(current_defs[key], value, f"Style extraction behavior changed: {key}")
    old_runner_defs = ast_definitions(original_runner)
    for key in ("CANDIDATES", "build_style_pipeline"):
        assert_equal(current_defs[key], old_runner_defs[key], f"Style builder extraction changed: {key}")
    for component, name in ((1, "style_hist"), (2, "char_style_svm")):
        path = v2 / "style" / name
        record = verify_manifest_file(v2, f"style/{name}/oof_scores.npy")
        pred_record = verify_manifest_file(v2, f"style/{name}/oof_predictions.csv")
        scores = np.load(path / "oof_scores.npy", allow_pickle=False)
        predictions = pd.read_csv(path / "oof_predictions.csv", dtype={"id": str, "prompt_id": str})
        for column in ("id", "prompt_id", "model"):
            assert_equal(predictions[column].tolist(), train[column].tolist(), f"Style row mismatch: {column}")
        np.testing.assert_array_equal(predictions["fold"], assignments["fold"])
        np.testing.assert_array_equal(predictions["prediction"], np.asarray(LABELS)[scores.argmax(axis=1)])
        evidence[:, component] = normalize_evidence(scores, probabilities=name == "style_hist")
        audit.append({
            "component": COMPONENTS[component], "score_artifact": record,
            "predictions_artifact": pred_record, "source_refactor_ast_verified": True,
            "original_source_sha256": file_sha256(original_source),
            "current_source_sha256": file_sha256(ROOT / "src/name_that_llm/stylometry.py"),
            "definition_sha256": file_sha256(v2 / "style/experiment_definition.json"),
            "row_ids_sha256": json_hash(train["id"].tolist()), "classes_sha256": json_hash(list(LABELS)),
        })

    definition = load_json(v3 / "linear/experiment_definition.json")
    for key, expected in common.items():
        assert_equal(definition[key], expected, f"Linear provenance mismatch: {key}")
    assert_equal(definition["labels"], list(LABELS), "Linear class order mismatch.")
    assert_equal(definition["candidates"], LINEAR_CANDIDATES, "Linear recipes changed.")
    assert_equal(definition["common_parameters"], LINEAR_COMMON, "Linear common parameters changed.")
    for relative, digest in definition["source_hashes"].items():
        assert_equal(file_sha256(ROOT / relative), digest, f"Linear source changed: {relative}")
    relative = "linear/joint_char_style_c16"
    path = v3 / relative
    record = verify_manifest_file(v3, f"{relative}/oof_scores.npy")
    pred_record = verify_manifest_file(v3, f"{relative}/oof_predictions.csv")
    schema = load_json(path / "score_schema.json")
    assert_equal(schema["labels"], list(LABELS), "Linear schema labels mismatch.")
    assert_equal(schema["scores_kind"], "predict_proba", "Linear score kind mismatch.")
    assert_equal(schema["folds_sha256"], common["folds_sha256"], "Linear schema fold mismatch.")
    assert_equal(schema["source_hashes"], definition["source_hashes"], "Linear schema source mismatch.")
    scores = np.load(path / "oof_scores.npy", allow_pickle=False)
    predictions = pd.read_csv(path / "oof_predictions.csv", dtype={"id": str, "prompt_id": str})
    for column in ("id", "prompt_id", "model"):
        assert_equal(predictions[column].tolist(), train[column].tolist(), f"Linear row mismatch: {column}")
    np.testing.assert_array_equal(predictions["fold"], assignments["fold"])
    np.testing.assert_array_equal(predictions["prediction"], np.asarray(LABELS)[scores.argmax(axis=1)])
    evidence[:, 3] = normalize_evidence(scores, probabilities=True)
    audit.append({
        "component": COMPONENTS[3], "score_artifact": record, "predictions_artifact": pred_record,
        "definition_sha256": file_sha256(v3 / "linear/experiment_definition.json"),
        "source_hashes": definition["source_hashes"],
        "row_ids_sha256": json_hash(train["id"].tolist()), "classes_sha256": json_hash(list(LABELS)),
    })
    return audited_oof_bundle(
        evidence, assignments["fold"].to_numpy(), train["text"], train["model"], train["prompt_id"],
        provenance={"kind": "audited_historical_outer_only", **common, "components": audit},
    )


def cache_key(definition, excluded, train, assignments):
    held_mask = assignments["fold"].isin(excluded).to_numpy()
    fitting, valid = np.flatnonzero(~held_mask), np.flatnonzero(held_mask)
    if set(train.iloc[fitting]["prompt_id"]) & set(train.iloc[valid]["prompt_id"]):
        raise ValueError("Prompt leakage in excluded-fold job.")
    groups = fold_groups(train, assignments)
    if set(groups.iloc[fitting]) & set(groups.iloc[valid]):
        raise ValueError("Explicit group leakage in excluded-fold job.")
    if set(train.iloc[fitting]["model"]) != set(LABELS):
        raise ValueError("Excluded-fold training set is missing a class.")
    key = {
        "excluded_folds": list(excluded),
        "train_sha256": definition["train_sha256"],
        "folds_sha256": definition["folds_sha256"],
        "source_hashes": definition["source_hashes"],
        "script_sha256": definition["script_sha256"],
        "plan_sha256": definition["plan_sha256"],
        "recipe_sha256": definition["recipe_sha256"],
        "versions": definition["versions"],
        "train_ids_sha256": json_hash(train.iloc[fitting]["id"].tolist()),
        "prediction_ids_sha256": json_hash(train.iloc[valid]["id"].tolist()),
        "prediction_folds_sha256": array_hash(assignments.iloc[valid]["fold"].to_numpy()),
        "classes_sha256": json_hash(list(LABELS)),
        "groups_sha256": json_hash(groups.tolist()),
        "threads": definition["threads"],
    }
    return key, fitting, valid


def load_cache(path, expected_key, expected_valid, train, assignments):
    metadata = load_json(path / "metadata.json")
    assert_equal(metadata["key"], expected_key, f"Stale or invalid cache key: {path}")
    assert_equal(file_sha256(path / "scores.npz"), metadata["scores_file_sha256"], f"Corrupt cache: {path}")
    with np.load(path / "scores.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["row_indexes"], expected_valid)
        np.testing.assert_array_equal(saved["ids"], train.iloc[expected_valid]["id"].to_numpy())
        np.testing.assert_array_equal(saved["folds"], assignments.iloc[expected_valid]["fold"].to_numpy())
        np.testing.assert_array_equal(saved["labels"], np.asarray(LABELS))
        np.testing.assert_array_equal(saved["components"], np.asarray(COMPONENTS))
        values = saved["evidence"]
    evidence_features(values)
    assert_equal(values.shape[0], len(expected_valid), "Cache evidence row count mismatch.")
    assert_equal(array_hash(values), metadata["evidence_sha256"], "Cache evidence digest mismatch.")
    return expected_valid, values, metadata


def fit_exclusion_job(path, excluded, train, assignments, definition):
    """Fit once for an unordered excluded pair and predict both fold halves."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    key, fitting, valid = cache_key(definition, excluded, train, assignments)
    if (path / "metadata.json").exists():
        return load_cache(path, key, valid, train, assignments)
    # A metadata file is the completion marker. Incomplete jobs may be safely
    # resumed from individual components with a matching immutable job key.
    key_path = path / "job_key.json"
    if key_path.exists():
        assert_equal(load_json(key_path), key, f"Incomplete job key changed: {path}")
    else:
        write_json(key_path, key)
    logger = logging.getLogger("nested_v4")
    handler = logging.FileHandler(path / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    started = time.perf_counter()
    values = np.empty((len(valid), len(COMPONENTS), len(LABELS)))
    records = []
    try:
        logger.info("Start excluded=%s fitting=%d prediction=%d", excluded, len(fitting), len(valid))
        with threadpool_limits(limits=definition["threads"]):
            for component, name in enumerate(COMPONENTS):
                component_path = path / f"component-{component}.npy"
                component_meta_path = path / f"component-{component}.json"
                if component_meta_path.exists():
                    record = load_json(component_meta_path)
                    assert_equal(record["job_key_sha256"], json_hash(key), "Component job key mismatch.")
                    assert_equal(record["component"], name, "Component name mismatch.")
                    assert_equal(file_sha256(component_path), record["file_sha256"], "Component cache corrupt.")
                    result = np.load(component_path, allow_pickle=False)
                    assert_equal(result.shape, (len(valid), len(LABELS)), "Component shape mismatch.")
                    if not np.isfinite(result).all():
                        raise ValueError("Cached component contains non-finite scores.")
                    logger.info("Reuse excluded=%s component=%s", excluded, name)
                else:
                    component_started = time.perf_counter()
                    model = fit_component(name, train.iloc[fitting]["text"], train.iloc[fitting]["model"])
                    result = component_evidence(model, train.iloc[valid]["text"])
                    # Inference must be passage-local, including normalization.
                    probe = train.iloc[valid[:2]]["text"].tolist()
                    np.testing.assert_allclose(
                        component_evidence(model, probe),
                        np.vstack([component_evidence(model, [text]) for text in probe]),
                        rtol=1e-10, atol=1e-10,
                    )
                    iterations = getattr(model.named_steps["classifier"], "n_iter_", None)
                    np.save(component_path, result)
                    record = {
                        "component": name, "job_key_sha256": json_hash(key),
                        "file_sha256": file_sha256(component_path),
                        "seconds": time.perf_counter() - component_started,
                        "optimizer_iterations": None if iterations is None else np.asarray(iterations).tolist(),
                        "batch_invariance_verified": True,
                    }
                    write_json(component_meta_path, record)
                    logger.info("Done excluded=%s component=%s seconds=%.3f", excluded, name, record["seconds"])
                    del model
                values[:, component] = result
                records.append(record)
        evidence_features(values)
        np.savez_compressed(
            path / "scores.npz", evidence=values, row_indexes=valid,
            ids=train.iloc[valid]["id"].to_numpy(dtype=str),
            folds=assignments.iloc[valid]["fold"].to_numpy(),
            labels=np.asarray(LABELS), components=np.asarray(COMPONENTS),
        )
        metadata = {
            "key": key, "components": records, "fitting_rows": len(fitting),
            "prediction_rows": len(valid), "seconds": time.perf_counter() - started,
            "evidence_sha256": array_hash(values),
            "scores_file_sha256": file_sha256(path / "scores.npz"),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        if source_hashes() != definition["source_hashes"]:
            raise RuntimeError("Experiment source changed while a job was running.")
        write_json(path / "metadata.json", metadata)
        logger.info("Completed excluded=%s seconds=%.3f", excluded, metadata["seconds"])
    finally:
        logger.removeHandler(handler)
        handler.close()
    return valid, values, metadata


def run_exclusion_jobs(jobs, output, train, assignments, definition, workers, category):
    results = {}
    pending = []
    for excluded in jobs:
        path = output / category / ("excluded-" + "-".join(map(str, excluded)))
        key, _, valid = cache_key(definition, excluded, train, assignments)
        if (path / "metadata.json").exists():
            results[excluded] = load_cache(path, key, valid, train, assignments)
            LOGGER.info("Validated completed cache excluded=%s", excluded)
        else:
            pending.append((excluded, path))
    if workers == 1:
        for excluded, path in pending:
            results[excluded] = fit_exclusion_job(path, excluded, train, assignments, definition)
    elif pending:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {
                pool.submit(fit_exclusion_job, path, excluded, train, assignments, definition): excluded
                for excluded, path in pending
            }
            for future in as_completed(futures):
                excluded = futures[future]
                results[excluded] = future.result()
                LOGGER.info("Completed job excluded=%s", excluded)
    return results


def evaluate(train, assignments, outer, pairs, output, definition):
    """Each meta-training row comes from its exact (outer, own-fold) exclusion."""
    outer.validate(train["text"], train["model"], fold_groups(train, assignments))
    np.testing.assert_array_equal(outer.folds, assignments["fold"].to_numpy())
    folds = assignments["fold"].to_numpy()
    probs = np.full((len(train), len(LABELS)), np.nan)
    decision = np.full_like(probs, np.nan)
    metrics, routing = [], []
    for outer_fold in range(5):
        fitting, held_out = np.flatnonzero(folds != outer_fold), np.flatnonzero(folds == outer_fold)
        inner = np.full((len(train), len(COMPONENTS), len(LABELS)), np.nan)
        counts = np.zeros(len(train), dtype=np.int64)
        for inner_fold in range(5):
            if inner_fold == outer_fold:
                continue
            excluded = tuple(sorted((outer_fold, inner_fold)))
            indexes, values, metadata = pairs[excluded]
            use = folds[indexes] == inner_fold
            rows = indexes[use]
            np.testing.assert_array_equal(rows, np.flatnonzero(folds == inner_fold))
            assert_equal(metadata["key"]["excluded_folds"], list(excluded), "Wrong excluded pair routed.")
            inner[rows] = values[use]
            counts[rows] += 1
            routing.append({
                "outer_fold": outer_fold, "inner_fold": inner_fold,
                "pair": list(excluded), "row_ids_sha256": json_hash(train.iloc[rows]["id"].tolist()),
                "train_ids_sha256": metadata["key"]["train_ids_sha256"],
                "pair_scores_sha256": metadata["scores_file_sha256"],
            })
        if not np.all(counts[fitting] == 1) or not np.all(counts[held_out] == 0):
            raise ValueError("Nested routing is incomplete or includes the outer fold.")
        with threadpool_limits(limits=definition["threads"]):
            model = fit_meta(inner[fitting], train.iloc[fitting]["model"])
            held_features = evidence_features(outer.evidence[held_out])
            probs[held_out] = model.predict_proba(held_features)
            decision[held_out] = model.decision_function(held_features)
        scale = model.named_steps["scale"]
        np.testing.assert_allclose(scale.mean_, evidence_features(inner[fitting]).mean(axis=0), atol=1e-12)
        assert_equal(int(scale.n_samples_seen_), len(fitting), "Meta scaler trained on wrong rows.")
        np.testing.assert_allclose(
            probs[held_out[:3]],
            np.vstack([model.predict_proba(held_features[i:i+1]) for i in range(min(3, len(held_out)))]),
            rtol=1e-12, atol=1e-12,
        )
        pred = np.asarray(LABELS)[probs[held_out].argmax(axis=1)]
        record = {
            "fold": outer_fold, "balanced_accuracy": float(balanced_accuracy_score(train.iloc[held_out]["model"], pred)),
            "training_rows": len(fitting), "validation_rows": len(held_out),
            "meta_train_evidence_sha256": array_hash(inner[fitting]),
            "meta_train_ids_sha256": json_hash(train.iloc[fitting]["id"].tolist()),
            "meta_scaler_training_rows_verified": True, "meta_prediction_batch_invariance": True,
            "optimizer_iterations": model.named_steps["classifier"].n_iter_.tolist(),
        }
        metrics.append(record)
        LOGGER.info("Outer fold complete %s", json.dumps(record))
        write_json(output / "fold_metrics.json", metrics)
    np.testing.assert_allclose(probs.sum(axis=1), 1., atol=1e-12)
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("Invalid stacked OOF probabilities.")
    prediction = np.asarray(LABELS)[probs.argmax(axis=1)]
    frame = train[["id", "prompt_id", "model"]].copy()
    frame["fold"] = folds
    if "group" in assignments:
        frame["group"] = assignments["group"].to_numpy()
    frame["prediction"] = prediction
    frame.to_csv(output / "oof_predictions.csv", index=False)
    np.save(output / "stack_oof_probabilities.npy", probs)
    np.save(output / "oof_decision.npy", decision)
    np.save(output / "outer_oof_evidence.npy", outer.evidence)
    np.savez_compressed(
        output / "aligned_oof.npz", probabilities=probs, decision=decision, outer_evidence=outer.evidence,
        ids=train["id"].to_numpy(dtype=str), folds=folds,
        labels=np.asarray(LABELS), components=np.asarray(COMPONENTS),
    )
    pd.DataFrame({
        "model": LABELS, "recall": recall_score(train["model"], prediction, labels=LABELS, average=None),
    }).to_csv(output / "per_class_recall.csv", index=False)
    pd.DataFrame(
        confusion_matrix(train["model"], prediction, labels=LABELS), index=LABELS, columns=LABELS,
    ).to_csv(output / "confusion_matrix.csv", index_label="true_model")
    result = {
        "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(train["model"], prediction)),
        "folds": metrics, "labels": list(LABELS), "components": list(COMPONENTS),
        "scores_kind": "predict_proba", "shape": list(probs.shape),
        "probabilities_sha256": file_sha256(output / "stack_oof_probabilities.npy"),
        "prediction_rows_sha256": json_hash(train["id"].tolist()),
        "source_hashes": definition["source_hashes"],
        "train_sha256": definition["train_sha256"],
        "folds_sha256": definition["folds_sha256"],
        "nested_protocol": "Every meta-training row excludes its own fold and the outer validation fold.",
        "number_pair_fits": 10, "base_fits_per_pair": 4,
    }
    write_json(output / "metrics.json", result)
    write_json(output / "nested_routing_audit.json", routing)
    write_json(output / "score_schema.json", {
        "labels": list(LABELS), "components": list(COMPONENTS),
        "probabilities": "Rows in exact train.csv order; columns in labels order.",
        "outer_evidence": "Shape (rows, four components, eighteen classes); normalized evidence.",
        "folds_sha256": definition["folds_sha256"], "source_hashes": definition["source_hashes"],
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/name-that-llm")
    parser.add_argument("--folds", type=Path, default=ROOT / "artifacts/baseline-v1/folds.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/improvement-v4/nested")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/name-that-llm/v4.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cached-outer-oof", action="store_true", help="Audit and reuse original exact v2/v3 OOF.")
    mode.add_argument("--refit-outer", action="store_true", help="Fresh outer predictions for supplied folds.")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--audit-only", action="store_true", help="Freeze and audit without fitting.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / "run.log")],
    )
    train, _ = load_data(args.data_dir)
    assignments = fixed_folds(train, args.folds)
    cached_mode = not args.refit_outer
    definition = {
        "experiment": "improvement-v4-nested-four-base-stack",
        "plan_sha256": file_sha256(args.plan),
        "train_sha256": file_sha256(args.data_dir / "train.csv"),
        "folds_sha256": file_sha256(args.folds),
        "training_fingerprint": training_fingerprint(train["text"], train["model"], train["prompt_id"]),
        "source_hashes": source_hashes(), "script_sha256": file_sha256(Path(__file__)),
        "recipe": recipe_definition(), "recipe_sha256": json_hash(recipe_definition()),
        "meta_parameters": META_PARAMETERS, "labels": list(LABELS), "versions": runtime_versions(),
        "threads": args.threads, "workers": args.workers,
        "outer_prediction_source": "audited_original_v2_v3_oof" if cached_mode else "fresh_outer_fits",
        "inner_protocol": "Ten unordered excluded-fold pairs; route the other excluded half into each outer training set.",
        "candidate_budget": "One frozen meta recipe over exactly four existing bases; no tuning.",
        "sources": [
            "https://arxiv.org/pdf/1105.5466",
            "https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.StackingClassifier.html",
            "https://github.com/scikit-learn/scikit-learn/blob/1.7.1/sklearn/ensemble/_stacking.py",
        ],
        "research_note": "Ting and Witten describe cross-fitted confidence inputs and distinguish internal folds from outer evaluation. Logistic meta model and normalized scores are project choices, not a paper reproduction. The historical paper names no GitHub repository.",
        "limitations": [
            "Existing bases and folds already underwent development selection.",
            "Nested evaluation prevents direct outer leakage but cannot remove historical selection optimism.",
            "Inner bases train on three folds while outer bases train on four.",
        ],
    }
    definition_path = args.output_dir / "experiment_definition.json"
    if definition_path.exists():
        assert_equal(load_json(definition_path), definition, "Frozen definition changed; use a new experiment directory.")
    else:
        write_json(definition_path, definition)
    (args.output_dir / "source").mkdir(exist_ok=True)
    for name in definition["source_hashes"]:
        destination = args.output_dir / "source" / name
        if destination.exists():
            assert_equal(file_sha256(destination), definition["source_hashes"][name], "Source snapshot changed.")
        else:
            destination.write_bytes((ROOT / "src/name_that_llm" / name).read_bytes())
    runner_snapshot = args.output_dir / "source" / Path(__file__).name
    if runner_snapshot.exists():
        assert_equal(file_sha256(runner_snapshot), definition["script_sha256"], "Runner snapshot changed.")
    else:
        runner_snapshot.write_bytes(Path(__file__).read_bytes())
    outer = None
    if cached_mode:
        outer = audit_legacy_outer(train, assignments, args.data_dir, args.folds)
        write_json(args.output_dir / "outer_oof_audit.json", outer.audit)
        LOGGER.info("All four historical outer OOF sources passed exact provenance audit.")
    if args.audit_only:
        LOGGER.info("Definition frozen and audit complete; no model fitting requested.")
        return
    pairs = run_exclusion_jobs(
        list(combinations(range(5), 2)), args.output_dir, train, assignments,
        definition, args.workers, "pairs",
    )
    if not cached_mode:
        single = run_exclusion_jobs(
            [(fold,) for fold in range(5)], args.output_dir, train, assignments,
            definition, args.workers, "outer",
        )
        evidence = np.full((len(train), len(COMPONENTS), len(LABELS)), np.nan)
        for indexes, values, _ in single.values():
            evidence[indexes] = values
        outer = audited_oof_bundle(
            evidence, assignments["fold"].to_numpy(), train["text"], train["model"], fold_groups(train, assignments),
            provenance={"kind": "fresh_group_outer_fits", "jobs": [value[2] for value in single.values()]},
        )
        write_json(args.output_dir / "outer_oof_audit.json", outer.audit)
    result = evaluate(train, assignments, outer, pairs, args.output_dir, definition)
    assert_equal(source_hashes(), definition["source_hashes"], "Source changed during experiment.")
    assert_equal(file_sha256(Path(__file__)), definition["script_sha256"], "Runner changed during experiment.")
    write_json(args.output_dir / "completion.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "source_unchanged_during_run": True, "metrics": result,
    })
    artifact_files = [
        "stack_oof_probabilities.npy", "outer_oof_evidence.npy", "oof_predictions.csv",
        "oof_decision.npy", "aligned_oof.npz", "metrics.json", "score_schema.json",
        "nested_routing_audit.json", "outer_oof_audit.json", "experiment_definition.json",
        "completion.json", "fold_metrics.json", "per_class_recall.csv", "confusion_matrix.csv",
    ]
    write_json(args.output_dir / "artifacts.json", {
        "files": [{
            "path": name, "bytes": (args.output_dir / name).stat().st_size,
            "sha256": file_sha256(args.output_dir / name),
        } for name in artifact_files],
    })
    LOGGER.info("Completed nested pooled OOF balanced accuracy=%.9f", result["pooled_oof_balanced_accuracy"])


if __name__ == "__main__":
    main()
