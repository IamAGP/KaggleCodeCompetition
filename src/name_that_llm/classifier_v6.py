"""Two preregistered probes of the unchanged LUAR representation for v6."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from name_that_llm.baseline import LABELS, file_sha256
from name_that_llm.classifier_v5 import validate_embeddings
from name_that_llm.ensemble_v2 import normalize_evidence
from name_that_llm.stacking_v4 import array_hash, validate_fold_assignment

LOGGER = logging.getLogger("embeddings_v6")
MODEL_DEFINITIONS = {
    "lr64": {"kind": "logistic", "parameters": {
        "C": 64.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 3000, "tol": 1e-4}},
    "rbf10": {"kind": "svc", "parameters": {
        "C": 10.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced",
        "probability": False, "decision_function_shape": "ovr", "cache_size": 512,
        "tol": .001, "max_iter": -1}},
}
RECIPES = {
    "lr64": {"model": "lr64", "embedding_weight": 1.0, "v4_weight": 0.0},
    "lr64_v4_half": {"model": "lr64", "embedding_weight": .5, "v4_weight": .5},
    "rbf10": {"model": "rbf10", "embedding_weight": 1.0, "v4_weight": 0.0},
    "rbf10_v4_half": {"model": "rbf10", "embedding_weight": .5, "v4_weight": .5},
}


def fit_classifier(embeddings, y, model_name):
    if model_name not in MODEL_DEFINITIONS:
        raise ValueError("Unknown frozen classifier")
    values = validate_embeddings(embeddings, len(y))
    labels = np.asarray(y)
    if labels.ndim != 1 or set(labels) != set(LABELS):
        raise ValueError("Training requires all 18 competition labels")
    definition = MODEL_DEFINITIONS[model_name]
    factory = LogisticRegression if definition["kind"] == "logistic" else SVC
    model = factory(**deepcopy(definition["parameters"]))
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(values, labels)
    for warning in caught:
        LOGGER.warning("%s %s: %s", model_name, warning.category.__name__, warning.message)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or getattr(model, "fit_status_", 0) != 0:
        raise RuntimeError(f"Frozen classifier {model_name} did not converge")
    if tuple(model.classes_) != LABELS:
        raise ValueError("Classifier class order differs")
    return model


def model_evidence(model, embeddings, model_name):
    """Use LR log probabilities or SVC OVR scores, without fitting calibration."""
    if model_name not in MODEL_DEFINITIONS or tuple(model.classes_) != LABELS:
        raise ValueError("Unknown model or incorrect class order")
    values = validate_embeddings(embeddings)
    with threadpool_limits(limits=2):
        if MODEL_DEFINITIONS[model_name]["kind"] == "logistic":
            return normalize_evidence(model.predict_proba(values), probabilities=True)
        return normalize_evidence(model.decision_function(values))


def combine_evidence(embedding_scores, v4_scores, recipe):
    if recipe not in RECIPES.values():
        raise ValueError("Recipe differs from the four frozen candidates")
    scores = np.asarray(embedding_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(LABELS) or not np.isfinite(scores).all():
        raise ValueError("Invalid normalized embedding evidence")
    # The probe evidence is already normalized by model_evidence. Re-normalize
    # only the completed incumbent, whose internal fusion has a different scale.
    v4 = normalize_evidence(v4_scores)
    if scores.shape != v4.shape:
        raise ValueError("Embedding and incumbent evidence rows differ")
    return recipe["embedding_weight"] * scores + recipe["v4_weight"] * v4


def choose_recipe(results):
    """Highest score, with exact ties resolved in frozen declaration order."""
    names = [name for name in RECIPES if name in results]
    if not names or set(names) != set(results):
        raise ValueError("Unknown or empty recipe results")
    for name in names:
        score = results[name]["balanced_accuracy"]
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid selection score")
    return max(names, key=lambda name: results[name]["balanced_accuracy"])


def evaluate_oof(embeddings, y, folds, groups, model_name):
    values = validate_embeddings(embeddings, len(y))
    labels = np.asarray(y)
    assignment = validate_fold_assignment(folds, labels, groups)
    groups = np.asarray(groups)
    result = np.full((len(values), len(LABELS)), np.nan, dtype=np.float64)
    audit = []
    for fold in range(5):
        fit = np.flatnonzero(assignment != fold)
        valid = np.flatnonzero(assignment == fold)
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError("Training groups overlap held-out groups")
        started = time.perf_counter()
        model = fit_classifier(values[fit], labels[fit], model_name)
        train_scores = model_evidence(model, values[fit], model_name)
        held_out_scores = model_evidence(model, values[valid], model_name)
        result[valid] = held_out_scores
        record = {
            "model": model_name, "definition": deepcopy(MODEL_DEFINITIONS[model_name]),
            "held_out_fold": fold, "training_rows": len(fit), "held_out_rows": len(valid),
            "training_indexes_sha256": array_hash(fit), "held_out_indexes_sha256": array_hash(valid),
            "training_embeddings_sha256": array_hash(values[fit]),
            "held_out_embeddings_sha256": array_hash(values[valid]),
            "held_out_scores_sha256": array_hash(held_out_scores),
            "training_groups": sorted(set(groups[fit])), "held_out_groups": sorted(set(groups[valid])),
            "classes": model.classes_.tolist(), "iterations": np.asarray(model.n_iter_).tolist(),
            "training_balanced_accuracy": float(balanced_accuracy_score(labels[fit], np.asarray(LABELS)[train_scores.argmax(axis=1)])),
            "held_out_balanced_accuracy": float(balanced_accuracy_score(labels[valid], np.asarray(LABELS)[held_out_scores.argmax(axis=1)])),
            "prediction_rule": "argmax of normalized evidence, in LABELS order",
            "seconds": time.perf_counter() - started,
        }
        if MODEL_DEFINITIONS[model_name]["kind"] == "svc":
            record.update({"support_vectors": int(len(model.support_)), "support_vectors_per_class": model.n_support_.tolist(),
                           "fitted_gamma": float(model._gamma), "fit_status": int(model.fit_status_),
                           "probability_calibration": False})
        audit.append(record)
        LOGGER.info("Completed model=%s fold=%d train_BA=%.6f valid_BA=%.6f seconds=%.3f", model_name, fold,
                    record["training_balanced_accuracy"], record["held_out_balanced_accuracy"], record["seconds"])
    if not np.isfinite(result).all():
        raise ValueError("Incomplete or nonfinite OOF evidence")
    return result, audit


def source_hashes():
    from name_that_llm.classifier_v5 import source_hashes as predecessor_hashes
    hashes = predecessor_hashes()
    source = Path(__file__).resolve()
    hashes[source.name] = file_sha256(source)
    scripts = source.parents[2] / "scripts"
    for name in ("run_embeddings_v5.py", "select_v4.py", "run_nested_v4.py", "extract_embeddings_v5.py"):
        hashes[f"scripts/{name}"] = file_sha256(scripts / name)
    return hashes
