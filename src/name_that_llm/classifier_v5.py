"""Frozen LUAR linear probe and fixed evidence fusion for experiment v5."""

from __future__ import annotations

import logging
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from name_that_llm.baseline import LABELS, file_sha256
from name_that_llm.ensemble_v2 import normalize_evidence
from name_that_llm.stacking_v4 import array_hash, validate_fold_assignment

LOGGER = logging.getLogger("embeddings_v5")
CLASSIFIER_PARAMETERS = {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs",
                         "max_iter": 3000, "tol": 1e-4}
RECIPES = {"luar": {"luar_weight": 1.0, "v4_weight": 0.0},
           "luar_v4_half": {"luar_weight": .5, "v4_weight": .5}}


def validate_embeddings(values, rows=None):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 512 or not len(values):
        raise ValueError("Expected nonempty row-by-512 LUAR embeddings")
    if rows is not None and len(values) != rows:
        raise ValueError("Embedding row count differs")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("Embeddings must be finite floating-point values")
    if not np.allclose(np.linalg.norm(values, axis=1), 1, atol=2e-6, rtol=2e-5):
        raise ValueError("Expected unit-norm embeddings")
    return values


def fit_classifier(embeddings, y):
    values = validate_embeddings(embeddings, len(y))
    labels = np.asarray(y)
    if labels.ndim != 1 or set(labels) != set(LABELS):
        raise ValueError("Training requires all 18 competition labels")
    model = LogisticRegression(**CLASSIFIER_PARAMETERS)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(values, labels)
    for warning in caught:
        LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("LUAR classifier did not converge")
    if tuple(model.classes_) != LABELS:
        raise ValueError("LUAR classifier class order differs")
    return model


def incumbent_evidence(stack_probs, base_evidence):
    stack = normalize_evidence(stack_probs, probabilities=True)
    base = np.asarray(base_evidence)
    if base.shape != (len(stack), 4, len(LABELS)) or not np.isfinite(base).all():
        raise ValueError("Invalid incumbent base evidence")
    return .5 * stack + .5 * base[:, [0, 1, 3]].mean(axis=1)


def combine_evidence(luar_probs, v4_scores, recipe):
    if recipe not in RECIPES.values():
        raise ValueError("Recipe differs from the two frozen candidates")
    luar = normalize_evidence(luar_probs, probabilities=True)
    v4 = normalize_evidence(v4_scores)
    if luar.shape != v4.shape:
        raise ValueError("Candidate and incumbent rows differ")
    return recipe["luar_weight"] * luar + recipe["v4_weight"] * v4


def evaluate_oof(embeddings, y, folds, groups):
    values = validate_embeddings(embeddings, len(y))
    labels = np.asarray(y)
    assignment = validate_fold_assignment(folds, labels, groups)
    groups = np.asarray(groups)
    result = np.full((len(values), len(LABELS)), np.nan)
    audit = []
    for fold in range(5):
        fit = np.flatnonzero(assignment != fold)
        valid = np.flatnonzero(assignment == fold)
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError("Training groups overlap held-out groups")
        started = time.perf_counter()
        model = fit_classifier(values[fit], labels[fit])
        with threadpool_limits(limits=2):
            result[valid] = model.predict_proba(values[valid])
        record = {"held_out_fold": fold, "training_rows": len(fit), "held_out_rows": len(valid),
                  "training_indexes_sha256": array_hash(fit), "held_out_indexes_sha256": array_hash(valid),
                  "training_groups": sorted(set(groups[fit])), "held_out_groups": sorted(set(groups[valid])),
                  "classes": model.classes_.tolist(), "iterations": model.n_iter_.tolist(),
                  "seconds": time.perf_counter() - started}
        audit.append(record)
        LOGGER.info("Completed fold=%d fit_rows=%d valid_rows=%d seconds=%.3f", fold, len(fit), len(valid), record["seconds"])
    normalize_evidence(result, probabilities=True)
    return result, audit


def source_hashes():
    from name_that_llm.stacking_v4 import source_hashes as incumbent_hashes
    hashes = incumbent_hashes()
    root = Path(__file__).resolve().parent
    for name in ("classifier_v5.py", "embeddings_v5.py"):
        hashes[name] = file_sha256(root / name)
    return hashes
