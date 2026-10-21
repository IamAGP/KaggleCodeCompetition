"""One frozen linear probe of the separately released StyleDistance encoder."""
from __future__ import annotations

import logging
from pathlib import Path
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from name_that_llm.baseline import LABELS, file_sha256
from name_that_llm.ensemble_v2 import normalize_evidence
from name_that_llm.stacking_v4 import array_hash, validate_fold_assignment

LOGGER = logging.getLogger("styledistance_v7")
CLASSIFIER_PARAMETERS = {"C": 64., "class_weight": "balanced", "solver": "lbfgs", "max_iter": 3000, "tol": 1e-4}
RECIPES = {"styledistance": {"embedding_weight": 1., "v4_weight": 0.},
           "styledistance_v4_half": {"embedding_weight": .5, "v4_weight": .5}}


def validate_embeddings(values, rows=None):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 768 or not len(values):
        raise ValueError("Expected nonempty row-by-768 StyleDistance embeddings")
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
        raise ValueError("All 18 competition labels are required")
    model = LogisticRegression(**CLASSIFIER_PARAMETERS)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(values, labels)
    for warning in caught:
        LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("StyleDistance linear classifier did not converge")
    if tuple(model.classes_) != LABELS:
        raise ValueError("Classifier class order differs")
    return model


def model_evidence(model, embeddings):
    if tuple(model.classes_) != LABELS:
        raise ValueError("Classifier class order differs")
    with threadpool_limits(limits=2):
        return normalize_evidence(model.predict_proba(validate_embeddings(embeddings)), probabilities=True)


def combine_evidence(embedding_scores, v4_scores, recipe):
    if recipe not in RECIPES.values():
        raise ValueError("Recipe differs from the two frozen candidates")
    scores = np.asarray(embedding_scores)
    incumbent = normalize_evidence(v4_scores)
    if scores.shape != incumbent.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid candidate evidence shape or values")
    return recipe["embedding_weight"] * scores + recipe["v4_weight"] * incumbent


def choose_recipe(results):
    names = [name for name in RECIPES if name in results]
    if not names or set(names) != set(results):
        raise ValueError("Unknown or empty recipe results")
    if any(not np.isfinite(results[n]["balanced_accuracy"]) or not 0 <= results[n]["balanced_accuracy"] <= 1 for n in names):
        raise ValueError("Invalid candidate score")
    return max(names, key=lambda n: results[n]["balanced_accuracy"])


def evaluate_oof(embeddings, y, folds, groups):
    values = validate_embeddings(embeddings, len(y))
    labels, groups = np.asarray(y), np.asarray(groups)
    assignment = validate_fold_assignment(folds, labels, groups)
    evidence = np.full((len(values), len(LABELS)), np.nan)
    audit = []
    for fold in range(5):
        fitting, valid = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        if set(groups[fitting]) & set(groups[valid]):
            raise ValueError("Training and held-out groups overlap")
        started = time.perf_counter()
        model = fit_classifier(values[fitting], labels[fitting])
        evidence[valid] = model_evidence(model, values[valid])
        train_predictions = np.asarray(LABELS)[model_evidence(model, values[fitting]).argmax(axis=1)]
        record = {"held_out_fold": fold, "training_rows": len(fitting), "held_out_rows": len(valid),
                  "training_indexes_sha256": array_hash(fitting), "held_out_indexes_sha256": array_hash(valid),
                  "training_embeddings_sha256": array_hash(values[fitting]), "held_out_embeddings_sha256": array_hash(values[valid]),
                  "held_out_scores_sha256": array_hash(evidence[valid]),
                  "training_groups": sorted(set(groups[fitting])), "held_out_groups": sorted(set(groups[valid])),
                  "classes": model.classes_.tolist(), "iterations": model.n_iter_.tolist(),
                  "training_balanced_accuracy": float(balanced_accuracy_score(labels[fitting], train_predictions)),
                  "seconds": time.perf_counter() - started}
        audit.append(record)
        LOGGER.info("Completed fold=%d train_BA=%.6f seconds=%.3f", fold, record["training_balanced_accuracy"], record["seconds"])
    if not np.isfinite(evidence).all():
        raise ValueError("Incomplete OOF evidence")
    return evidence, audit


def source_hashes():
    from name_that_llm.classifier_v5 import source_hashes as imported_hashes
    hashes = imported_hashes()
    root = Path(__file__).resolve().parent
    for name in ("encoder_v7.py", "classifier_v7.py"):
        hashes[name] = file_sha256(root / name)
    for name in ("run_embeddings_v5.py", "select_v4.py", "run_nested_v4.py", "extract_embeddings_v7.py"):
        hashes[f"scripts/{name}"] = file_sha256(root.parents[1] / "scripts" / name)
    return hashes
