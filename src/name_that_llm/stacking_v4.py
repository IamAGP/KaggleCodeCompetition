"""Prompt-group cross-fitted stacking with an explicit audited OOF boundary.

The evaluation driver uses double exclusions for its inner predictions.
This estimator generates ordinary group OOF only inside its own fit dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from name_that_llm.baseline import LABELS, file_sha256
from name_that_llm.ensemble_v2 import normalize_evidence
from name_that_llm.ensemble_v3 import make_v3_component

LOGGER = logging.getLogger("nested_v4")
COMPONENTS = (
    "sparse:char15_c4",
    "style:style_hist",
    "style:char_style_svm",
    "linear:joint_char_style_c16",
)
META_PARAMETERS = {"C": .05, "solver": "lbfgs", "max_iter": 2000, "tol": 1e-4}
SOURCE_FILES = (
    "baseline.py", "sparse_v2.py", "stylometry.py", "linear_v3.py",
    "ensemble_v2.py", "ensemble_v3.py", "stacking_v4.py",
)


def json_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            default=lambda item: item.item() if isinstance(item, np.generic) else item.tolist(),
        ).encode()
    ).hexdigest()


def array_hash(array):
    values = np.ascontiguousarray(array)
    return hashlib.sha256(
        str(values.dtype).encode() + json.dumps(list(values.shape)).encode() + values.tobytes()
    ).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: file_sha256(root / name) for name in SOURCE_FILES}


def recipe_definition():
    # Include every effective sklearn parameter, including estimator defaults.
    return {
        "components": list(COMPONENTS),
        "estimator_parameters": {
            name: {
                key: repr(value) for key, value in make_v3_component(name).get_params(deep=True).items()
            }
            for name in COMPONENTS
        },
        "normalization": "normalize_evidence per row, log clipped probabilities",
        "meta": META_PARAMETERS,
        "meta_preprocessing": "StandardScaler fitted on meta training rows only",
    }


def training_fingerprint(X, y, groups):
    return json_hash([list(X), list(y), list(groups)])


def validate_training(X, y, groups):
    if isinstance(X, str) or groups is None:
        raise ValueError("Pass a sequence of texts and explicit training prompt groups.")
    texts = np.asarray(X, dtype=object)
    targets = np.asarray(y, dtype=object)
    prompt_groups = np.asarray(groups, dtype=object)
    if any(a.ndim != 1 for a in (texts, targets, prompt_groups)):
        raise ValueError("Texts, labels and prompt groups must be one-dimensional.")
    if not len(texts) or not len(texts) == len(targets) == len(prompt_groups):
        raise ValueError("Texts, labels and prompt groups must have equal nonzero length.")
    if not all(isinstance(t, str) for t in texts):
        raise TypeError("All passages must be strings.")
    if any(g is None or (isinstance(g, float) and np.isnan(g)) for g in prompt_groups):
        raise ValueError("Prompt groups cannot be missing.")
    if set(targets) != set(LABELS):
        raise ValueError("All 18 competition classes are required.")
    return texts, targets, prompt_groups


def validate_fold_assignment(folds, y, groups, *, n_splits=5):
    assignment = np.asarray(folds)
    targets, prompt_groups = np.asarray(y), np.asarray(groups)
    if assignment.ndim != 1 or len(assignment) != len(targets) or len(groups) != len(targets):
        raise ValueError("Fold, label and group lengths differ.")
    if set(assignment) != set(range(n_splits)):
        raise ValueError(f"Expected folds numbered 0 through {n_splits - 1}.")
    for group in set(prompt_groups):
        if len(set(assignment[prompt_groups == group])) != 1:
            raise ValueError("A prompt group crosses folds.")
    for fold in range(n_splits):
        for mask in (assignment == fold, assignment != fold):
            if set(targets[mask]) != set(LABELS):
                raise ValueError("Every training and held-out fold must contain all classes.")
    return assignment.astype(np.int64)


def make_group_assignment(X, y, groups, *, n_splits=5, seed=42):
    assignment = np.full(len(X), -1, dtype=np.int64)
    cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid) in enumerate(cv.split(X, y, groups)):
        assignment[valid] = fold
    return validate_fold_assignment(assignment, y, groups, n_splits=n_splits)


def fit_component(name, X, y):
    LOGGER.info("Fit component=%s rows=%d", name, len(X))
    model = make_v3_component(name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    for warning in caught:
        LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError(f"Component failed to converge: {name}")
    if tuple(model.classes_) != LABELS:
        raise ValueError("Base classifier class order mismatch.")
    return model


def component_evidence(model, X):
    if tuple(model.classes_) != LABELS:
        raise ValueError("Base classifier class order mismatch.")
    probabilities = hasattr(model, "predict_proba")
    values = model.predict_proba(X) if probabilities else model.decision_function(X)
    return normalize_evidence(values, probabilities=probabilities)


def evidence_features(evidence):
    """Turn (rows, four components, eighteen classes) into 72 features."""
    values = np.asarray(evidence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(COMPONENTS), len(LABELS)):
        raise ValueError("Expected row-by-component-by-class evidence.")
    if not np.isfinite(values).all():
        raise ValueError("Evidence contains non-finite values.")
    return values.reshape(len(values), -1)


def fit_meta(evidence, y):
    features = evidence_features(evidence)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(**META_PARAMETERS)),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, y)
    for warning in caught:
        LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Meta classifier failed to converge.")
    if tuple(model.classes_) != LABELS:
        raise ValueError("Meta classifier class order mismatch.")
    return model


@dataclass(frozen=True)
class AuditedOOF:
    """An OOF bundle bound to exact training rows, folds, source and recipe.

    Constructed by the evaluator after auditing producer provenance, or by
    generate_fit_oof. A raw array is deliberately not accepted by fit().
    """

    evidence: np.ndarray
    folds: np.ndarray
    audit: dict

    def validate(self, X, y, groups):
        evidence_features(self.evidence)
        validate_fold_assignment(self.folds, y, groups)
        expected = {
            "training_fingerprint": training_fingerprint(X, y, groups),
            "evidence_sha256": array_hash(self.evidence),
            "fold_assignment_sha256": array_hash(self.folds),
            "source_hashes": source_hashes(),
            "recipe_sha256": json_hash(recipe_definition()),
            "components": list(COMPONENTS),
            "labels": list(LABELS),
            "group_oof_verified": True,
        }
        if self.evidence.shape[0] != len(X):
            raise ValueError("OOF evidence row count differs.")
        for key, value in expected.items():
            if self.audit.get(key) != value:
                raise ValueError(f"OOF audit mismatch: {key}")
        if not self.audit.get("provenance"):
            raise ValueError("OOF provenance is required.")
        return self


def audited_oof_bundle(evidence, folds, X, y, groups, *, provenance):
    """Seal an OOF matrix after the caller verifies its fold producer records."""
    values = np.array(evidence, dtype=np.float64, copy=True)
    assignment = validate_fold_assignment(folds, y, groups).copy()
    values.flags.writeable = False
    assignment.flags.writeable = False
    audit = {
        "training_fingerprint": training_fingerprint(X, y, groups),
        "evidence_sha256": array_hash(values),
        "fold_assignment_sha256": array_hash(assignment),
        "source_hashes": source_hashes(),
        "recipe_sha256": json_hash(recipe_definition()),
        "components": list(COMPONENTS),
        "labels": list(LABELS),
        "group_oof_verified": True,
        "provenance": provenance,
    }
    return AuditedOOF(values, assignment, audit).validate(X, y, groups)


def generate_fit_oof(X, y, groups, *, threads=2):
    texts, targets, prompt_groups = validate_training(X, y, groups)
    folds = make_group_assignment(texts, targets, prompt_groups)
    evidence = np.full((len(texts), len(COMPONENTS), len(LABELS)), np.nan)
    provenance = []
    with threadpool_limits(limits=threads):
        for fold in range(5):
            fitting, valid = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            for component, name in enumerate(COMPONENTS):
                model = fit_component(name, texts[fitting], targets[fitting])
                evidence[valid, component] = component_evidence(model, texts[valid])
            provenance.append({
                "held_out_fold": fold, "training_indexes_sha256": array_hash(fitting),
                "held_out_indexes_sha256": array_hash(valid),
            })
    return audited_oof_bundle(
        evidence, folds, texts, targets, prompt_groups,
        provenance={"kind": "runtime_group_oof", "seed": 42, "folds": provenance},
    )


class NestedStacker(ClassifierMixin, BaseEstimator):
    """Four frozen evidence bases and a group cross-fitted logistic meta model."""

    requires_groups = True

    def __init__(self, threads=2):
        self.threads = threads

    def fit(self, X, y, groups=None, *, fit_oof=None):
        texts, targets, prompt_groups = validate_training(X, y, groups)
        if not isinstance(self.threads, int) or not 1 <= self.threads <= 2:
            raise ValueError("Use one or two training threads.")
        if fit_oof is None:
            fit_oof = generate_fit_oof(texts, targets, prompt_groups, threads=self.threads)
        elif not isinstance(fit_oof, AuditedOOF):
            raise TypeError("fit_oof must be an AuditedOOF instance, never raw OOF scores.")
        fit_oof.validate(texts, targets, prompt_groups)
        with threadpool_limits(limits=self.threads):
            self.meta_ = fit_meta(fit_oof.evidence, targets)
            self.models_ = [fit_component(name, texts, targets) for name in COMPONENTS]
        self.classes_ = np.asarray(LABELS)
        self.fit_oof_audit_ = fit_oof.audit.copy()
        self.n_features_in_ = 1
        return self

    def base_evidence(self, X):
        check_is_fitted(self, ["models_", "meta_", "classes_"])
        if isinstance(X, str):
            raise ValueError("Expected a sequence of passages.")
        texts = list(X)
        with threadpool_limits(limits=self.threads):
            return np.stack([component_evidence(model, texts) for model in self.models_], axis=1)

    def predict_proba(self, X):
        check_is_fitted(self, ["models_", "meta_", "classes_"])
        return self.meta_.predict_proba(evidence_features(self.base_evidence(X)))

    def decision_function(self, X):
        check_is_fitted(self, ["models_", "meta_", "classes_"])
        return self.meta_.decision_function(evidence_features(self.base_evidence(X)))

    def predict(self, X):
        check_is_fitted(self, ["models_", "meta_", "classes_"])
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
