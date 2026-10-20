"""Fixed-weight combinations of text-only attribution evidence."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from name_that_llm.baseline import LABELS, fit_checked
from name_that_llm.sparse_v2 import SPARSE_CANDIDATES, build_sparse_pipeline
from name_that_llm.stylometry import build_style_pipeline
from name_that_llm.generative import GenerativeConfig, build_generative


def normalize_evidence(scores, *, probabilities=False):
    """Put different score types on a common per-passage scale, without fitting."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS) or not np.isfinite(values).all():
        raise ValueError("Expected finite passage-by-label evidence.")
    if probabilities:
        if np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Invalid class probabilities.")
        values = np.log(np.clip(values, 1e-12, 1.0))
    centered = values - values.mean(axis=1, keepdims=True)
    return centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-8)


def make_component(name):
    family, variant = name.split(":", 1)
    if family == "sparse":
        return build_sparse_pipeline(next(c for c in SPARSE_CANDIDATES if c.name == variant))
    if family == "style":
        return build_style_pipeline(variant)
    if family == "generative":
        return build_generative(GenerativeConfig(variant=variant))
    raise ValueError(f"Unknown component: {name}")


class EvidenceEnsemble(ClassifierMixin, BaseEstimator):
    def __init__(self, components, weights):
        self.components = components
        self.weights = weights

    def fit(self, X, y):
        weights = np.asarray(self.weights, dtype=float)
        if len(weights) != len(self.components) or not len(weights):
            raise ValueError("Each component needs a weight.")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Weights must be positive and finite.")
        self.weights_ = weights / weights.sum()
        self.models_ = []
        self.classes_ = np.asarray(LABELS)
        for name in self.components:
            model = fit_checked(make_component(name), X, y)
            if tuple(model.classes_) != LABELS:
                raise ValueError("Component class order differs from competition labels.")
            self.models_.append(model)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["models_", "weights_", "classes_"])
        combined = None
        for model, weight in zip(self.models_, self.weights_):
            is_probability = hasattr(model, "predict_proba")
            raw = model.predict_proba(X) if is_probability else model.decision_function(X)
            evidence = normalize_evidence(raw, probabilities=is_probability)
            combined = weight * evidence if combined is None else combined + weight * evidence
        return combined

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]
