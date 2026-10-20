"""Fixed, passage-local ensembles for the separately recorded v3 experiment."""

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from name_that_llm.baseline import LABELS
from name_that_llm.ensemble_v2 import make_component, normalize_evidence


def make_v3_component(name):
    if name.startswith("kernel:"):
        from name_that_llm.kernel_v3 import build_kernel_pipeline
        return build_kernel_pipeline(name.split(":", 1)[1])
    if name.startswith("linear:"):
        from name_that_llm.linear_v3 import build_linear_pipeline
        return build_linear_pipeline(name.split(":", 1)[1])
    if name.startswith("boosting:"):
        from name_that_llm.boosting_v3 import build_boosting_pipeline
        return build_boosting_pipeline(name.split(":", 1)[1])
    if name.startswith("topic:"):
        from name_that_llm.topic_v3 import build_topic_pipeline
        return build_topic_pipeline(name.split(":", 1)[1])
    return make_component(name)


def combine_evidence(evidence, weights, pooling="evidence"):
    """Combine aligned evidence without learning from inference rows."""
    arrays = [np.asarray(values, dtype=float) for values in evidence]
    weights = np.asarray(weights, dtype=float)
    if not arrays or len(arrays) != len(weights):
        raise ValueError("One weight is required per component.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be positive and finite.")
    if pooling not in {"evidence", "probability"}:
        raise ValueError("Unknown pooling method.")
    shape = arrays[0].shape
    if len(shape) != 2 or shape[1] != len(LABELS):
        raise ValueError("Expected passage-by-class scores.")
    if any(a.shape != shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Component evidence must be finite and aligned.")
    if pooling == "probability":
        arrays = [softmax(a, axis=1) for a in arrays]
    return sum(a * w for a, w in zip(arrays, weights)) / weights.sum()


class V3Ensemble(ClassifierMixin, BaseEstimator):
    def __init__(self, components, weights, pooling="evidence"):
        self.components = components
        self.weights = weights
        self.pooling = pooling

    def fit(self, X, y, groups=None):
        # Check the recipe before starting any expensive fits.
        combine_evidence(
            [np.zeros((1, len(LABELS))) for _ in self.components],
            self.weights, self.pooling,
        )
        self.classes_ = np.asarray(LABELS)
        self.models_ = []
        for name in self.components:
            model = make_v3_component(name)
            if getattr(model, "requires_groups", False):
                if groups is None:
                    raise ValueError("This component needs training prompt groups.")
                model.fit(X, y, groups=groups)
            else:
                model.fit(X, y)
            if tuple(model.classes_) != LABELS:
                raise ValueError("Component class order differs from the competition labels.")
            self.models_.append(model)
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["classes_", "models_"])
        evidence = []
        for model in self.models_:
            probability = hasattr(model, "predict_proba")
            raw = model.predict_proba(X) if probability else model.decision_function(X)
            evidence.append(normalize_evidence(raw, probabilities=probability))
        return combine_evidence(evidence, self.weights, self.pooling)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]
