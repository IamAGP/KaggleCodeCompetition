"""Training-only prompt nuisance removal and fixed content masking for v3.

The masking experiment is motivated by arXiv:2104.08530, not a reproduction.
Prompt-centered ridge is a separately derived hypothesis: train within prompts,
then predict individual passages relative to the training global feature mean.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.linalg import solve
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

from name_that_llm.stylometry import FUNCTION_WORDS, StyleFeatures


TOPIC_CANDIDATES = {
    "centered_ridge_half": {
        "centering": 0.5, "ridge_alpha": 1.0,
        "features": "raw char3-5 TFIDF + 0.5 normalized standardized style",
    },
    "centered_ridge_full": {
        "centering": 1.0, "ridge_alpha": 1.0,
        "features": "raw char3-5 TFIDF + 0.5 normalized standardized style",
    },
    "masked_char_style_svm": {
        "mask": "retain fixed function words; other letters '*'; digits '#'",
        "C": 1.0,
        "features": "masked char3-5 TFIDF + 0.5 normalized standardized raw style",
    },
}

MASK_TOKEN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+", re.UNICODE)
KEEP_WORDS = frozenset(FUNCTION_WORDS)


class MaskContent(TransformerMixin, BaseEstimator):
    """Preserve punctuation, function words and word lengths, deterministically."""

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    @staticmethod
    def mask(text):
        if not isinstance(text, str):
            raise TypeError("Expected a passage string.")

        def replacement(match):
            word = match.group()
            if word.lower() in KEEP_WORDS:
                return word
            return "".join("#" if c.isdigit() else "*" if c.isalpha() else c for c in word)

        return MASK_TOKEN.sub(replacement, text)

    def transform(self, X):
        if isinstance(X, str):
            raise ValueError("Expected a sequence of passages.")
        return [self.mask(text) for text in X]


def build_topic_features(masked=False):
    char = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, max_features=250_000,
        lowercase=False, strip_accents=None, sublinear_tf=True, dtype=np.float64,
    )
    if masked:
        char = Pipeline([("mask", MaskContent()), ("tfidf", char)])
    return Pipeline([
        ("features", FeatureUnion([
            ("char", char),
            ("style", Pipeline([
                ("extract", StyleFeatures()), ("scale", StandardScaler()),
                ("normalize", Normalizer()),
            ])),
        ], transformer_weights={"char": 1.0, "style": 0.5})),
        ("normalize", Normalizer()),
    ])


def center_by_training_groups(values, group_indices, strength):
    """Apply A = I - strength*G - (1-strength)*J to rows.

    G averages rows inside each training prompt; J is the training global mean.
    These are symmetric projectors and A never uses labels or held-out rows.
    """
    matrix = np.asarray(values, dtype=np.float64)
    result = matrix - (1.0 - strength) * matrix.mean(axis=0, keepdims=True)
    for indices in group_indices:
        result[indices] -= strength * matrix[indices].mean(axis=0, keepdims=True)
    return result


class PromptCenteredRidge(ClassifierMixin, BaseEstimator):
    """Fit ridge on prompt-centered training features; infer with a global mean.

    Fit requires training prompt groups. Prediction deliberately cannot accept
    groups: a passage is scored without reference to any other test passage.
    The Gram implementation avoids densifying the sparse feature matrix.
    """

    requires_groups = True

    def __init__(self, strength=1.0, alpha=1.0):
        self.strength = strength
        self.alpha = alpha

    def fit(self, X, y, groups=None):
        if not 0.0 <= self.strength <= 1.0 or self.alpha <= 0:
            raise ValueError("Invalid centering strength or ridge regularization.")
        texts, labels = list(X), np.asarray(y)
        if groups is None:
            raise ValueError("Training prompt groups are required.")
        groups = np.asarray(groups)
        if labels.ndim != 1 or groups.ndim != 1 or not len(texts) == len(labels) == len(groups):
            raise ValueError("Texts, labels and groups must be aligned one-dimensional arrays.")
        self.classes_, label_indices = np.unique(labels, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Need at least two classes.")
        _, group_codes = np.unique(groups, return_inverse=True)
        group_indices = [np.flatnonzero(group_codes == code) for code in np.unique(group_codes)]
        if len(group_indices) < 2:
            raise ValueError("Need at least two training prompt groups.")
        self.features_ = build_topic_features()
        features = self.features_.fit_transform(texts, labels)
        targets = np.eye(len(self.classes_))[label_indices]
        prior = targets.mean(axis=0)
        targets -= prior
        gram = (features @ features.T).toarray()
        centered = center_by_training_groups(gram, group_indices, self.strength)
        centered = center_by_training_groups(centered.T, group_indices, self.strength).T
        # Roundoff can leave tiny asymmetry; restoring symmetry precedes the SPD solve.
        centered = (centered + centered.T) * 0.5
        centered.flat[::len(texts) + 1] += self.alpha
        dual = solve(centered, targets, assume_a="pos", check_finite=True)
        adjusted_dual = center_by_training_groups(dual, group_indices, self.strength)
        self.coef_ = np.asarray(features.T @ adjusted_dual).T
        feature_mean = np.asarray(features.mean(axis=0)).ravel()
        self.intercept_ = prior - self.coef_ @ feature_mean
        self.n_training_groups_ = len(group_indices)
        if not np.isfinite(self.coef_).all() or not np.isfinite(self.intercept_).all():
            raise ValueError("Non-finite fitted parameters.")
        return self

    def decision_function(self, X):
        check_is_fitted(self, ["features_", "classes_", "coef_", "intercept_"])
        return np.asarray(self.features_.transform(X) @ self.coef_.T) + self.intercept_

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def build_topic_pipeline(name):
    if name == "centered_ridge_half":
        return PromptCenteredRidge(strength=0.5, alpha=1.0)
    if name == "centered_ridge_full":
        return PromptCenteredRidge(strength=1.0, alpha=1.0)
    if name == "masked_char_style_svm":
        return Pipeline([
            ("features", build_topic_features(masked=True)),
            ("classifier", LinearSVC(C=1.0, max_iter=10000, dual=True, random_state=42, tol=1e-4)),
        ])
    raise ValueError(f"Unknown topic candidate: {name}")
