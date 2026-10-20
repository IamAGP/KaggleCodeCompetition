"""Two fixed nonlinear style-kernel hypotheses for the v3 second stage."""

from __future__ import annotations

import hashlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from name_that_llm.stylometry import StyleFeatures


KERNEL_CANDIDATES = {
    "style_rbf_c4": {
        "C": 4.0, "char_weight": 0.0, "style_weight": 1.0,
        "features": "465 fixed style features -> training-only StandardScaler",
        "gamma": "inverse median positive squared training style distance",
        "bandwidth_sample_size": 256, "selection": "lowest SHA256(text) values",
        "classifier": "OneVsRestClassifier(SVC(kernel=precomputed)), n_jobs=1",
    },
    "additive_char_style_rbf_c4": {
        "C": 4.0, "char_weight": 0.5, "style_weight": 0.5,
        "features": "char3-5 TFIDF cosine + RBF standardized fixed style features",
        "gamma": "inverse median positive squared training style distance",
        "bandwidth_sample_size": 256, "selection": "lowest SHA256(text) values",
        "classifier": "OneVsRestClassifier(SVC(kernel=precomputed)), n_jobs=1",
    },
}


def bandwidth_sample_indices(texts, max_samples=256):
    """Choose training passages reproducibly and independently of row order."""
    return np.asarray(sorted(
        range(len(texts)),
        key=lambda index: (hashlib.sha256(texts[index].encode("utf-8")).digest(), texts[index]),
    )[:max_samples], dtype=int)


def median_style_gamma(styles, indices):
    subset = np.asarray(styles, dtype=np.float64)[indices]
    squared = euclidean_distances(subset, squared=True)
    distances = squared[np.triu_indices(len(subset), k=1)]
    positive = distances[distances > 1e-12]
    median = float(np.median(positive)) if len(positive) else 1.0
    return 1.0 / median


def combine_style_char_kernel(styles_left, styles_right, *, gamma,
                              char_left=None, char_right=None, char_weight=0.0):
    """RBF style similarity plus optional normalized character inner product."""
    if not np.isfinite(gamma) or gamma <= 0 or not 0 <= char_weight <= 1:
        raise ValueError("Invalid kernel bandwidth or mixture weight.")
    squared = euclidean_distances(styles_left, styles_right, squared=True)
    kernel = np.exp(-gamma * squared)
    if char_weight:
        if char_left is None or char_right is None:
            raise ValueError("Character features are required for an additive kernel.")
        inner = char_left @ char_right.T
        if hasattr(inner, "toarray"):
            inner = inner.toarray()
        kernel = (1.0 - char_weight) * kernel + char_weight * np.asarray(inner)
    if not np.isfinite(kernel).all():
        raise ValueError("Non-finite kernel values.")
    return np.ascontiguousarray(kernel, dtype=np.float64)


class StyleKernelClassifier(ClassifierMixin, BaseEstimator):
    """Train-only kernel state; score each text against stored training features."""

    def __init__(self, C=4.0, char_weight=0.0, bandwidth_sample_size=256, batch_size=256):
        self.C = C
        self.char_weight = char_weight
        self.bandwidth_sample_size = bandwidth_sample_size
        self.batch_size = batch_size

    def fit(self, X, y):
        texts, labels = list(X), np.asarray(y)
        if self.C <= 0 or not 0 <= self.char_weight <= 1:
            raise ValueError("Invalid classifier configuration.")
        if self.bandwidth_sample_size < 2 or self.batch_size < 1:
            raise ValueError("Invalid bandwidth sample or prediction batch size.")
        if labels.ndim != 1 or len(texts) != len(labels) or len(texts) < 2:
            raise ValueError("Expected aligned texts and labels.")
        self.extractor_ = StyleFeatures()
        self.scaler_ = StandardScaler()
        self.training_styles_ = self.scaler_.fit_transform(self.extractor_.transform(texts))
        sample = bandwidth_sample_indices(texts, self.bandwidth_sample_size)
        self.gamma_ = median_style_gamma(self.training_styles_, sample)
        self.n_bandwidth_samples_ = len(sample)
        self.training_chars_ = None
        if self.char_weight:
            self.vectorizer_ = TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=2,
                max_features=250_000, lowercase=False, strip_accents=None,
                sublinear_tf=True, norm="l2", dtype=np.float64,
            )
            self.training_chars_ = self.vectorizer_.fit_transform(texts)
        gram = combine_style_char_kernel(
            self.training_styles_, self.training_styles_, gamma=self.gamma_,
            char_left=self.training_chars_, char_right=self.training_chars_,
            char_weight=self.char_weight,
        )
        self.classifier_ = OneVsRestClassifier(
            SVC(C=self.C, kernel="precomputed", tol=1e-3, cache_size=256,
                probability=False, max_iter=-1, random_state=42),
            n_jobs=1,
        )
        self.classifier_.fit(gram, labels)
        self.classes_ = self.classifier_.classes_
        if len(self.classes_) < 2:
            raise ValueError("Need at least two classes.")
        return self

    def _kernel(self, texts):
        styles = self.scaler_.transform(self.extractor_.transform(texts))
        chars = self.vectorizer_.transform(texts) if self.char_weight else None
        return combine_style_char_kernel(
            styles, self.training_styles_, gamma=self.gamma_,
            char_left=chars, char_right=self.training_chars_,
            char_weight=self.char_weight,
        )

    def decision_function(self, X):
        check_is_fitted(self, ["classifier_", "classes_", "gamma_"])
        if isinstance(X, str):
            raise ValueError("Expected a sequence of passages.")
        texts = list(X)
        if not texts:
            return np.empty((0, len(self.classes_)), dtype=np.float64)
        batches = []
        for start in range(0, len(texts), self.batch_size):
            scores = self.classifier_.decision_function(self._kernel(texts[start:start + self.batch_size]))
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
            batches.append(scores)
        return np.concatenate(batches)

    def predict(self, X):
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def build_kernel_pipeline(name):
    if name == "style_rbf_c4":
        return StyleKernelClassifier(C=4.0, char_weight=0.0)
    if name == "additive_char_style_rbf_c4":
        return StyleKernelClassifier(C=4.0, char_weight=0.5)
    raise ValueError(f"Unknown kernel candidate: {name}")
