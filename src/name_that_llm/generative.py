"""Small, predeclared count-based attribution models for improvement-v2.

The NB-SVM variant implements the log-count-ratio weighting in Wang and
Manning (2012), with character features and one-versus-rest multiclass fitting.
It does not use their optional coefficient interpolation. This is an adaptation,
not a reproduction of their sentiment experiment.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted


LOGGER = logging.getLogger("name_that_llm.generative")


@dataclass(frozen=True)
class GenerativeConfig:
    """Three candidates are fixed before any improvement-v2 validation scores."""

    variant: str = "char_nbsvm"
    alpha: float = 1.0
    c: float = 1.0
    min_df: int = 2
    max_features: int = 250_000
    max_iter: int = 10_000
    tolerance: float = 1e-4
    seed: int = 42


GENERATIVE_CONFIGS = tuple(
    GenerativeConfig(variant=variant)
    for variant in ("char_mnb", "char_cnb", "char_nbsvm")
)


class GenerativeAttribution(ClassifierMixin, BaseEstimator):
    """Text-to-score estimator; only ``fit`` constructs a vocabulary.

    ``decision_function`` columns follow ``classes_`` (lexical label order).
    Scores are uncalibrated. NB scores are per-feature average log evidence;
    this preserves each row's predicted class while reducing length dependence.
    """

    def __init__(self, config: GenerativeConfig | None = None):
        self.config = config

    def fit(self, text, labels):
        config = self.config or GenerativeConfig()
        if config.variant not in {c.variant for c in GENERATIVE_CONFIGS}:
            raise ValueError(f"Unknown generative variant: {config.variant}")
        if config.alpha <= 0:
            raise ValueError("Smoothing must be strictly positive.")
        labels = np.asarray(labels)
        self.classes_ = np.unique(labels)
        if len(self.classes_) < 2:
            raise ValueError("At least two labels are required.")
        self.vectorizer_ = CountVectorizer(
            analyzer="char",
            ngram_range=(3, 5) if config.variant == "char_nbsvm" else (2, 5),
            lowercase=False, strip_accents=None, binary=True,
            min_df=config.min_df, max_features=config.max_features,
            dtype=np.float64,
        )
        counts = self.vectorizer_.fit_transform(text)
        self.variant_ = config.variant
        self.solver_iterations_ = []
        if config.variant == "char_mnb":
            self.classifier_ = MultinomialNB(alpha=config.alpha, fit_prior=False)
            self.classifier_.fit(counts, labels)
        elif config.variant == "char_cnb":
            self.classifier_ = ComplementNB(
                alpha=config.alpha, fit_prior=False, norm=True,
            )
            self.classifier_.fit(counts, labels)
        else:
            total = np.asarray(counts.sum(axis=0)).ravel()
            self.coef_ = np.empty((len(self.classes_), counts.shape[1]))
            self.intercept_ = np.empty(len(self.classes_))
            for index, label in enumerate(self.classes_):
                target = labels == label
                positive_count = np.asarray(counts[target].sum(axis=0)).ravel()
                positive = positive_count + config.alpha
                negative = total - positive_count + config.alpha
                ratio = np.log(positive / positive.sum()) - np.log(
                    negative / negative.sum()
                )
                weighted = counts.multiply(ratio).tocsr()
                classifier = LinearSVC(
                    C=config.c, dual=True, random_state=config.seed,
                    tol=config.tolerance, max_iter=config.max_iter,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    classifier.fit(weighted, target)
                for warning in caught:
                    LOGGER.warning("%s label=%s %s: %s", config.variant, label,
                                   warning.category.__name__, warning.message)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    raise RuntimeError(f"NB-SVM did not converge for {label}.")
                # Fold the NB weighting into coefficients for cheap inference.
                self.coef_[index] = classifier.coef_[0] * ratio
                self.intercept_[index] = classifier.intercept_[0]
                self.solver_iterations_.append(int(classifier.n_iter_))
        self.n_features_in_ = counts.shape[1]
        return self

    def decision_function(self, text):
        check_is_fitted(self, ["classes_", "vectorizer_", "variant_"])
        counts = self.vectorizer_.transform(text)
        if self.variant_ == "char_nbsvm":
            scores = np.asarray(counts @ self.coef_.T) + self.intercept_
        else:
            scores = np.asarray(counts @ self.classifier_.feature_log_prob_.T)
            if self.variant_ == "char_mnb":
                scores += self.classifier_.class_log_prior_
            denominator = np.maximum(
                np.asarray(counts.sum(axis=1)).ravel(), 1.0,
            )
            scores /= denominator[:, None]
        if not np.isfinite(scores).all():
            raise ValueError("Model produced non-finite scores.")
        return scores

    def predict(self, text):
        return self.classes_[self.decision_function(text).argmax(axis=1)]


def build_generative(config: GenerativeConfig) -> GenerativeAttribution:
    return GenerativeAttribution(config)
