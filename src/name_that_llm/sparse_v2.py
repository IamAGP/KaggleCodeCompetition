"""Predeclared sparse-feature variants for the second experiment."""

from dataclasses import asdict, dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import numpy as np


@dataclass(frozen=True)
class SparseConfig:
    name: str
    ngram_min: int
    ngram_max: int
    c: float
    min_df: int = 2
    max_features: int = 300_000
    seed: int = 42


SPARSE_CANDIDATES = (
    SparseConfig("char15_c4", 1, 5, 4.0),
    SparseConfig("char26_c4", 2, 6, 4.0),
    SparseConfig("char35_c4", 3, 5, 4.0),
    SparseConfig("char15_c16", 1, 5, 16.0),
)


def build_sparse_pipeline(config: SparseConfig) -> Pipeline:
    return Pipeline([
        ("features", TfidfVectorizer(
            analyzer="char", ngram_range=(config.ngram_min, config.ngram_max),
            lowercase=False, strip_accents=None, min_df=config.min_df,
            max_features=config.max_features, sublinear_tf=True, dtype=np.float32,
        )),
        ("classifier", LinearSVC(
            C=config.c, dual=True, random_state=config.seed,
            max_iter=10_000, tol=1e-4,
        )),
    ])
