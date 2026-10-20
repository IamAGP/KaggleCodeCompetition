"""Joint multinomial text/style models for the frozen v3 second stage."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from name_that_llm.stylometry import StyleFeatures


LINEAR_CANDIDATES = {
    "joint_char_style_c4": {"C": 4.},
    "joint_char_style_c16": {"C": 16.},
}

LINEAR_COMMON = {
    "char_features": {
        "analyzer": "char", "ngram_range": [1, 5], "min_df": 2,
        "max_features": 300000, "lowercase": False, "sublinear_tf": True,
        "dtype": "float64",
    },
    "style_features": "465 StyleFeatures -> StandardScaler -> L2 Normalizer",
    "union_weights": {"char": 1., "style": .5},
    "union_postprocess": "L2 Normalizer",
    "classifier": {
        "solver": "lbfgs", "max_iter": 1500, "tol": 1e-4,
        "loss": "joint multinomial negative log likelihood",
        "penalty": "default L2", "fit_intercept": True,
    },
}


def build_linear_pipeline(name):
    """Fit character vocabulary, style scaler and joint classifier on fit rows."""
    if name not in LINEAR_CANDIDATES:
        raise ValueError(f"Unknown linear recipe: {name}")
    return Pipeline([
        ("features", FeatureUnion([
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(1, 5), min_df=2,
                max_features=300000, lowercase=False, strip_accents=None,
                sublinear_tf=True, dtype=np.float64,
            )),
            ("style", Pipeline([
                ("extract", StyleFeatures()),
                ("scale", StandardScaler()),
                ("normalize", Normalizer()),
            ])),
        ], transformer_weights={"char": 1., "style": .5})),
        ("normalize", Normalizer()),
        ("classifier", LogisticRegression(
            C=LINEAR_CANDIDATES[name]["C"],
            solver="lbfgs", max_iter=1500, tol=1e-4,
        )),
    ])
