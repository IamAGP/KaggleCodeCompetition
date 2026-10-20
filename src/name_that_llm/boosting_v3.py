"""Three fixed CPU LightGBM recipes using the existing style measurements.

DART motivation: https://arxiv.org/html/1505.01866v1.
Official implementation: https://github.com/lightgbm-org/LightGBM.
This bounded experiment is not a reproduction or a state-of-the-art claim.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

from name_that_llm.stylometry import StyleFeatures


BOOSTING_CANDIDATES = {
    "lightgbm7": {
        "boosting_type": "gbdt",
        "n_estimators": 400,
        "learning_rate": .035,
        "num_leaves": 7,
        "max_depth": 4,
        "min_child_samples": 25,
        "reg_lambda": 3.,
        "colsample_bytree": .8,
    },
    "lightgbm15": {
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": .04,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 30,
        "reg_lambda": 10.,
        "colsample_bytree": .7,
    },
    "dart7": {
        "boosting_type": "dart",
        "n_estimators": 300,
        "learning_rate": .08,
        "num_leaves": 7,
        "max_depth": 4,
        "min_child_samples": 25,
        "reg_lambda": 3.,
        "colsample_bytree": .8,
        "drop_rate": .03,
        "skip_drop": .5,
        "max_drop": 30,
    },
}

BOOSTING_COMMON = {
    "objective": "multiclass",
    "n_jobs": 2,
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
    "subsample": 1.,
    "subsample_freq": 0,
    "max_bin": 127,
    "reg_alpha": 0.,
}


def build_boosting_pipeline(name):
    """Return a fresh pipeline whose only learned component sees fit rows."""
    if name not in BOOSTING_CANDIDATES:
        raise ValueError(f"Unknown boosting recipe: {name}")
    return Pipeline([
        ("style", StyleFeatures()),
        ("classifier", LGBMClassifier(
            **BOOSTING_COMMON, **BOOSTING_CANDIDATES[name],
        )),
    ])
