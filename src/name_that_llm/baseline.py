"""Shared training and submission code, also embedded in the Kaggle notebook."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC

LOGGER = logging.getLogger("name_that_llm")
LABELS = tuple(sorted((
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash",
    "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-5", "qwen3.8-max", "glm-5.3",
    "glm-5.3-flash", "kimi-k3",
)))


@dataclass(frozen=True)
class BaselineConfig:
    """Fixed before inspecting any validation results."""

    variant: str = "char"
    seed: int = 42
    n_splits: int = 5
    c: float = 1.0
    min_df: int = 2
    char_max_features: int = 250_000
    word_max_features: int = 100_000
    max_iter: int = 10_000
    tolerance: float = 1e-4


def validate_frame(frame: pd.DataFrame, *, training: bool) -> None:
    required = ["id", "text"] + (["prompt_id", "model"] if training else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input must contain at least one passage.")
    for column in required:
        if frame[column].isna().any() or not frame[column].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all():
            raise ValueError(f"Column {column!r} contains missing, empty or non-string values.")
    if frame["id"].duplicated().any():
        raise ValueError("Passage IDs must be unique.")
    if training and set(frame["model"]) != set(LABELS):
        raise ValueError("Training labels must match all 18 competition labels.")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Explicit string types retain leading zeros and the exact CSV-decoded text.
    train = pd.read_csv(data_dir / "train.csv", dtype=str, keep_default_na=False)
    test = pd.read_csv(data_dir / "test.csv", dtype=str, keep_default_na=False)
    validate_frame(train, training=True)
    validate_frame(test, training=False)
    return train, test


def build_pipeline(config: BaselineConfig) -> Pipeline:
    if config.variant not in {"char", "char_word"}:
        raise ValueError(f"Unknown baseline variant: {config.variant}")
    common = dict(min_df=config.min_df, sublinear_tf=True, dtype=np.float32)
    features = [("char", TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), lowercase=False,
        strip_accents=None, max_features=config.char_max_features, **common,
    ))]
    if config.variant == "char_word":
        features.append(("word", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), lowercase=True,
            strip_accents=None, stop_words=None,
            max_features=config.word_max_features, **common,
        )))
    return Pipeline([
        ("features", FeatureUnion(features)),
        ("normalize", Normalizer()),
        ("classifier", LinearSVC(
            C=config.c, dual=True, random_state=config.seed,
            max_iter=config.max_iter, tol=config.tolerance,
        )),
    ])


def make_folds(train: pd.DataFrame, config: BaselineConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    validate_frame(train, training=True)
    splitter = GroupKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.seed,
    )
    folds = list(splitter.split(train["text"], train["model"], groups=train["prompt_id"]))
    counts = np.zeros(len(train), dtype=int)
    for fitting, held_out in folds:
        if set(train.iloc[fitting]["prompt_id"]) & set(train.iloc[held_out]["prompt_id"]):
            raise ValueError("A prompt appears in both sides of a fold.")
        for indexes in (fitting, held_out):
            if set(train.iloc[indexes]["model"]) != set(LABELS):
                raise ValueError("Every fold must contain all 18 labels on both sides.")
        counts[held_out] += 1
    if not np.all(counts == 1):
        raise ValueError("Each passage must be held out exactly once.")
    return folds


def fit_checked(pipeline: Pipeline, text: pd.Series, labels: pd.Series) -> Pipeline:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(text, labels)
    for warning in caught:
        LOGGER.warning("%s: %s", warning.category.__name__, warning.message)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("LinearSVC did not converge; inspect the run log before proceeding.")
    return pipeline


def validate_submission(
    test: pd.DataFrame, submission: pd.DataFrame,
) -> None:
    validate_frame(test, training=False)
    if list(submission.columns) != ["id", "model"]:
        raise ValueError("Submission columns must be exactly id, model.")
    if len(submission) != len(test):
        raise ValueError("Submission row count differs from the runtime test set.")
    if submission.isna().any().any() or submission["id"].duplicated().any():
        raise ValueError("Submission contains missing values or duplicate IDs.")
    if submission["id"].tolist() != test["id"].tolist():
        raise ValueError("Submission IDs must match runtime test IDs in order.")
    if not set(submission["model"]).issubset(LABELS):
        raise ValueError("Submission contains an unknown model label.")


def predict_submission(pipeline: Pipeline, test: pd.DataFrame) -> pd.DataFrame:
    validate_frame(test, training=False)
    predictions = pipeline.predict(test["text"])
    submission = pd.DataFrame({"id": test["id"].to_numpy(), "model": predictions})
    validate_submission(test, submission)
    return submission


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


class BaselineExperiment:
    """Compare two predeclared candidates on a single fixed prompt split."""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train, test = load_data(self.data_dir)
        configs = [BaselineConfig(variant=v) for v in ("char", "char_word")]
        folds = make_folds(train, configs[0])
        assignment = train[["id", "prompt_id"]].copy()
        assignment["fold"] = -1
        for fold, (_, held_out) in enumerate(folds):
            assignment.loc[assignment.index[held_out], "fold"] = fold
        assignment.to_csv(self.output_dir / "folds.csv", index=False)
        summary = {
            "selection_metric": "pooled_oof_balanced_accuracy",
            "tie_break": "char (simpler candidate)",
            "versions": runtime_versions(),
            "train_sha256": file_sha256(self.data_dir / "train.csv"),
            "train_rows": len(train), "prompt_count": train["prompt_id"].nunique(),
            "labels": list(LABELS), "candidates": {},
            "limitations": [
                "Prompt groups can still share related topics.",
                "Features were motivated by a label-visible review of all training data.",
                "Selecting the better of two candidates can make its CV estimate optimistic.",
                "Visible test predictions are a format check, not a scored evaluation.",
            ],
        }
        for config in configs:
            oof = np.empty(len(train), dtype=object)
            fold_metrics = []
            candidate_dir = self.output_dir / config.variant
            candidate_dir.mkdir(exist_ok=True)
            for fold, (fitting, held_out) in enumerate(folds):
                LOGGER.info("Start %s fold=%d train=%d valid=%d",
                            config.variant, fold, len(fitting), len(held_out))
                started = time.perf_counter()
                pipeline = fit_checked(
                    build_pipeline(config), train.iloc[fitting]["text"],
                    train.iloc[fitting]["model"],
                )
                oof[held_out] = pipeline.predict(train.iloc[held_out]["text"])
                score = balanced_accuracy_score(train.iloc[held_out]["model"], oof[held_out])
                feature_sizes = {
                    name: len(vectorizer.vocabulary_)
                    for name, vectorizer in pipeline.named_steps["features"].transformer_list
                }
                result = {
                    "fold": fold, "balanced_accuracy": float(score),
                    "train_rows": len(fitting), "validation_rows": len(held_out),
                    "seconds": time.perf_counter() - started,
                    "feature_counts": feature_sizes,
                    "solver_iterations": int(pipeline.named_steps["classifier"].n_iter_),
                }
                fold_metrics.append(result)
                LOGGER.info("Done %s %s", config.variant, json.dumps(result))
            predictions = train[["id", "prompt_id", "model"]].copy()
            predictions["fold"] = assignment["fold"]
            predictions["prediction"] = oof
            predictions.to_csv(candidate_dir / "oof_predictions.csv", index=False)
            matrix = confusion_matrix(train["model"], oof, labels=LABELS)
            pd.DataFrame(matrix, index=LABELS, columns=LABELS).rename_axis("true_model").to_csv(
                candidate_dir / "confusion_matrix.csv"
            )
            recalls = recall_score(train["model"], oof, labels=LABELS, average=None)
            per_class = pd.DataFrame({
                "model": LABELS, "recall": recalls,
                "support": matrix.sum(axis=1),
                "correct": matrix.diagonal(),
            })
            per_class.to_csv(candidate_dir / "per_class_recall.csv", index=False)
            scores = [result["balanced_accuracy"] for result in fold_metrics]
            summary["candidates"][config.variant] = {
                "config": asdict(config),
                "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(train["model"], oof)),
                "fold_mean": float(np.mean(scores)),
                "fold_standard_deviation": float(np.std(scores, ddof=1)),
                "folds": fold_metrics,
                "per_class": per_class.to_dict(orient="records"),
            }
        selected = max(
            summary["candidates"],
            key=lambda name: summary["candidates"][name]["pooled_oof_balanced_accuracy"],
        )
        summary["selected_variant"] = selected
        # Write the comparison before the separate full-data training step.
        self._write_summary(summary)
        LOGGER.info("Selected %s; refitting on all %d training rows", selected, len(train))
        config = BaselineConfig(**summary["candidates"][selected]["config"])
        started = time.perf_counter()
        pipeline = fit_checked(build_pipeline(config), train["text"], train["model"])
        submission = predict_submission(pipeline, test)
        submission_path = self.output_dir / "submission.parquet"
        submission.to_parquet(submission_path, index=False)
        validate_submission(test, pd.read_parquet(submission_path))
        summary["full_fit_and_preview_seconds"] = time.perf_counter() - started
        summary["visible_test_rows"] = len(test)
        summary["preview_submission_sha256"] = file_sha256(submission_path)
        self._write_summary(summary)
        LOGGER.info("Preview submission verified: rows=%d; this is not a leaderboard score.", len(test))
        return summary

    def _write_summary(self, summary: dict) -> None:
        (self.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
