"""Checks for leakage, runtime test handling, and competition file contracts."""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from name_that_llm.baseline import (
    LABELS,
    BaselineConfig,
    build_pipeline,
    fit_checked,
    load_data,
    make_folds,
    predict_submission,
    validate_submission,
)


@pytest.fixture
def training_frame():
    rows = [
        {
            "id": f"{group:03d}{label_index:03d}",
            "prompt_id": f"prompt-{group}",
            "text": f"A passage for topic {group}, with punctuation—preserved.",
            "model": label,
        }
        for group in range(10)
        for label_index, label in enumerate(LABELS)
        if not (group == 0 and label == "claude-fable-5")
    ]
    frame = pd.DataFrame(rows)
    # Positional folds must work independently of a DataFrame's index labels.
    frame.index = np.arange(len(frame)) * 7 + 100
    return frame


def test_folds_keep_whole_prompts_and_cover_every_passage(training_frame):
    folds = make_folds(training_frame, BaselineConfig())
    visits = np.zeros(len(training_frame), dtype=int)
    for fitting, held_out in folds:
        fit_prompts = set(training_frame.iloc[fitting]["prompt_id"])
        held_out_prompts = set(training_frame.iloc[held_out]["prompt_id"])
        assert fit_prompts.isdisjoint(held_out_prompts)
        assert set(fitting).isdisjoint(held_out)
        assert len(fitting) + len(held_out) == len(training_frame)
        assert set(training_frame.iloc[held_out]["model"]) == set(LABELS)
        visits[held_out] += 1
    assert np.all(visits == 1)
    repeated = make_folds(training_frame, BaselineConfig())
    for first, second in zip(folds, repeated, strict=True):
        np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize("variant", ["char", "char_word"])
def test_prediction_does_not_learn_held_out_vocabulary_or_idf(variant):
    pipeline = build_pipeline(BaselineConfig(variant=variant, min_df=1))
    text = pd.Series(
        ["UPPER Café—warm prose.", "UPPER Café—gentle prose.",
         "lower plain terse text.", "lower plain compact text."]
    )
    fit_checked(pipeline, text, pd.Series([LABELS[0]] * 2 + [LABELS[1]] * 2))
    features = dict(pipeline.named_steps["features"].transformer_list)
    original = {
        name: (vectorizer.vocabulary_.copy(), vectorizer.idf_.copy())
        for name, vectorizer in features.items()
    }
    pipeline.predict(pd.Series(["🦊🦊🦊 heldoutuniqueword alone."]))
    for name, vectorizer in features.items():
        assert vectorizer.vocabulary_ == original[name][0]
        np.testing.assert_array_equal(vectorizer.idf_, original[name][1])
        assert "🦊🦊🦊" not in vectorizer.vocabulary_
        assert "heldoutuniqueword" not in vectorizer.vocabulary_
    assert "UPP" in features["char"].vocabulary_
    assert "fé—" in features["char"].vocabulary_


def test_load_data_preserves_leading_zero_ids_and_exact_csv_text(tmp_path, training_frame):
    training_frame.to_csv(tmp_path / "train.csv", index=False)
    expected_test = pd.DataFrame(
        {"id": ["000001", "000002"], "text": ["NA", 'Café—"quoted"\nsecond line']}
    )
    expected_test.to_csv(tmp_path / "test.csv", index=False)
    train, test = load_data(tmp_path)
    assert train["id"].tolist() == training_frame["id"].tolist()
    pd.testing.assert_frame_equal(test, expected_test)


@pytest.mark.parametrize("size", [1, 7, 4000])
def test_prediction_handles_runtime_size_order_and_nondefault_index(size, tmp_path):
    test = pd.DataFrame(
        {
            "id": [f"{number:012d}" for number in range(size, 0, -1)],
            "text": [f"Text passage {number}." for number in range(size)],
        },
        index=np.arange(size) * 3 + 80,
    )
    expected_labels = np.array([LABELS[number % len(LABELS)] for number in range(size)])

    class RecordingPredictor:
        def predict(self, text):
            pd.testing.assert_series_equal(text, test["text"])
            return expected_labels

    submission = predict_submission(RecordingPredictor(), test)
    assert submission["id"].tolist() == test["id"].tolist()
    assert submission["model"].tolist() == expected_labels.tolist()
    path = tmp_path / "submission.parquet"
    submission.to_parquet(path, index=False)
    round_trip = pd.read_parquet(path)
    validate_submission(test, round_trip)
    pd.testing.assert_frame_equal(submission, round_trip)


@pytest.mark.parametrize(
    "defect",
    ["missing_row", "duplicate_id", "wrong_order", "unknown_label", "null_label", "extra_column"],
)
def test_submission_rejects_invalid_competition_outputs(defect):
    test = pd.DataFrame({"id": ["0001", "0002"], "text": ["One passage.", "Another passage."]})
    submission = pd.DataFrame({"id": test["id"], "model": [LABELS[0], LABELS[1]]})
    if defect == "missing_row":
        submission = submission.iloc[:1]
    elif defect == "duplicate_id":
        submission.loc[1, "id"] = "0001"
    elif defect == "wrong_order":
        submission = submission.iloc[::-1]
    elif defect == "unknown_label":
        submission.loc[1, "model"] = "not-a-competition-label"
    elif defect == "null_label":
        submission.loc[1, "model"] = None
    elif defect == "extra_column":
        submission["confidence"] = 1.0
    with pytest.raises(ValueError):
        validate_submission(test, submission)


def test_convergence_failure_stops_training():
    class UnconvergedEstimator:
        def fit(self, text, labels):
            warnings.warn("Solver did not converge", ConvergenceWarning)
            return self

    with pytest.raises(RuntimeError, match="did not converge"):
        fit_checked(
            UnconvergedEstimator(),
            pd.Series(["one", "two"]),
            pd.Series([LABELS[0], LABELS[1]]),
        )
