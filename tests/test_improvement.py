"""Leakage and score-combination contracts for the separately recorded v2."""

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from name_that_llm.baseline import LABELS
from name_that_llm.ensemble_v2 import EvidenceEnsemble, normalize_evidence
from name_that_llm.generative import GenerativeAttribution, GenerativeConfig
from name_that_llm.sparse_v2 import SparseConfig, build_sparse_pipeline
from name_that_llm.stylometry import StyleFeatures


@pytest.mark.parametrize("probabilities", [False, True])
def test_evidence_is_rowwise_and_preserves_individual_argmax(probabilities):
    rng = np.random.default_rng(807)
    raw = rng.normal(size=(7, len(LABELS)))
    if probabilities:
        raw = np.exp(raw)
        raw /= raw.sum(axis=1, keepdims=True)
    combined_batch = normalize_evidence(raw, probabilities=probabilities)
    separate_rows = np.vstack(
        [normalize_evidence(row[None, :], probabilities=probabilities) for row in raw]
    )
    np.testing.assert_allclose(combined_batch, separate_rows, atol=1e-14)
    np.testing.assert_array_equal(combined_batch.argmax(axis=1), raw.argmax(axis=1))
    np.testing.assert_allclose(combined_batch.mean(axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(combined_batch.std(axis=1), 1.0, atol=1e-14)


def test_margin_normalization_is_invariant_to_positive_row_scaling_and_offsets():
    raw = np.random.default_rng(23).normal(size=(3, len(LABELS)))
    transformed = raw * np.array([0.5, 5.0, 20.0])[:, None] + np.array([100, -5, 0])[:, None]
    np.testing.assert_allclose(normalize_evidence(raw), normalize_evidence(transformed), atol=1e-12)
    np.testing.assert_array_equal(
        normalize_evidence(np.ones((2, len(LABELS)))),
        np.zeros((2, len(LABELS))),
    )


@pytest.mark.parametrize(
    ("scores", "probabilities"),
    [
        (np.ones((2, len(LABELS) - 1)), False),
        (np.full((2, len(LABELS)), np.nan), False),
        (np.ones((2, len(LABELS))), True),
        (np.full((2, len(LABELS)), -1.0 / len(LABELS)), True),
    ],
)
def test_evidence_rejects_malformed_scores(scores, probabilities):
    with pytest.raises(ValueError):
        normalize_evidence(scores, probabilities=probabilities)


def test_style_fit_ignores_labels_and_other_passages():
    text = ["If this works—why? We agree.", "", "Café, naïve, 42; “yes!”"]
    features = StyleFeatures()
    before_fit = features.transform(text)
    features.fit(["Unrelated training passage."], ["arbitrary-label"])
    after_fit = features.transform(text)
    features.fit(["Entirely different corpus."], ["different-label"])
    np.testing.assert_array_equal(before_fit, after_fit)
    np.testing.assert_array_equal(before_fit, features.transform(text))
    np.testing.assert_array_equal(
        before_fit,
        np.vstack([features.transform([row]) for row in text]),
    )
    names = features.get_feature_names_out()
    assert len(set(names)) == len(names) == before_fit.shape[1]
    assert np.isfinite(before_fit).all()
    # No vocabulary, learned means, labels, or training passages are retained.
    assert vars(features) == {}


@pytest.fixture
def tiny_binary_training():
    return (
        ["Alpha—warm careful prose.", "Alpha—gentle careful prose.",
         "Beta plain compact text.", "Beta plain terse text."],
        ["zeta", "zeta", "alpha", "alpha"],
    )


@pytest.mark.parametrize("variant", ["char_mnb", "char_cnb", "char_nbsvm"])
def test_generative_predict_keeps_vocabulary_and_class_order(variant, tiny_binary_training):
    text, labels = tiny_binary_training
    model = GenerativeAttribution(GenerativeConfig(variant=variant, min_df=1))
    model.fit(text, labels)
    assert tuple(model.classes_) == ("alpha", "zeta")
    vocabulary = model.vectorizer_.vocabulary_.copy()
    test = ["Alpha—warm prose.", "🦊🦊🦊 neverseenword", "Beta terse text."]
    batched = model.decision_function(test)
    individual = np.vstack([model.decision_function([row]) for row in test])
    np.testing.assert_allclose(batched, individual, atol=1e-12)
    np.testing.assert_array_equal(model.predict(test), model.classes_[batched.argmax(axis=1)])
    assert vocabulary == model.vectorizer_.vocabulary_
    assert "🦊🦊🦊" not in vocabulary
    assert np.isfinite(batched).all()


def test_sparse_predict_leaves_fold_vocabulary_and_idf_frozen(tiny_binary_training):
    text, labels = tiny_binary_training
    model = build_sparse_pipeline(SparseConfig("test-only", 1, 5, 4.0, min_df=1))
    model.fit(text, labels)
    vectorizer = model.named_steps["features"]
    vocabulary, idf = vectorizer.vocabulary_.copy(), vectorizer.idf_.copy()
    model.predict(["🦊🦊🦊 validation-only vocabulary", "A brand-new passage."])
    assert vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(vectorizer.idf_, idf)
    assert "🦊" not in vectorizer.vocabulary_


def test_ensemble_rejects_reversed_component_class_order(monkeypatch):
    class WronglyOrderedModel:
        def fit(self, X, y):
            self.classes_ = np.asarray(LABELS[::-1])
            return self

    monkeypatch.setattr(
        "name_that_llm.ensemble_v2.make_component",
        lambda name: WronglyOrderedModel(),
    )
    with pytest.raises(ValueError, match="class order"):
        EvidenceEnsemble(["test:component"], [1.0]).fit(["one", "two"], LABELS[:2])


@pytest.fixture(scope="module")
def fitted_winning_recipe():
    # Synthetic integration check only; it does not rerun competition selection.
    passages, labels = [], []
    for index, label in enumerate(reversed(LABELS)):
        for repeat in range(4):
            passages.append(
                f"Source {chr(65 + index)} uses {'careful ' * (index % 4 + 1)}"
                f"words{'—' if index % 2 else ';'} example {repeat}. "
                f"{'Why ask?' if index % 3 else 'We explain.'}"
            )
            labels.append(label)
    with threadpool_limits(limits=1):
        model = EvidenceEnsemble(
            ["sparse:char15_c4", "style:style_hist"], [1, 1],
        ).fit(pd.Series(passages), pd.Series(labels))
        yield model


def test_winning_recipe_prediction_matches_selected_blend_math(fitted_winning_recipe):
    model = fitted_winning_recipe
    text = pd.Series(
        ["A new explanation—carefully stated.", "We explain another case.", "Why ask?"],
        index=[90, 20, 77],
    )
    component_evidence = []
    for component in model.models_:
        probabilities = hasattr(component, "predict_proba")
        raw = component.predict_proba(text) if probabilities else component.decision_function(text)
        component_evidence.append(normalize_evidence(raw, probabilities=probabilities))
    expected = (component_evidence[0] + component_evidence[1]) / 2
    np.testing.assert_allclose(model.decision_function(text), expected, atol=1e-12)
    np.testing.assert_array_equal(model.predict(text), np.asarray(LABELS)[expected.argmax(axis=1)])
    assert tuple(model.classes_) == LABELS


def test_winning_recipe_scores_do_not_depend_on_batch_composition(fitted_winning_recipe):
    model = fitted_winning_recipe
    text = ["One new passage.", "An unrelated passage—twice.", "What happens next?"]
    batched = model.decision_function(text)
    individual = np.vstack([model.decision_function([row]) for row in text])
    np.testing.assert_allclose(batched, individual, atol=1e-12)
    reordered = model.decision_function([text[2], text[0], text[2], text[1]])
    np.testing.assert_allclose(reordered, batched[[2, 0, 2, 1]], atol=1e-12)
