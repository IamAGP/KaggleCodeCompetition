"""Contracts for frozen v3 selection, pooling, and training-only group use."""

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import softmax
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from name_that_llm.baseline import LABELS
from name_that_llm.ensemble_v2 import normalize_evidence
from name_that_llm.ensemble_v3 import V3Ensemble, combine_evidence, make_v3_component
from name_that_llm.topic_v3 import MaskContent, PromptCenteredRidge


@pytest.mark.parametrize("pooling", ["evidence", "probability"])
def test_pooling_matches_weighted_row_scores_and_is_batch_invariant(pooling):
    rng = np.random.default_rng(608)
    components = [rng.normal(size=(5, len(LABELS))) for _ in range(3)]
    weights = [1, 2, 1]
    expected = sum(
        weight * (softmax(scores, axis=1) if pooling == "probability" else scores)
        for weight, scores in zip(weights, components)
    ) / 4
    actual = combine_evidence(components, weights, pooling)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    if pooling == "probability":
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-14)
    separate = np.vstack([
        combine_evidence([scores[i:i + 1] for scores in components], weights, pooling)
        for i in range(5)
    ])
    np.testing.assert_allclose(actual, separate, atol=1e-14)
    order = [4, 0, 4, 2]
    np.testing.assert_allclose(
        combine_evidence([scores[order] for scores in components], weights, pooling),
        actual[order],
        atol=1e-14,
    )


@pytest.mark.parametrize("failure", ["shape", "nan", "weight", "count", "method"])
def test_pooling_rejects_malformed_inputs(failure):
    scores = [np.ones((3, len(LABELS))), np.ones((3, len(LABELS)))]
    weights, method = [1, 1], "evidence"
    if failure == "shape":
        scores[1] = scores[1][:, :-1]
    elif failure == "nan":
        scores[0][0, 0] = np.nan
    elif failure == "weight":
        weights = [1, 0]
    elif failure == "count":
        weights = [1]
    else:
        method = "unrecognized"
    with pytest.raises(ValueError):
        combine_evidence(scores, weights, method)


@pytest.mark.parametrize("pooling", ["evidence", "probability"])
def test_ensemble_routes_only_training_groups_and_preserves_runtime_pooling(monkeypatch, pooling):
    rng = np.random.default_rng(417)
    margins = rng.normal(size=(4, len(LABELS)))
    probabilities = softmax(rng.normal(size=(4, len(LABELS))), axis=1)
    observed = {}

    class GroupedMargins:
        requires_groups = True

        def fit(self, X, y, groups):
            observed["texts"], observed["labels"], observed["groups"] = list(X), list(y), list(groups)
            self.classes_ = np.asarray(LABELS)
            return self

        def decision_function(self, X):
            return margins[np.asarray(list(X), dtype=int)]

    class UngroupedProbabilities:
        def fit(self, X, y):
            observed["ungrouped_fit"] = True
            self.classes_ = np.asarray(LABELS)
            return self

        def predict_proba(self, X):
            return probabilities[np.asarray(list(X), dtype=int)]

    monkeypatch.setattr(
        "name_that_llm.ensemble_v3.make_v3_component",
        lambda name: GroupedMargins() if name == "grouped" else UngroupedProbabilities(),
    )
    train = pd.Series(["0", "1", "2", "3"], index=[40, 7, 10, 2])
    groups = np.asarray(["prompt-a", "prompt-b", "prompt-a", "prompt-b"])
    model = V3Ensemble(["grouped", "plain"], [1, 2], pooling).fit(train, LABELS[:4], groups=groups)
    assert observed["texts"] == train.tolist()
    assert observed["groups"] == groups.tolist()
    assert observed["labels"] == list(LABELS[:4])
    assert observed["ungrouped_fit"]
    expected = combine_evidence(
        [normalize_evidence(margins), normalize_evidence(probabilities, probabilities=True)],
        [1, 2], pooling,
    )
    np.testing.assert_allclose(model.decision_function(train), expected, atol=1e-14)
    np.testing.assert_array_equal(model.predict(train), np.asarray(LABELS)[expected.argmax(axis=1)])
    reordered = ["3", "0", "3", "1"]
    np.testing.assert_allclose(model.decision_function(reordered), expected[[3, 0, 3, 1]], atol=1e-14)
    with pytest.raises(ValueError, match="training prompt groups"):
        V3Ensemble(["grouped"], [1]).fit(train, LABELS[:4])


def test_v3_rejects_reordered_component_classes(monkeypatch):
    class WrongClasses:
        def fit(self, X, y):
            self.classes_ = np.asarray(LABELS[::-1])
            return self

    monkeypatch.setattr("name_that_llm.ensemble_v3.make_v3_component", lambda name: WrongClasses())
    with pytest.raises(ValueError, match="class order"):
        V3Ensemble(["wrong"], [1]).fit(["a", "b"], LABELS[:2])


@pytest.mark.parametrize("strength", [0.5, 1.0])
def test_centered_ridge_matches_independent_dense_primal_with_unbalanced_groups(monkeypatch, strength):
    rng = np.random.default_rng(291)
    features = rng.normal(size=(9, 5))
    labels = np.asarray(["a", "b", "a", "b", "c", "a", "b", "c", "c"])
    groups = np.asarray(["first"] * 2 + ["second"] * 3 + ["third"] * 4)

    class NumericFeatures:
        def fit_transform(self, X, y=None):
            return sparse.csr_matrix(X)

        def transform(self, X):
            return sparse.csr_matrix(X)

    monkeypatch.setattr("name_that_llm.topic_v3.build_topic_features", lambda: NumericFeatures())
    model = PromptCenteredRidge(strength=strength, alpha=1.7).fit(features, labels, groups)
    group_projection = np.zeros((len(groups), len(groups)))
    for name in np.unique(groups):
        indices = np.flatnonzero(groups == name)
        group_projection[np.ix_(indices, indices)] = 1 / len(indices)
    global_projection = np.ones_like(group_projection) / len(groups)
    centering = np.eye(len(groups)) - strength * group_projection - (1 - strength) * global_projection
    centered = centering @ features
    targets = np.eye(3)[np.searchsorted(model.classes_, labels)]
    prior = targets.mean(axis=0)
    expected_coef = np.linalg.solve(
        centered.T @ centered + 1.7 * np.eye(features.shape[1]),
        centered.T @ (targets - prior),
    ).T
    expected_intercept = prior - expected_coef @ features.mean(axis=0)
    np.testing.assert_allclose(model.coef_, expected_coef, atol=1e-12)
    np.testing.assert_allclose(model.intercept_, expected_intercept, atol=1e-12)
    query = rng.normal(size=(3, features.shape[1]))
    np.testing.assert_allclose(
        model.decision_function(query), query @ expected_coef.T + expected_intercept, atol=1e-12,
    )
    np.testing.assert_allclose(
        model.decision_function(query[:1]), model.decision_function(query)[0:1], atol=1e-12,
    )
    assert model.n_training_groups_ == 3


def test_masking_is_stateless_preserves_function_words_punctuation_and_lengths():
    passages = ["The cobalt—21, isn't here.", "Café, naïve? 東京!", ""]
    mask = MaskContent()
    expected = mask.transform(passages)
    assert expected[0] == "The ******—##, ***'* here."
    assert all(len(first) == len(second) for first, second in zip(passages, expected))
    mask.fit(["Completely different training data."], ["unrelated-label"])
    assert mask.transform(passages) == expected
    assert vars(mask) == {}
    with pytest.raises(ValueError, match="sequence"):
        mask.transform("one passage")


@pytest.mark.parametrize("name", ["centered_ridge_half", "centered_ridge_full", "masked_char_style_svm"])
def test_topic_factory_uses_training_only_transform_and_is_cloneable(name):
    text = [
        "The red kite is here. Yet it is calm.", "The blue kite is here; indeed it is calm.",
        "The red boat is here. Yet it is calm.", "The blue boat is here; indeed it is calm.",
        "The red cloud is here. Yet it is calm.", "The blue cloud is here; indeed it is calm.",
    ]
    labels = np.asarray(["a", "b"] * 3)
    groups = np.repeat(np.arange(3), 2)
    model = clone(make_v3_component("topic:" + name))
    kwargs = {"groups": groups} if getattr(model, "requires_groups", False) else {}
    with threadpool_limits(limits=1):
        model.fit(text, labels, **kwargs)
        features = model.features_ if isinstance(model, PromptCenteredRidge) else model.named_steps["features"]
        char = features.named_steps["features"].transformer_list[0][1]
        if name == "masked_char_style_svm":
            char = char.named_steps["tfidf"]
        vocabulary, idf = char.vocabulary_.copy(), char.idf_.copy()
        probe = ["unseenqzxvocabulary—completely fresh!", "", "東京"]
        together = np.asarray(model.decision_function(probe))
        separate = np.concatenate([np.asarray(model.decision_function([row])) for row in probe], axis=0)
    np.testing.assert_allclose(together, separate, atol=1e-12)
    assert np.isfinite(together).all()
    assert vocabulary == char.vocabulary_
    np.testing.assert_array_equal(idf, char.idf_)


def test_boosting_factory_has_independent_estimators_and_bounded_cpu():
    pytest.importorskip("lightgbm")
    for name in ["lightgbm7", "lightgbm15", "dart7"]:
        first = make_v3_component("boosting:" + name)
        second = clone(first)
        assert first.named_steps["classifier"] is not second.named_steps["classifier"]
        params = first.named_steps["classifier"].get_params()
        assert params["n_jobs"] == 2
        assert params["deterministic"] is True
        assert params["subsample_freq"] == 0
        assert params["n_estimators"] > 0
    with pytest.raises(ValueError, match="Unknown boosting recipe"):
        make_v3_component("boosting:missing")


@pytest.mark.parametrize("name", ["style_rbf_c4", "additive_char_style_rbf_c4"])
def test_kernel_factory_keeps_training_state_and_prediction_batch_invariant(name):
    passages = [
        "The red kite is here. Yet it is calm.", "The blue kite is here; indeed it is calm.",
        "The red boat is here. Yet it is calm.", "The blue boat is here; indeed it is calm.",
        "The red cloud is here. Yet it is calm.", "The blue cloud is here; indeed it is calm.",
    ]
    labels = ["a", "b"] * 3
    model = clone(make_v3_component("kernel:" + name))
    model.set_params(batch_size=2, bandwidth_sample_size=4)
    with threadpool_limits(limits=1):
        model.fit(passages, labels)
        means, scales, gamma = model.scaler_.mean_.copy(), model.scaler_.scale_.copy(), model.gamma_
        probe = ["A wholly new qzxvocabulary passage—carefully.", "", "東京"]
        combined = model.decision_function(probe)
        separate = np.vstack([model.decision_function([row]) for row in probe])
        reordered = model.decision_function([probe[2], probe[0], probe[2], probe[1]])
        train_kernel = model._kernel(passages)
    np.testing.assert_allclose(combined, separate, atol=1e-12)
    np.testing.assert_allclose(reordered, combined[[2, 0, 2, 1]], atol=1e-12)
    np.testing.assert_allclose(train_kernel, train_kernel.T, atol=1e-12)
    assert np.linalg.eigvalsh(train_kernel).min() > -1e-10
    np.testing.assert_allclose(np.diag(train_kernel), 1, atol=1e-12)
    np.testing.assert_array_equal(means, model.scaler_.mean_)
    np.testing.assert_array_equal(scales, model.scaler_.scale_)
    assert gamma == model.gamma_ and gamma > 0
    assert model.n_bandwidth_samples_ == 4
    assert model.decision_function([]).shape == (0, 2)
    np.testing.assert_array_equal(model.predict(probe), model.classes_[combined.argmax(axis=1)])
    if name == "additive_char_style_rbf_c4":
        assert "qzx" not in model.vectorizer_.vocabulary_


def test_kernel_bandwidth_sampling_is_training_row_order_independent():
    from name_that_llm.kernel_v3 import bandwidth_sample_indices, median_style_gamma

    passages = ["delta", "alpha", "gamma", "beta", "epsilon"]
    chosen = [passages[i] for i in bandwidth_sample_indices(passages, max_samples=3)]
    reordered = passages[::-1]
    assert chosen == [reordered[i] for i in bandwidth_sample_indices(reordered, max_samples=3)]
    styles = np.asarray([[0, 0], [0, 1], [2, 0], [2, 1]], dtype=float)
    pair_distances = [1, 4, 5, 5, 4, 1]
    assert median_style_gamma(styles, np.arange(4)) == 1 / np.median(pair_distances)


@pytest.mark.parametrize("name", ["joint_char_style_c4", "joint_char_style_c16"])
def test_linear_factory_multinomial_probabilities_and_training_only_transforms(name):
    from name_that_llm.stylometry import StyleFeatures

    passages, labels = [], []
    for topic in ["kite", "boat", "cloud"]:
        for label, ending in [
            ("a", "Yet it is calm."), ("b", "Indeed, it works!"), ("c", "Why ask—what happens next?"),
        ]:
            passages.append(f"The {topic} is here. {ending}")
            labels.append(label)
    model = clone(make_v3_component("linear:" + name))
    with threadpool_limits(limits=1):
        model.fit(passages, labels)
        features = dict(model.named_steps["features"].transformer_list)
        char = features["char"]
        scaler = features["style"].named_steps["scale"]
        vocabulary, idf, mean = char.vocabulary_.copy(), char.idf_.copy(), scaler.mean_.copy()
        np.testing.assert_allclose(mean, StyleFeatures().transform(passages).mean(axis=0), atol=1e-12)
        assert scaler.n_samples_seen_ == len(passages)
        probe = ["Unseen qzxvocabulary—repeated carefully.", "", "東京"]
        probabilities = model.predict_proba(probe)
        np.testing.assert_allclose(probabilities, softmax(model.decision_function(probe), axis=1), atol=1e-12)
        np.testing.assert_allclose(
            probabilities, np.vstack([model.predict_proba([row]) for row in probe]), atol=1e-12,
        )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
    assert model.named_steps["classifier"].n_iter_.shape == (1,)
    assert tuple(model.classes_) == ("a", "b", "c")
    assert vocabulary == char.vocabulary_ and "qzx" not in vocabulary
    np.testing.assert_array_equal(idf, char.idf_)
    np.testing.assert_array_equal(mean, scaler.mean_)


@pytest.fixture(params=[("boosting", "topic"), ("kernel", "linear")])
def synthetic_selection_workspace(tmp_path, monkeypatch, request):
    repository = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("select_v3_under_test", repository / "scripts/select_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The public checkout has no private experiment artifacts. Build the entire
    # selector contract here; candidate names need not refer to fitted models.
    char, hist, style = "sparse:char15_c4", "style:style_hist", "style:char_style_svm"
    first_family, second_family = request.param
    recipe_components = {
        "v2": [char, hist],
        "style_pair": [style, hist],
        "three_evidence": [char, hist, style],
        "three_probability": [char, hist, style],
        "with_first_branch": [char, hist, first_family],
        "with_second_branch": [char, hist, second_family],
        "four": [char, hist, first_family, second_family],
    }
    plan = {
        "branch_families": list(request.param),
        first_family + "_candidates": ["toy_first_a", "toy_first_b", "toy_first_c"],
        second_family + "_candidates": ["toy_second_a", "toy_second_b", "toy_second_c"],
        "development_min_gain": 0.0075,
        "blend_recipes": {
            name: {
                "components": components,
                "weights": [1] * len(components),
                "pooling": "probability" if name == "three_probability" else "evidence",
            }
            for name, components in recipe_components.items()
        },
    }
    train = pd.DataFrame({
        "id": [f"synthetic-{i}" for i in range(90)],
        "prompt_id": np.repeat([f"prompt-{i}" for i in range(5)], len(LABELS)),
        "model": list(LABELS) * 5,
        "text": ["Synthetic passage."] * 90,
    })
    fold_values = np.repeat(np.arange(5), len(LABELS))
    identity = train[["id", "prompt_id", "model"]].copy()
    identity["fold"] = fold_values
    baseline = tmp_path / "artifacts/baseline-v1"
    baseline.mkdir(parents=True)
    identity.to_csv(baseline / "folds.csv", index=False)
    plan["folds_sha256"] = hashlib.sha256((baseline / "folds.csv").read_bytes()).hexdigest()
    root = tmp_path / "artifacts/improvement-v3"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan))
    old = tmp_path / "artifacts/improvement-v2"
    (old / "sparse").mkdir(parents=True)
    rng = np.random.default_rng(18)
    raw = rng.normal(size=(len(train), len(LABELS)))
    np.savez(old / "sparse/char15_c4_oof.npz", ids=train.id.to_numpy(dtype=str), labels=LABELS, scores=raw)
    metadata = {}
    for family, candidates in [
        ("style", ["style_hist", "char_style_svm"]),
        (first_family, plan[first_family + "_candidates"]),
        (second_family, plan[second_family + "_candidates"]),
    ]:
        for candidate in candidates:
            destination = (old if family == "style" else root) / family / candidate
            destination.mkdir(parents=True)
            identity.to_csv(destination / "oof_predictions.csv", index=False)
            schema = {"labels": list(LABELS), "scores_kind": "decision_function"}
            (destination / "score_schema.json").write_text(json.dumps(schema))
            np.save(destination / "oof_scores.npy", raw)
            if family == "style":
                metadata[candidate] = schema
    (old / "style/metrics.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(module, "load_data", lambda path: (train, pd.DataFrame()))
    monkeypatch.chdir(tmp_path)
    return module, root


def test_selection_accepts_aligned_scores_and_records_fixed_plan(synthetic_selection_workspace):
    module, root = synthetic_selection_workspace
    module.main([])
    selected = json.loads((root / "selection/selected.json").read_text())
    assert selected["selected_name"] == "v2"  # Declared recipe order breaks exact ties.
    assert selected["delta_from_v2"] == 0
    assert not selected["passes_development_gate"]
    assert len(selected["all_results"]) == 13
    assert len(selected["score_source_sha256"]) == 9


@pytest.mark.parametrize("failure", ["classes", "folds", "ids", "labels"])
def test_selection_rejects_misaligned_oof_sources(synthetic_selection_workspace, failure):
    module, root = synthetic_selection_workspace
    plan = json.loads((root / "plan.json").read_text())
    family = plan["branch_families"][0]
    candidate = plan[family + "_candidates"][0]
    folder = root / family / candidate
    if failure == "classes":
        schema_path = folder / "score_schema.json"
        schema = json.loads(schema_path.read_text())
        schema["labels"] = list(LABELS[::-1])
        schema_path.write_text(json.dumps(schema))
    else:
        path = folder / "oof_predictions.csv"
        frame = pd.read_csv(path)
        column = {"folds": "fold", "ids": "id", "labels": "model"}[failure]
        frame.loc[0, column] = frame.loc[1, column] if column != "fold" else 4
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Misaligned"):
        module.main([])


def test_selection_rejects_changed_frozen_fold_file(synthetic_selection_workspace):
    module, root = synthetic_selection_workspace
    folds = root.parent / "baseline-v1/folds.csv"
    # Even a semantically harmless edit invalidates the frozen byte-level input.
    folds.write_bytes(folds.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen plan"):
        module.main([])


@pytest.fixture
def synthetic_notebook_checkpoint(tmp_path, monkeypatch):
    import sys

    repository = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("build_v3_under_test", repository / "scripts/build_notebook_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "src/name_that_llm/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text('"""Synthetic package source; no private inputs."""\n')
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps({
        "passes_development_gate": True, "balanced_accuracy": 0.4,
        "recipe": {"components": ["linear:toy"], "weights": [1], "pooling": "evidence"},
    }))
    robustness = tmp_path / "robustness"
    robustness.mkdir()
    (robustness / "metrics.json").write_text(json.dumps({"passes": True, "v3": 0.4, "v2": 0.35}))
    (robustness / "plan.json").write_text(json.dumps({
        "selected_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "source_sha256": {
            str(source.relative_to(tmp_path)): hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }))
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["build_notebook_v3.py", "--selected", str(selected), "--robustness", str(robustness)])
    return module, tmp_path, source


def test_notebook_builder_creates_compilable_clear_output_notebook_from_frozen_sources(synthetic_notebook_checkpoint):
    import nbformat

    module, root, source = synthetic_notebook_checkpoint
    module.main()
    notebook = nbformat.read(root / "notebooks/improvement-v3/name-that-llm-validated-ensemble-v3.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<generated v3 notebook>", "exec")
            assert cell.outputs == [] and cell.execution_count is None
    assert notebook.metadata.experiment.embedded_source_sha256["__init__.py"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_notebook_builder_rejects_source_mutation_after_validation(synthetic_notebook_checkpoint):
    module, root, source = synthetic_notebook_checkpoint
    source.write_text(source.read_text() + "# Changed after frozen validation.\n")
    with pytest.raises(RuntimeError, match="source changed"):
        module.main()
    assert not (root / "notebooks/improvement-v3").exists()
