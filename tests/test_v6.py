"""Focused synthetic checks for the separately registered v6 frozen probes."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from name_that_llm import classifier_v6 as classifier
from name_that_llm.baseline import LABELS


@pytest.fixture
def grouped_vectors():
    labels = np.tile(np.asarray(LABELS), 10)
    groups = np.repeat(np.asarray([f"group-{index}" for index in range(10)]), 18)
    folds = np.repeat(np.arange(10) // 2, 18)
    values = np.zeros((len(labels), 512), dtype=np.float32)
    values[np.arange(len(labels)), np.tile(np.arange(18), 10)] = 1.
    values[:, 30] = np.arange(len(labels)) / 200
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    order = np.random.default_rng(6).permutation(len(labels))
    return values[order], labels[order], groups[order], folds[order]


def normalized(values, probabilities=False):
    values = np.asarray(values, dtype=float)
    if probabilities:
        values = np.log(np.clip(values, 1e-12, 1.))
    values = values - values.mean(axis=1, keepdims=True)
    return values / np.maximum(values.std(axis=1, keepdims=True), 1e-8)


def test_exact_two_models_and_four_recipes_match_frozen_plan():
    plan = json.loads((Path(__file__).resolve().parents[1] / "configs/name-that-llm/v6.json").read_text())
    assert classifier.MODEL_DEFINITIONS == plan["models"]
    assert classifier.RECIPES == plan["candidates"]
    assert list(classifier.MODEL_DEFINITIONS) == ["lr64", "rbf10"]
    assert list(classifier.RECIPES) == ["lr64", "lr64_v4_half", "rbf10", "rbf10_v4_half"]
    assert classifier.MODEL_DEFINITIONS["lr64"]["parameters"] == {
        "C": 64., "class_weight": "balanced", "solver": "lbfgs", "max_iter": 3000, "tol": .0001,
    }
    assert classifier.MODEL_DEFINITIONS["rbf10"]["parameters"] == {
        "C": 10., "kernel": "rbf", "gamma": "scale", "class_weight": "balanced",
        "probability": False, "decision_function_shape": "ovr", "cache_size": 512,
        "tol": .001, "max_iter": -1,
    }


@pytest.mark.parametrize("name", ["lr64", "rbf10"])
def test_fixed_fits_and_training_only_gamma(grouped_vectors, name):
    values, labels, _, folds = grouped_vectors
    train = folds != 0
    model = classifier.fit_classifier(values[train], labels[train], name)
    assert tuple(model.classes_) == LABELS
    for parameter, expected in classifier.MODEL_DEFINITIONS[name]["parameters"].items():
        assert model.get_params()[parameter] == expected
    assert not hasattr(model, "named_steps"), "No fitted scaler or other transform is permitted"
    if name == "lr64":
        assert model.get_params()["warm_start"] is False
        assert model.get_params()["penalty"] == "l2"
    else:
        expected_gamma = 1. / (512 * values[train].astype(np.float64).var())
        np.testing.assert_allclose(model._gamma, expected_gamma, rtol=1e-12, atol=0)
        assert not np.isclose(expected_gamma, 1. / (512 * values.astype(np.float64).var()), rtol=1e-8)
        assert model.probA_.size == model.probB_.size == 0
    evidence = classifier.model_evidence(model, values[~train], name)
    assert evidence.shape == (np.sum(~train), 18) and np.isfinite(evidence).all()
    np.testing.assert_allclose(evidence.mean(axis=1), 0., atol=1e-12)
    np.testing.assert_allclose(evidence.std(axis=1), 1., atol=1e-12)


@pytest.mark.parametrize("name", ["lr64", "rbf10"])
def test_evidence_uses_declared_score_type_without_svc_calibration(grouped_vectors, name):
    values = grouped_vectors[0][:3]
    raw = np.random.default_rng(5).normal(size=(3, 18))
    probabilities = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)
    calls = []

    def probability(query):
        assert name == "lr64", "SVC must never fit or request probabilities"
        calls.append("probability")
        return probabilities

    def decisions(query):
        assert name == "rbf10"
        calls.append("decision")
        return raw

    model = SimpleNamespace(classes_=np.asarray(LABELS), predict_proba=probability,
                            decision_function=decisions)
    actual = classifier.model_evidence(model, values, name)
    expected = normalized(probabilities, True) if name == "lr64" else normalized(raw)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert calls == ["probability" if name == "lr64" else "decision"]


def test_blend_uses_renormalized_completed_v4_evidence():
    rng = np.random.default_rng(6)
    scores = normalized(rng.normal(size=(6, 18)))
    incumbent = rng.normal(size=(6, 18)) * np.arange(1, 7)[:, None] + 37
    for name, recipe in classifier.RECIPES.items():
        actual = classifier.combine_evidence(scores, incumbent, recipe)
        expected = recipe["embedding_weight"] * scores + recipe["v4_weight"] * normalized(incumbent)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    with pytest.raises(ValueError):
        classifier.combine_evidence(scores, incumbent, {"model": "rbf10", "embedding_weight": .9, "v4_weight": .1})


@pytest.mark.parametrize("name", ["lr64", "rbf10"])
def test_oof_model_instances_and_diagnostics_obey_fold_boundaries(grouped_vectors, monkeypatch, name):
    values, labels, groups, folds = grouped_vectors
    row_index = {row.tobytes(): index for index, row in enumerate(values)}
    fits, calls = [], []

    class Model:
        classes_ = np.asarray(LABELS)
        n_iter_ = np.asarray([3])
        n_support_ = np.ones(18, dtype=int)
        support_ = np.arange(18)
        fit_status_ = 0
        _gamma = 1.25

        def __init__(self, rows):
            self.rows = set(rows)

    def fit(train, target, model_name):
        assert model_name == name
        rows = [row_index[row.tobytes()] for row in train]
        np.testing.assert_array_equal(target, labels[rows])
        held_out = set(range(5)) - set(folds[rows])
        assert len(held_out) == 1
        assert set(rows) == set(np.flatnonzero(folds != held_out.pop()))
        model = Model(rows)
        fits.append(model)
        return model

    def evidence(model, query, model_name):
        assert model_name == name
        rows = [row_index[row.tobytes()] for row in query]
        if set(rows) == model.rows:
            kind = "training_diagnostic"
        else:
            assert set(rows) == set(range(len(values))) - model.rows
            assert {groups[index] for index in model.rows}.isdisjoint(groups[rows])
            kind = "held_out"
        calls.append((id(model), kind))
        scores = np.full((len(rows), 18), -1.)
        scores[np.arange(len(rows)), [LABELS.index(labels[index]) for index in rows]] = 1.
        return normalized(scores)

    monkeypatch.setattr(classifier, "fit_classifier", fit)
    monkeypatch.setattr(classifier, "model_evidence", evidence)
    scores, audit = classifier.evaluate_oof(values, labels, folds, groups, name)
    assert len(fits) == len({id(model) for model in fits}) == 5
    np.testing.assert_array_equal(np.asarray(LABELS)[scores.argmax(axis=1)], labels)
    for fold, (model, record) in enumerate(zip(fits, audit, strict=True)):
        assert sorted(kind for identity, kind in calls if identity == id(model)) == ["held_out", "training_diagnostic"]
        assert record["held_out_fold"] == fold
        assert set(record["training_groups"]).isdisjoint(record["held_out_groups"])
        assert record["training_balanced_accuracy"] == record["held_out_balanced_accuracy"] == 1.
        assert record["held_out_scores_sha256"] == classifier.array_hash(scores[folds == fold])
        if name == "rbf10":
            assert record["probability_calibration"] is False
    leaky_folds = folds.copy()
    leaky_folds[0] = (leaky_folds[0] + 1) % 5
    with pytest.raises(ValueError, match="crosses folds"):
        classifier.evaluate_oof(values, labels, leaky_folds, groups, name)


def test_selection_ties_follow_declaration_order_not_input_dictionary_order():
    results = {name: {"balanced_accuracy": .5} for name in reversed(classifier.RECIPES)}
    assert classifier.choose_recipe(results) == "lr64"
    results["rbf10"]["balanced_accuracy"] = .6
    assert classifier.choose_recipe(results) == "rbf10"
    for malformed in ({}, {"unknown": {"balanced_accuracy": .5}},
                      {"lr64": {"balanced_accuracy": np.nan}}):
        with pytest.raises(ValueError):
            classifier.choose_recipe(malformed)


@pytest.fixture
def runner_fixture(tmp_path, monkeypatch, grouped_vectors):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("v6_runner_under_test", scripts / "run_embeddings_v6.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.chdir(tmp_path)
    values, labels, groups, folds = grouped_vectors
    for folder in ("cache", "model", "data/name-that-llm", "artifacts/improvement-v6", "configs/name-that-llm"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for name in ("cache/train-embeddings.npz", "cache/train-embeddings.json", "model/manifest.json",
                 "configs/name-that-llm/v5.json", "configs/name-that-llm/v6.json", "predecessor.json",
                 "artifacts/improvement-v6/frozen-plan.json"):
        (tmp_path / name).write_text("synthetic\n")
    train = pd.DataFrame({"id": [f"r-{index}" for index in range(len(values))],
                          "text": [f"synthetic {index}" for index in range(len(values))],
                          "model": labels, "prompt_id": groups})
    train.to_csv(tmp_path / "data/name-that-llm/train.csv", index=False)
    plan = {"embedding_plan": "configs/name-that-llm/v5.json", "predecessor_result": {"path": "predecessor.json"},
            "train_embedding_sha256": runner.file_sha256(tmp_path / "cache/train-embeddings.npz"),
            "development": {"min_gain": .0075}, "robustness": {"min_gain": .005, "min_fold_wins": 4,
                "seed": 31415, "bootstrap_draws": 100}, "limitations": ["Synthetic reused data."]}
    monkeypatch.setattr(runner, "validate_plan", lambda path: plan)
    correct = np.full((len(labels), 18), -1.)
    correct[np.arange(len(labels)), [LABELS.index(label) for label in labels]] = 1.
    score = normalized(correct)
    wrong = np.roll(correct, 1, axis=1)
    baseline, predictions = runner.evaluate(wrong, labels, folds)
    monkeypatch.setattr(runner, "load_incumbent", lambda plan, mode: (
        train, folds, groups, wrong, baseline, predictions, []))
    monkeypatch.setattr(runner, "load_embedding_cache", lambda *args, **kwargs: (
        values, {"provenance": {"runtime": {"synthetic": True}, "model_files": []}}))
    calls = []

    def evaluate(input_values, input_y, input_folds, input_groups, name):
        np.testing.assert_array_equal(input_values, values)
        np.testing.assert_array_equal(input_y, labels)
        np.testing.assert_array_equal(input_folds, folds)
        np.testing.assert_array_equal(input_groups, groups)
        calls.append(name)
        return score.copy(), [{"synthetic": True}]

    monkeypatch.setattr(runner, "evaluate_oof", evaluate)

    def execute(mode):
        runner.run(mode, tmp_path / "configs/name-that-llm/v6.json", tmp_path / "cache", tmp_path / "model",
                   tmp_path / mode, tmp_path / "development/selected.json")
        return json.loads((tmp_path / mode / ("selected.json" if mode == "development" else "metrics.json")).read_text())

    return SimpleNamespace(runner=runner, root=tmp_path, execute=execute, calls=calls)


def test_runner_fits_two_models_once_then_only_the_selected_model(runner_fixture, capsys):
    f = runner_fixture
    result = f.execute("development")
    assert set(result["all_results"]) == set(classifier.RECIPES)
    assert result["selected_name"] == "lr64" and result["passes_development_gate"]
    assert f.calls == ["lr64", "rbf10"]
    robust = f.execute("robustness")
    assert list(robust["all_results"]) == ["lr64"] and robust["passes"]
    assert f.calls == ["lr64", "rbf10", "lr64"]
    assert not (f.root / "robustness/rbf10_oof_evidence.npy").exists()


@pytest.mark.parametrize("filename", ["lr64_oof_evidence.npy", "rbf10_oof_evidence.npy", "fit_audit.json", "selected.json"])
def test_robustness_rejects_changed_development_evidence_before_fitting(runner_fixture, filename, capsys):
    f = runner_fixture
    f.execute("development")
    with (f.root / "development" / filename).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError):
        f.execute("robustness")
    assert f.calls == ["lr64", "rbf10"]
    assert not (f.root / "robustness").exists()


def test_exact_predecessor_embedding_cache_is_required_before_fitting(runner_fixture):
    f = runner_fixture
    (f.root / "cache/train-embeddings.npz").write_text("different vectors\n")
    with pytest.raises(ValueError, match="exactly the recorded v5"):
        f.execute("development")
    assert f.calls == []


def test_failed_development_gate_blocks_robustness_even_if_flag_is_edited(runner_fixture, capsys):
    f = runner_fixture
    f.execute("development")
    original = f.runner.verify_selection

    def falsely_passing_selection(path):
        result = original(path)
        result["passes_development_gate"] = True
        result["delta_from_v4"] = .001
        return result

    f.runner.verify_selection = falsely_passing_selection
    with pytest.raises(ValueError, match="Development gate failed"):
        f.execute("robustness")
    assert f.calls == ["lr64", "rbf10"]
