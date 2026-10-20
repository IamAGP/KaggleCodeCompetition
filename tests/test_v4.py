"""Synthetic contracts for nested stacking; no private data or artifacts required."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from name_that_llm.baseline import LABELS
from name_that_llm import stacking_v4 as stacking


@pytest.fixture
def grouped_rows():
    """Interleaved rows prevent tests from accidentally relying on fold order."""
    labels = np.tile(np.asarray(LABELS, dtype=object), 10)
    groups = np.repeat(np.asarray([f"prompt-{i}" for i in range(10)]), len(LABELS))
    folds = np.repeat(np.arange(10) // 2, len(LABELS))
    order = np.random.default_rng(501).permutation(len(labels))
    return (
        np.asarray([str(i) for i in range(len(labels))], dtype=object)[order],
        labels[order],
        groups[order],
        folds[order],
    )


def raw_scores(texts, component=0):
    ids = np.asarray(texts, dtype=int)
    classes = np.arange(len(LABELS))
    return np.cos(ids[:, None] * .17 + classes[None, :] * .31 + component)


def row_normalize(scores):
    centered = scores - scores.mean(axis=1, keepdims=True)
    return centered / centered.std(axis=1, keepdims=True)


def synthetic_bundle(grouped_rows):
    texts, labels, groups, folds = grouped_rows
    evidence = np.stack([
        row_normalize(raw_scores(texts, component))
        for component in range(len(stacking.COMPONENTS))
    ], axis=1)
    return stacking.audited_oof_bundle(
        evidence, folds, texts, labels, groups,
        provenance={"kind": "synthetic_fixture", "no_private_artifacts": True},
    )


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def nested_fixture(grouped_rows, monkeypatch):
    runner = load_script("run_nested_v4")
    texts, labels, groups, folds = grouped_rows
    train = pd.DataFrame({
        "id": [f"row-{text}" for text in texts],
        "text": texts, "model": labels, "prompt_id": groups,
    })
    assignments = train[["id", "prompt_id", "model"]].copy()
    assignments["fold"] = folds
    definition = {
        "train_sha256": "synthetic-train",
        "folds_sha256": "synthetic-folds",
        "source_hashes": stacking.source_hashes(),
        "recipe_sha256": stacking.json_hash(stacking.recipe_definition()),
        "versions": {"test": "synthetic"},
        "threads": 1,
        "script_sha256": "synthetic-runner",
        "plan_sha256": "synthetic-plan",
    }
    observed = []
    row_fold = dict(zip(texts, folds))
    row_label = dict(zip(texts, labels))

    class RecordingBase:
        classes_ = np.asarray(LABELS)
        named_steps = {"classifier": SimpleNamespace(n_iter_=np.asarray([1]))}

        def __init__(self, name, fitting):
            self.component = stacking.COMPONENTS.index(name)
            self.fitting = set(fitting)
            self.excluded = set(range(5)) - {row_fold[row] for row in fitting}

        def decision_function(self, query):
            assert self.fitting.isdisjoint(query)
            assert {row_fold[row] for row in query} <= self.excluded
            values = raw_scores(query, self.component)
            # Exclusion-specific scores expose a swap of the two halves or pairs.
            values[:, 0] += sum(2 ** fold for fold in self.excluded) / 7
            return values

    def fit_base(name, X, y):
        assert list(y) == [row_label[row] for row in X]
        fitted_folds = {row_fold[row] for row in X}
        assert len(fitted_folds) == 3
        expected = {row for row in texts if row_fold[row] in fitted_folds}
        assert set(X) == expected
        observed.append((name, fitted_folds, set(X)))
        return RecordingBase(name, X)

    monkeypatch.setattr(runner, "fit_component", fit_base)
    # Freeze test provenance while collaborators may edit source concurrently.
    monkeypatch.setattr(runner, "source_hashes", lambda: definition["source_hashes"])
    return runner, train, assignments, definition, observed


def test_nested_pair_fits_exclude_both_folds_and_meta_receives_only_correct_half(
    nested_fixture, grouped_rows, monkeypatch, tmp_path,
):
    runner, train, assignments, definition, observed = nested_fixture
    folds = assignments["fold"].to_numpy()
    pairs = runner.run_exclusion_jobs(
        list(combinations(range(5), 2)), tmp_path, train, assignments,
        definition, workers=1, category="pairs",
    )
    assert len(observed) == 40
    for component in stacking.COMPONENTS:
        exclusions = {
            tuple(sorted(set(range(5)) - fitted_folds))
            for name, fitted_folds, _ in observed if name == component
        }
        assert exclusions == set(combinations(range(5), 2))

    outer = synthetic_bundle(grouped_rows)
    real_fit_meta = runner.fit_meta
    meta_calls = []

    def inspect_meta(evidence, y):
        outer_fold = len(meta_calls)
        indexes = np.flatnonzero(folds != outer_fold)
        expected = []
        for row in indexes:
            pair = tuple(sorted((outer_fold, int(folds[row]))))
            pair_indexes, pair_values, _ = pairs[pair]
            expected.append(pair_values[np.flatnonzero(pair_indexes == row)[0]])
        np.testing.assert_array_equal(evidence, np.asarray(expected))
        np.testing.assert_array_equal(y, train.iloc[indexes]["model"])
        assert not np.array_equal(evidence, outer.evidence[indexes])
        model = real_fit_meta(evidence, y)
        valid_features = outer.evidence[folds == outer_fold].reshape(-1, 72)
        real_predict = model.predict_proba

        def predict_on_outer_only(features):
            assert all(np.any(np.all(valid_features == row, axis=1)) for row in features)
            return real_predict(features)

        model.predict_proba = predict_on_outer_only
        meta_calls.append(outer_fold)
        return model

    monkeypatch.setattr(runner, "fit_meta", inspect_meta)
    metrics = runner.evaluate(train, assignments, outer, pairs, tmp_path, definition)
    assert meta_calls == list(range(5))
    assert metrics["number_pair_fits"] == 10
    saved = pd.read_csv(tmp_path / "oof_predictions.csv")
    assert saved["id"].tolist() == train["id"].tolist()
    assert saved["fold"].tolist() == folds.tolist()
    # Reusing all ten jobs must verify caches and fit zero additional models.
    runner.run_exclusion_jobs(
        list(combinations(range(5), 2)), tmp_path, train, assignments,
        definition, workers=1, category="pairs",
    )
    assert len(observed) == 40


@pytest.mark.parametrize("changed", ["source_hashes", "script_sha256", "plan_sha256", "folds_sha256"])
def test_pair_cache_rejects_changed_producer_before_any_reuse(nested_fixture, tmp_path, changed):
    runner, train, assignments, definition, observed = nested_fixture
    runner.fit_exclusion_job(tmp_path, (0, 3), train, assignments, definition)
    different = deepcopy(definition)
    different[changed] = {"different": "source"} if changed == "source_hashes" else "different"
    with pytest.raises(ValueError, match="cache key"):
        runner.fit_exclusion_job(tmp_path, (0, 3), train, assignments, different)
    assert len(observed) == 4


@pytest.mark.parametrize("changed", ["row_indexes", "ids", "folds", "labels", "components", "evidence"])
def test_pair_cache_rejects_corrupt_row_class_fold_or_score_content(nested_fixture, tmp_path, changed):
    runner, train, assignments, definition, _ = nested_fixture
    runner.fit_exclusion_job(tmp_path, (1, 4), train, assignments, definition)
    path = tmp_path / "scores.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if changed == "evidence":
        arrays[changed][0, 0, 0] += .5
    else:
        arrays[changed] = arrays[changed][::-1]
    np.savez_compressed(path, **arrays)
    # Re-sign only the file digest: independent row/class/fold/evidence checks
    # must still catch the inconsistent content.
    metadata = runner.load_json(tmp_path / "metadata.json")
    metadata["scores_file_sha256"] = runner.file_sha256(path)
    runner.write_json(tmp_path / "metadata.json", metadata)
    with pytest.raises((ValueError, AssertionError)):
        runner.fit_exclusion_job(tmp_path, (1, 4), train, assignments, definition)


def test_runtime_oof_never_fits_a_held_out_prompt_and_routes_rows(monkeypatch, grouped_rows):
    texts, labels, groups, _ = grouped_rows
    row_group = dict(zip(texts, groups))
    seen = []

    class RecordingBase:
        classes_ = np.asarray(LABELS)

        def __init__(self, name, fitting):
            self.component = stacking.COMPONENTS.index(name)
            self.fitting = set(fitting)

        def decision_function(self, query):
            query = list(query)
            fitting_groups = {row_group[row] for row in self.fitting}
            query_groups = {row_group[row] for row in query}
            assert fitting_groups.isdisjoint(query_groups)
            assert self.fitting | set(query) == set(texts)
            seen.append((self.component, self.fitting, query))
            return raw_scores(query, self.component)

    monkeypatch.setattr(stacking, "fit_component", lambda name, X, y: RecordingBase(name, X))
    bundle = stacking.generate_fit_oof(texts, labels, groups, threads=1)
    assert len(seen) == 5 * len(stacking.COMPONENTS)
    for component in range(len(stacking.COMPONENTS)):
        queries = [row for c, _, query in seen if c == component for row in query]
        assert sorted(queries) == sorted(texts)
        np.testing.assert_allclose(
            bundle.evidence[:, component],
            row_normalize(raw_scores(texts, component)),
            atol=1e-12,
        )
    bundle.validate(texts, labels, groups)


@pytest.mark.parametrize("changed", ["text", "target", "group", "row_order"])
def test_audited_oof_cannot_be_reused_for_different_training_rows(grouped_rows, changed):
    bundle = synthetic_bundle(grouped_rows)
    texts, labels, groups, _ = [x.copy() for x in grouped_rows]
    if changed == "text":
        texts[0] = "different passage"
    elif changed == "target":
        labels[0], labels[1] = labels[1], labels[0]
        if labels[0] == labels[1]:
            labels[0] = LABELS[(LABELS.index(labels[0]) + 1) % len(LABELS)]
    elif changed == "group":
        groups[0] = "new-prompt"
    else:
        texts = texts[::-1]
    with pytest.raises(ValueError, match="training_fingerprint"):
        bundle.validate(texts, labels, groups)


@pytest.mark.parametrize("changed", [
    "evidence", "folds", "source_hashes", "recipe_sha256", "components",
    "labels", "group_oof_verified", "provenance",
])
def test_audited_oof_rejects_tampered_cache(grouped_rows, changed):
    bundle = synthetic_bundle(grouped_rows)
    texts, labels, groups, _ = grouped_rows
    if changed == "evidence":
        evidence = bundle.evidence.copy()
        evidence[0, 0, 0] += .125
        bundle = replace(bundle, evidence=evidence)
    elif changed == "folds":
        bundle = replace(bundle, folds=(bundle.folds + 1) % 5)
    else:
        audit = deepcopy(bundle.audit)
        audit[changed] = {} if changed in ("source_hashes", "provenance") else None
        bundle = replace(bundle, audit=audit)
    with pytest.raises(ValueError, match="audit mismatch|provenance"):
        bundle.validate(texts, labels, groups)


def test_oof_bundle_is_sealed_and_rejects_raw_scores(grouped_rows):
    texts, labels, groups, _ = grouped_rows
    bundle = synthetic_bundle(grouped_rows)
    assert not bundle.evidence.flags.writeable
    assert not bundle.folds.flags.writeable
    with pytest.raises(TypeError, match="AuditedOOF"):
        stacking.NestedStacker(threads=1).fit(texts, labels, groups, fit_oof=bundle.evidence)


def test_fold_validation_rejects_group_leakage_even_if_class_coverage_is_intact(grouped_rows):
    _, labels, groups, folds = grouped_rows
    assignment = folds.copy()
    assignment[0] = (assignment[0] + 1) % 5
    with pytest.raises(ValueError, match="prompt group crosses"):
        stacking.validate_fold_assignment(assignment, labels, groups)


@pytest.mark.parametrize("failure", ["missing_fold", "missing_class", "nonfinite", "shape"])
def test_malformed_fold_or_evidence_arrays_fail_closed(grouped_rows, failure):
    _, labels, groups, folds = grouped_rows
    if failure == "missing_fold":
        folds = np.where(folds == 4, 3, folds)
        with pytest.raises(ValueError, match="Expected folds"):
            stacking.validate_fold_assignment(folds, labels, groups)
    elif failure == "missing_class":
        labels = labels.copy()
        labels[(folds == 0) & (labels == LABELS[0])] = LABELS[1]
        with pytest.raises(ValueError, match="all classes"):
            stacking.validate_fold_assignment(folds, labels, groups)
    else:
        values = np.zeros((len(labels), 4, 18))
        if failure == "nonfinite":
            values[0, 0, 0] = np.nan
        else:
            values = values[:, :, :-1]
        with pytest.raises(ValueError):
            stacking.evidence_features(values)


def test_real_meta_scaler_learns_only_supplied_training_rows(grouped_rows):
    bundle = synthetic_bundle(grouped_rows)
    _, labels, _, folds = grouped_rows
    fitting = folds != 0
    with threadpool_limits(limits=1):
        model = stacking.fit_meta(bundle.evidence[fitting], labels[fitting])
    np.testing.assert_allclose(
        model["scale"].mean_, bundle.evidence[fitting].reshape(sum(fitting), 72).mean(axis=0),
        atol=1e-13,
    )
    assert model["scale"].n_samples_seen_ == sum(fitting)
    assert model["classifier"].C == .05
    assert model["classifier"].solver == "lbfgs"
    assert model["classifier"].max_iter == 2000
    assert tuple(model.classes_) == LABELS


def test_stack_prediction_is_text_only_and_independent_of_batch(monkeypatch, grouped_rows):
    texts, labels, groups, _ = grouped_rows
    bundle = synthetic_bundle(grouped_rows)
    fitted = []

    class MarginBase:
        classes_ = np.asarray(LABELS)

        def __init__(self, component):
            self.component = component

        def decision_function(self, query):
            return raw_scores(query, self.component)

    class ProbabilityBase(MarginBase):
        def predict_proba(self, query):
            return softmax(raw_scores(query, self.component), axis=1)

    def fit_base(name, X, y):
        assert list(X) == list(texts)
        assert list(y) == list(labels)
        fitted.append(name)
        component = stacking.COMPONENTS.index(name)
        return (ProbabilityBase if component % 2 else MarginBase)(component)

    monkeypatch.setattr(stacking, "fit_component", fit_base)
    model = stacking.NestedStacker(threads=1).fit(texts, labels, groups, fit_oof=bundle)
    assert fitted == list(stacking.COMPONENTS)
    assert clone(model).get_params() == {"threads": 1}
    query = ["4", "19", "30", "100", "4"]
    scores = model.predict_proba(query)
    separate = np.vstack([model.predict_proba([row]) for row in query])
    np.testing.assert_allclose(scores, separate, atol=1e-12)
    np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-12)
    order = [3, 0, 3, 2]
    np.testing.assert_allclose(model.predict_proba(np.asarray(query)[order]), scores[order], atol=1e-12)
    np.testing.assert_array_equal(model.predict(query), np.asarray(LABELS)[scores.argmax(axis=1)])


def test_component_class_order_is_enforced_before_evidence_is_used():
    class ReorderedBase:
        classes_ = np.asarray(LABELS[::-1])

        def decision_function(self, X):
            pytest.fail("A class mismatch must fail before calculating evidence.")

    with pytest.raises(ValueError, match="class order mismatch"):
        stacking.component_evidence(ReorderedBase(), ["one passage"])


@pytest.fixture
def selection_fixture(grouped_rows, monkeypatch, tmp_path):
    """Create the complete producer contract without reading private artifacts."""
    selector = load_script("select_v4")
    texts, labels, groups, folds = grouped_rows
    train = pd.DataFrame({
        "id": [f"row-{text}" for text in texts],
        "text": texts, "model": labels, "prompt_id": groups,
    })
    monkeypatch.setattr(selector, "load_data", lambda _: (train, pd.DataFrame()))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/name-that-llm").mkdir(parents=True)
    train.to_csv(tmp_path / "data/name-that-llm/train.csv", index=False)
    fold_path = tmp_path / "folds.csv"
    rows = train[["id", "prompt_id", "model"]].copy()
    rows["fold"] = folds
    rows.to_csv(fold_path, index=False)
    rng = np.random.default_rng(606)
    probabilities = softmax(rng.normal(size=(len(train), 18)), axis=1)
    evidence = rng.normal(size=(len(train), 4, 18))
    evidence = np.stack([row_normalize(evidence[:, c]) for c in range(4)], axis=1)
    baseline = evidence[:, [0, 1, 3]].mean(axis=1)
    baseline_result, _ = selector.evaluate(baseline, labels, folds)
    plan = {
        "folds_sha256": selector.file_sha256(fold_path),
        "baseline": {"development_balanced_accuracy": baseline_result["balanced_accuracy"]},
        # Reverse insertion order makes the deterministic tie rule observable.
        "candidates": {
            "stack_v3_half": {"stack_weight": .5, "v3_weight": .5},
            "stack": {"stack_weight": 1., "v3_weight": 0.},
        },
        "development_min_gain": .0075,
        "limitations": ["Synthetic test fixture."],
        "robustness": {
            "seed": 31415, "n_splits": 5, "min_pooled_gain": .005, "min_fold_wins": 4,
            "related_prompt_pairs": [
                ["p010", "p082"], ["p100", "p146"], ["p017", "p148"],
                ["p128", "p136"], ["p008", "p131"], ["p055", "p058"], ["p004", "p111"],
            ],
        },
    }
    plan_path = tmp_path / "plan.json"
    selector.save_json(plan_path, plan)
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "stack_oof_probabilities.npy", probabilities)
    np.save(run / "outer_oof_evidence.npy", evidence)
    rows["prediction"] = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    rows.to_csv(run / "oof_predictions.csv", index=False)
    definition = {
        "plan_sha256": selector.file_sha256(plan_path),
        "source_hashes": stacking.source_hashes(),
        "script_sha256": selector.file_sha256(Path(selector.__file__).with_name("run_nested_v4.py")),
        "folds_sha256": selector.file_sha256(fold_path),
        "train_sha256": selector.file_sha256(tmp_path / "data/name-that-llm/train.csv"),
        "labels": list(LABELS), "meta_parameters": stacking.META_PARAMETERS,
        "recipe": stacking.recipe_definition(),
        "outer_prediction_source": "audited_original_v2_v3_oof",
    }
    selector.save_json(run / "experiment_definition.json", definition)
    selector.save_json(run / "score_schema.json", {
        "labels": list(LABELS), "components": list(stacking.COMPONENTS),
        "source_hashes": definition["source_hashes"], "folds_sha256": definition["folds_sha256"],
    })
    selector.save_json(run / "metrics.json", {"synthetic": True})
    selector.save_json(run / "completion.json", {
        "all_checks_passed": True, "source_unchanged_during_run": True,
        "metrics": {"synthetic": True},
    })

    def seal_manifest():
        names = (
            "stack_oof_probabilities.npy", "outer_oof_evidence.npy", "oof_predictions.csv",
            "experiment_definition.json", "score_schema.json", "metrics.json", "completion.json",
        )
        selector.save_json(run / "artifacts.json", {"files": [
            {"path": name, "bytes": (run / name).stat().st_size, "sha256": selector.file_sha256(run / name)}
            for name in names
        ]})

    seal_manifest()
    return SimpleNamespace(
        selector=selector, train=train, folds=folds, fold_path=fold_path,
        run=run, plan=plan, plan_path=plan_path, probabilities=probabilities,
        evidence=evidence, definition=definition, seal_manifest=seal_manifest, root=tmp_path,
    )


def test_selection_uses_only_frozen_stack_and_half_blend(selection_fixture, capsys):
    f = selection_fixture
    f.selector.select(f.plan_path, f.run, f.fold_path, f.root / "selection")
    actual = f.selector.read_json(f.root / "selection/selected.json")
    stack = row_normalize(np.log(f.probabilities))
    baseline = f.evidence[:, [0, 1, 3], :].mean(axis=1)
    labels = f.train["model"].to_numpy()
    expected = {}
    for name, evidence in {"stack": stack, "stack_v3_half": .5 * (stack + baseline)}.items():
        expected[name], _ = f.selector.evaluate(evidence, labels, f.folds)
        assert actual["all_results"][name] == expected[name]
    chosen = max(expected, key=lambda name: (expected[name]["balanced_accuracy"], name == "stack"))
    assert actual["selected_name"] == chosen
    assert set(actual["all_results"]) == {"stack", "stack_v3_half"}
    assert actual["passes_development_gate"] == (actual["delta_from_v3"] >= .0075)
    assert actual["evaluator_sha256"] == f.definition["script_sha256"]
    capsys.readouterr()


def test_selection_ties_favor_stack_regardless_of_candidate_json_order(selection_fixture, capsys):
    f = selection_fixture
    stack = row_normalize(np.log(f.probabilities))
    evidence = np.repeat(stack[:, None, :], 4, axis=1)
    np.save(f.run / "outer_oof_evidence.npy", evidence)
    result, _ = f.selector.evaluate(stack, f.train["model"].to_numpy(), f.folds)
    f.plan["baseline"]["development_balanced_accuracy"] = result["balanced_accuracy"]
    f.selector.save_json(f.plan_path, f.plan)
    f.definition["plan_sha256"] = f.selector.file_sha256(f.plan_path)
    f.selector.save_json(f.run / "experiment_definition.json", f.definition)
    f.seal_manifest()
    f.selector.select(f.plan_path, f.run, f.fold_path, f.root / "selection")
    assert f.selector.read_json(f.root / "selection/selected.json")["selected_name"] == "stack"
    capsys.readouterr()


@pytest.mark.parametrize("changed", [
    "source_hashes", "script_sha256", "plan_sha256", "folds_sha256",
    "train_sha256", "labels", "meta_parameters",
])
def test_selector_rejects_stale_producer_even_with_consistent_artifact_manifest(selection_fixture, changed):
    f = selection_fixture
    f.definition[changed] = {"stale": True} if changed in ("source_hashes", "meta_parameters") else "stale"
    f.selector.save_json(f.run / "experiment_definition.json", f.definition)
    f.seal_manifest()
    with pytest.raises(ValueError, match=f"provenance mismatch: {changed}"):
        f.selector.select(f.plan_path, f.run, f.fold_path, f.root / "selection")
    assert not (f.root / "selection").exists()


@pytest.mark.parametrize("changed", ["output", "row_order", "class_prediction", "incomplete"])
def test_selector_rejects_output_corruption_or_incomplete_run(selection_fixture, changed):
    f = selection_fixture
    if changed == "output":
        np.save(f.run / "stack_oof_probabilities.npy", f.probabilities[::-1])
    elif changed in ("row_order", "class_prediction"):
        rows = pd.read_csv(f.run / "oof_predictions.csv")
        if changed == "row_order":
            rows = rows.iloc[::-1]
        else:
            rows["prediction"] = LABELS[0]
        rows.to_csv(f.run / "oof_predictions.csv", index=False)
        f.seal_manifest()
    else:
        completion = f.selector.read_json(f.run / "completion.json")
        completion["all_checks_passed"] = False
        f.selector.save_json(f.run / "completion.json", completion)
        f.seal_manifest()
    with pytest.raises(ValueError):
        f.selector.select(f.plan_path, f.run, f.fold_path, f.root / "selection")


def test_robustness_groups_keep_all_seven_related_pairs_together_and_freeze_selection(
    selection_fixture, monkeypatch,
):
    f = selection_fixture
    train = pd.DataFrame({
        "id": [f"synthetic-{i}" for i in range(150 * 18)],
        "prompt_id": np.repeat([f"p{i:03}" for i in range(150)], 18),
        "model": np.tile(LABELS, 150),
    })
    monkeypatch.setattr(f.selector, "load_data", lambda _: (train, pd.DataFrame()))
    selected_path = f.root / "selected.json"
    f.selector.save_json(selected_path, {
        "plan_sha256": f.selector.file_sha256(f.plan_path),
        "passes_development_gate": True, "source_hashes": stacking.source_hashes(),
        "selector_sha256": f.selector.file_sha256(Path(f.selector.__file__)),
        "evaluator_sha256": f.definition["script_sha256"],
    })
    out = f.root / "robustness"
    f.selector.prepare_robustness(f.plan_path, selected_path, out)
    rows = pd.read_csv(out / "folds.csv")
    assert rows["group"].nunique() == 143
    for first, second in f.plan["robustness"]["related_prompt_pairs"]:
        pair = rows[rows["prompt_id"].isin([first, second])]
        assert pair["group"].nunique() == 1
        assert pair["fold"].nunique() == 1
    frozen = f.selector.read_json(out / "frozen.json")
    assert frozen["selected_sha256"] == f.selector.file_sha256(selected_path)
    assert frozen["settings"]["seed"] == 31415
    assert frozen["group_count"] == 143
    assert frozen["folds_sha256"] == f.selector.file_sha256(out / "folds.csv")


def test_robustness_checks_only_preselected_candidate_without_reselecting(selection_fixture, capsys):
    f = selection_fixture
    selected_path = f.root / "selected.json"
    selected = {
        "passes_development_gate": True,
        "selected_name": "stack_v3_half", "recipe": f.plan["candidates"]["stack_v3_half"],
        "plan_sha256": f.selector.file_sha256(f.plan_path),
        "source_hashes": stacking.source_hashes(),
        "selector_sha256": f.selector.file_sha256(Path(f.selector.__file__)),
        "evaluator_sha256": f.definition["script_sha256"],
    }
    f.selector.save_json(selected_path, selected)
    f.selector.save_json(f.root / "frozen.json", {
        "plan_sha256": selected["plan_sha256"],
        "selected_sha256": f.selector.file_sha256(selected_path),
        "source_hashes": selected["source_hashes"],
        "selector_sha256": selected["selector_sha256"],
        "evaluator_sha256": selected["evaluator_sha256"],
        "folds_sha256": f.selector.file_sha256(f.fold_path),
    })
    f.definition["outer_prediction_source"] = "fresh_outer_fits"
    f.selector.save_json(f.run / "experiment_definition.json", f.definition)
    f.seal_manifest()
    f.selector.select(f.plan_path, f.run, f.fold_path, f.root / "check", selected_path)
    result = f.selector.read_json(f.root / "check/metrics.json")
    assert result["selected_name"] == selected["selected_name"]
    assert result["recipe"] == selected["recipe"]
    assert list(result["all_results"]) == [selected["selected_name"]]
    assert result["passes"] == (result["delta_from_v3"] >= .005 and result["fold_wins"] >= 4)
    capsys.readouterr()


def test_notebook_builder_embeds_source_and_runtime_group_oof_and_blocks_failed_gate(tmp_path):
    import nbformat

    builder = load_script("build_notebook_v4")
    selected_path, robust_path = tmp_path / "selected.json", tmp_path / "robust.json"
    selected = {
        "passes_development_gate": True, "recipe": {"stack_weight": .5, "v3_weight": .5},
        "source_hashes": stacking.source_hashes(), "input_sha256": {},
        "balanced_accuracy": .4,
        "selector_sha256": builder.digest(Path("scripts/select_v4.py")),
        "evaluator_sha256": builder.digest(Path("scripts/run_nested_v4.py")),
        "plan_sha256": builder.digest(Path("configs/name-that-llm/v4.json")),
    }
    selected_path.write_text(json.dumps(selected))
    robust = {
        "passes": True, "selected_sha256": builder.digest(selected_path),
        "recipe": selected["recipe"], "source_hashes": selected["source_hashes"],
        "input_sha256": {}, "balanced_accuracy": .39, "baseline": {"balanced_accuracy": .38},
        **{key: selected[key] for key in ("selector_sha256", "evaluator_sha256", "plan_sha256")},
    }
    robust_path.write_text(json.dumps(robust))
    path = builder.build(selected_path, robust_path, tmp_path / "notebook")
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'groups=train["prompt_id"]' in code
    assert "NestedStacker(threads=2).fit" in code
    assert "fit_oof=" not in notebook.cells[-1].source
    for name, expected in notebook.metadata["experiment"]["embedded_source_sha256"].items():
        assert builder.digest(Path("src/name_that_llm") / name) == expected
    metadata = json.loads((path.parent / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == metadata["model_sources"] == []
    robust["passes"] = False
    robust_path.write_text(json.dumps(robust))
    with pytest.raises(ValueError, match="gates"):
        builder.build(selected_path, robust_path, tmp_path / "must-not-build")
    assert not (tmp_path / "must-not-build").exists()
    robust["passes"] = True
    for key in ("selector_sha256", "evaluator_sha256", "plan_sha256"):
        original = robust[key]
        robust[key] = "stale"
        robust_path.write_text(json.dumps(robust))
        with pytest.raises(ValueError, match="changed|differ"):
            builder.build(selected_path, robust_path, tmp_path / "must-not-build")
        robust[key] = original
