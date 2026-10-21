"""Focused synthetic contracts for the distinct frozen StyleDistance experiment."""

from contextlib import nullcontext
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from name_that_llm import classifier_v7 as classifier
from name_that_llm import encoder_v7 as encoder
from name_that_llm.baseline import LABELS
from test_v5 import LiteralTokenizer
from test_v6 import grouped_vectors as luar_grouped_vectors, normalized


class ArrayTensor(np.ndarray):
    """Minimal inference tensor stand-in: tests require no Torch or model weights."""

    def unsqueeze(self, axis):
        return array_tensor(np.expand_dims(np.asarray(self), axis))

    def expand(self, shape):
        return array_tensor(np.broadcast_to(np.asarray(self), shape))

    def to(self, dtype):
        return array_tensor(self.astype(dtype))

    def size(self):
        return self.shape

    def clamp(self, min):
        return array_tensor(np.maximum(self, min))

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def array_tensor(values):
    return np.asarray(values).view(ArrayTensor)


def hidden_features(ids):
    # Padding has deliberately nonzero hidden values: the mask must exclude it.
    features = np.arange(768, dtype=np.float32)[None, None, :]
    tokens = np.asarray(ids, dtype=np.float32)[..., None]
    return array_tensor(np.sin(tokens * (features + 1) * .001) +
                        np.cos(tokens * (features + 7) * .0001))


@pytest.fixture
def synthetic_encoder():
    instance = encoder.FrozenStyleDistanceEncoder.__new__(encoder.FrozenStyleDistanceEncoder)
    instance.settings = encoder.EncoderSettings(max_batch_windows=3)
    instance.tokenizer = LiteralTokenizer()
    instance.torch = SimpleNamespace(inference_mode=nullcontext, from_numpy=array_tensor)
    instance.calls = []

    def model(*, input_ids, attention_mask):
        instance.calls.append((input_ids.copy(), attention_mask.copy()))
        return SimpleNamespace(last_hidden_state=hidden_features(input_ids))

    instance.model = model
    return instance


@pytest.mark.parametrize("length", [0, 1, 509, 510, 511, 1019, 1020, 1021, 5101])
def test_windows_preserve_every_raw_token_once(synthetic_encoder, length):
    text = (" A\nZ\t🙂!?é" * (length // 9 + 1))[:length]
    result = synthetic_encoder.tokenize(text)
    assert synthetic_encoder.tokenizer.calls == [text]
    assert result.content_tokens == length
    assert result.windows == max(1, (length + 509) // 510)
    assert all(ids[0] == 0 and ids[-1] == 2 and 2 <= len(ids) <= 512 for ids in result.input_ids)
    assert [token for ids in result.input_ids for token in ids[1:-1]] == [ord(c) + 10 for c in text]


def test_masked_mean_includes_specials_and_excludes_padding():
    hidden = np.zeros((2, 4, 768), dtype=np.float32)
    hidden[0, :, :] = np.asarray([2, 4, 6, 9000])[:, None]
    hidden[1, :, :] = np.asarray([1, 3, 5, 7])[:, None]
    mask = np.asarray([[1, 1, 1, 0], [1, 1, 1, 1]])
    pooled = encoder.pool_window_embeddings(array_tensor(hidden), array_tensor(mask))
    np.testing.assert_array_equal(pooled, np.full((2, 768), 4.))


def test_passage_aggregation_uses_token_weights_not_window_norms_or_other_rows(synthetic_encoder):
    texts = ["A" * 510 + "z", " Case\n\t🙂", "", "b" * 510 + "y" * 499]
    actual = synthetic_encoder.encode(texts)
    assert synthetic_encoder.tokenizer.calls == texts
    expected = []
    for text in texts:
        windows = synthetic_encoder.tokenize(text).input_ids
        # Independent token-level reference: window sum divided by total visible tokens.
        pooled_sum = np.zeros(768)
        for ids in windows:
            pooled_sum += np.asarray(hidden_features([ids]))[0].mean(axis=0).astype(float) * len(ids)
        mean = pooled_sum / sum(map(len, windows))
        expected.append(mean / np.linalg.norm(mean))
    assert actual.dtype == np.float32 and actual.shape == (4, 768)
    np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=3e-6)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=2e-7)
    for ids, masks in synthetic_encoder.calls:
        assert len(ids) <= 3 and ids.shape[1] == masks.sum(axis=1).max()
        assert (ids[np.asarray(masks) == 0] == 1).all()
    together = actual.copy()
    alone = np.concatenate([synthetic_encoder.encode([text]) for text in texts])
    reverse = synthetic_encoder.encode(texts[::-1])[::-1]
    synthetic_encoder.settings = replace(synthetic_encoder.settings, max_batch_windows=1)
    single_window_batches = synthetic_encoder.encode(texts)
    np.testing.assert_allclose(together, alone, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(together, reverse, atol=2e-8, rtol=2e-6)
    np.testing.assert_allclose(together, single_window_batches, atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("kind", ["zero", "nan", "width"])
def test_invalid_model_output_is_rejected(synthetic_encoder, kind):
    def model(*, input_ids, attention_mask):
        shape = (*input_ids.shape, 767 if kind == "width" else 768)
        values = np.full(shape, np.nan if kind == "nan" else 0, dtype=np.float32)
        return SimpleNamespace(last_hidden_state=array_tensor(values))
    synthetic_encoder.model = model
    with pytest.raises(ValueError, match="Invalid StyleDistance"):
        synthetic_encoder.encode(["a"])


@pytest.mark.parametrize("kwargs", [{"content_tokens": 511}, {"sequence_length": 514},
                                     {"max_batch_windows": 0}, {"device": "cuda"}, {"dtype": "bfloat16"}])
def test_encoder_settings_reject_unplanned_preprocessing(kwargs):
    with pytest.raises(ValueError):
        encoder.EncoderSettings(**kwargs)


@pytest.fixture
def synthetic_model_manifest(tmp_path, monkeypatch):
    directory = tmp_path / "checkpoint"
    files, reviewed = [], {}
    for name in encoder.REQUIRED_MODEL_FILES:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {name}\n")
        digest = encoder.sha256_file(path)
        files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": digest})
        if name in encoder.REVIEWED_SOURCE_SHA256:
            reviewed[name] = digest
    manifest = {"model_id": encoder.MODEL_ID, "revision": encoder.MODEL_REVISION, "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(encoder, "REVIEWED_SOURCE_SHA256", reviewed)
    return directory, manifest


def test_constructor_uses_verified_local_weights_and_no_remote_code(synthetic_model_manifest, monkeypatch):
    directory, manifest = synthetic_model_manifest
    calls = []
    parameter = SimpleNamespace(numel=lambda: 768, dtype="float32")

    class Model:
        config = SimpleNamespace(model_type="roberta", hidden_size=768, _attn_implementation="eager")

        def to(self, **kwargs):
            calls.append(("to", kwargs))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

        def requires_grad_(self, value):
            calls.append(("grad", value))

        def parameters(self):
            return iter([parameter])

    tokenizer = LiteralTokenizer()
    tokenizer.num_special_tokens_to_add = lambda pair: 2

    def load_model(path, **kwargs):
        calls.append(("model", path, kwargs))
        return Model(), {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set(), "error_msgs": []}

    def load_tokenizer(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return tokenizer

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float32="float32", set_num_threads=lambda n: calls.append(("threads", n)),
        get_num_threads=lambda: 2, use_deterministic_algorithms=lambda value: calls.append(("deterministic", value))))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=load_model), AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer)))
    monkeypatch.setattr(encoder.importlib.metadata, "version", lambda name: "synthetic-version")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(name, "before-synthetic-test")
    model = encoder.FrozenStyleDistanceEncoder(directory)
    assert ("model", str(directory.resolve()), {"local_files_only": True, "trust_remote_code": False,
        "use_safetensors": True, "dtype": "float32", "output_loading_info": True}) in calls
    assert ("tokenizer", str(directory.resolve()), {"local_files_only": True, "trust_remote_code": False,
        "use_fast": True}) in calls
    assert ("eval",) in calls and ("grad", False) in calls and ("deterministic", True) in calls
    assert ("to", {"device": "cpu", "dtype": "float32"}) in calls
    assert model.provenance()["model_files"] == manifest["files"]
    assert model.runtime["loading_info"]["missing_keys"] == []
    calls.clear()
    (directory / "config.json").write_text("modified after source review")
    with pytest.raises(ValueError, match="manifest mismatch"):
        encoder.FrozenStyleDistanceEncoder(directory)
    assert calls == [], "A changed package must fail before any model/tokenizer load"


@pytest.fixture
def synthetic_cache(synthetic_model_manifest, tmp_path):
    directory, manifest = synthetic_model_manifest
    cache = tmp_path / "cache"
    cache.mkdir()
    plan, source = tmp_path / "plan.json", tmp_path / "source.csv"
    plan.write_text(json.dumps({"encoder": asdict(encoder.EncoderSettings())}))
    source.write_text("synthetic source\n")
    ids, texts = ["a", "b"], [" Case\n", "x" * 511]
    rows = encoder.build_row_manifest(ids, texts)
    values = np.zeros((2, 768), dtype=np.float32)
    values[0, 0] = values[1, 1] = 1
    arrays = {"embeddings": values, "ids": np.asarray(ids),
              "text_sha256": np.asarray([row["text_sha256"] for row in rows]),
              "content_tokens": np.asarray([6, 511]), "window_counts": np.asarray([1, 2])}
    metadata = {"status": "complete", "split": "train", "rows": rows, "shape": [2, 768], "dtype": "float32",
                "plan_sha256": encoder.sha256_file(plan), "source_csv_sha256": encoder.sha256_file(source),
                "script_sha256": encoder.sha256_file(Path(__file__).resolve().parents[1] / "scripts/extract_embeddings_v7.py"),
                "report": {"content_tokens": [6, 511], "window_counts": [1, 2]},
                "provenance": {"model_id": encoder.MODEL_ID, "revision": encoder.MODEL_REVISION,
                    "manifest_sha256": encoder.sha256_file(directory / "manifest.json"), "model_files": manifest["files"],
                    "runtime": {"settings": asdict(encoder.EncoderSettings())},
                    "encoder_source_sha256": encoder.sha256_file(encoder.__file__)}}

    def write():
        path = cache / "train-embeddings.npz"
        np.savez_compressed(path, **arrays)
        metadata["artifact"] = {"path": path.name, "size_bytes": path.stat().st_size, "sha256": encoder.sha256_file(path)}
        (cache / "train-embeddings.json").write_text(json.dumps(metadata))

    def load():
        return encoder.load_embedding_cache(cache, "train", ids, texts, model_dir=directory,
                                            plan_path=plan, source_path=source)
    write()
    return SimpleNamespace(write=write, load=load, metadata=metadata, arrays=arrays, ids=ids, texts=texts,
                           plan=plan, source=source, model=directory)


def test_cache_complete_roundtrip(synthetic_cache):
    values, metadata = synthetic_cache.load()
    np.testing.assert_array_equal(values, synthetic_cache.arrays["embeddings"])
    assert metadata == synthetic_cache.metadata


@pytest.mark.parametrize("change", ["raw_text", "row_order", "model_bytes", "source_csv", "plan",
                                     "source_hash", "vector_width", "vector_norm", "window_coverage"])
def test_cache_rejects_changed_rows_model_inputs_and_vectors(synthetic_cache, change):
    f = synthetic_cache
    if change == "raw_text":
        f.texts[0] = f.texts[0].strip()
    elif change == "row_order":
        f.ids.reverse()
        f.texts.reverse()
    elif change == "model_bytes":
        (f.model / "model.safetensors").write_text("changed bytes")
    elif change == "source_csv":
        f.source.write_text("modified source")
    elif change == "plan":
        f.plan.write_text(f.plan.read_text() + "\n")
    elif change == "source_hash":
        f.metadata["provenance"]["encoder_source_sha256"] = "unbound"
    elif change == "vector_width":
        f.arrays["embeddings"] = f.arrays["embeddings"][:, :512]
    elif change == "vector_norm":
        f.arrays["embeddings"] *= 2
    elif change == "window_coverage":
        f.arrays["window_counts"][1] = 1
        f.metadata["report"]["window_counts"][1] = 1
    f.write()
    with pytest.raises(ValueError):
        f.load()


@pytest.fixture
def grouped_vectors(luar_grouped_vectors):
    values, labels, groups, folds = luar_grouped_vectors
    return np.pad(values, ((0, 0), (0, 256))), labels, groups, folds


def test_probe_and_two_recipes_match_frozen_plan(grouped_vectors):
    plan = json.loads((Path(__file__).resolve().parents[1] / "configs/name-that-llm/v7.json").read_text())
    assert classifier.CLASSIFIER_PARAMETERS == plan["classifier"] == {
        "C": 64., "class_weight": "balanced", "solver": "lbfgs", "max_iter": 3000, "tol": .0001,
    }
    assert classifier.RECIPES == plan["candidates"]
    assert list(classifier.RECIPES) == ["styledistance", "styledistance_v4_half"]
    values, labels, _, folds = grouped_vectors
    model = classifier.fit_classifier(values[folds != 0], labels[folds != 0])
    for name, value in classifier.CLASSIFIER_PARAMETERS.items():
        assert model.get_params()[name] == value
    assert model.get_params()["warm_start"] is False
    assert model.get_params()["penalty"] == "l2"
    assert not hasattr(model, "named_steps")
    scores = classifier.model_evidence(model, values[folds == 0])
    np.testing.assert_allclose(scores, normalized(model.predict_proba(values[folds == 0]), True))
    with pytest.raises(ValueError, match="768"):
        classifier.fit_classifier(values[:, :512], labels)


def test_half_blend_renormalizes_completed_v4_and_ties_favor_standalone():
    rng = np.random.default_rng(7)
    scores, incumbent = normalized(rng.normal(size=(5, 18))), rng.normal(size=(5, 18)) * 4 + 8
    np.testing.assert_array_equal(classifier.combine_evidence(scores, incumbent, classifier.RECIPES["styledistance"]), scores)
    np.testing.assert_allclose(classifier.combine_evidence(scores, incumbent, classifier.RECIPES["styledistance_v4_half"]),
                               .5 * scores + .5 * normalized(incumbent))
    tied = {name: {"balanced_accuracy": .5} for name in reversed(classifier.RECIPES)}
    assert classifier.choose_recipe(tied) == "styledistance"
    with pytest.raises(ValueError):
        classifier.combine_evidence(scores, incumbent, {"embedding_weight": .6, "v4_weight": .4})


def test_each_fold_fits_only_other_groups_and_retains_input_hashes(grouped_vectors, monkeypatch):
    values, labels, groups, folds = grouped_vectors
    lookup = {row.tobytes(): i for i, row in enumerate(values)}
    fits, queries = [], []

    def fit(x, y):
        indexes = np.asarray([lookup[row.tobytes()] for row in x])
        missing = set(range(5)) - set(folds[indexes])
        assert len(missing) == 1
        fold = missing.pop()
        np.testing.assert_array_equal(indexes, np.flatnonzero(folds != fold))
        np.testing.assert_array_equal(y, labels[indexes])
        model = SimpleNamespace(classes_=np.asarray(LABELS), n_iter_=np.asarray([2]), fold=fold)
        fits.append(model)
        return model

    def evidence(model, x):
        indexes = np.asarray([lookup[row.tobytes()] for row in x])
        is_train = np.all(folds[indexes] != model.fold)
        np.testing.assert_array_equal(indexes, np.flatnonzero(folds != model.fold if is_train else folds == model.fold))
        queries.append((model.fold, is_train))
        scores = np.zeros((len(x), 18))
        scores[np.arange(len(x)), [LABELS.index(label) for label in labels[indexes]]] = 1
        return normalized(scores)

    monkeypatch.setattr(classifier, "fit_classifier", fit)
    monkeypatch.setattr(classifier, "model_evidence", evidence)
    scores, audit = classifier.evaluate_oof(values, labels, folds, groups)
    assert len({id(model) for model in fits}) == 5
    assert sorted(queries) == [(f, kind) for f in range(5) for kind in (False, True)]
    for fold, record in enumerate(audit):
        assert record["training_embeddings_sha256"] == classifier.array_hash(values[folds != fold])
        assert record["held_out_scores_sha256"] == classifier.array_hash(scores[folds == fold])
        assert record["training_balanced_accuracy"] == 1
        assert set(record["training_groups"]).isdisjoint(record["held_out_groups"])
    invalid = folds.copy()
    invalid[0] = (invalid[0] + 1) % 5
    with pytest.raises(ValueError, match="crosses folds"):
        classifier.evaluate_oof(values, labels, invalid, groups)


@pytest.fixture
def runner_fixture(tmp_path, monkeypatch, grouped_vectors):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("v7_runner_under_test", scripts / "run_embeddings_v7.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.chdir(tmp_path)
    values, labels, groups, folds = grouped_vectors
    for folder in ("cache", "model", "data/name-that-llm", "artifacts/improvement-v7", "configs/name-that-llm"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for name in ("cache/train-embeddings.npz", "cache/train-embeddings.json", "model/manifest.json",
                 "configs/name-that-llm/v7.json", "predecessor.json", "artifacts/improvement-v7/frozen-plan.json"):
        (tmp_path / name).write_text("synthetic\n")
    train = pd.DataFrame({"id": [f"r-{index}" for index in range(len(values))],
                          "text": [f"synthetic {index}" for index in range(len(values))],
                          "model": labels, "prompt_id": groups})
    train.to_csv(tmp_path / "data/name-that-llm/train.csv", index=False)
    plan = {"predecessors": {"predecessor.json": runner.file_sha256("predecessor.json")},
            "development": {"min_gain": .0075}, "robustness": {"min_gain": .005, "min_fold_wins": 4,
                "seed": 31415, "bootstrap_draws": 50}, "limitations": ["Synthetic reused data."]}
    monkeypatch.setattr(runner, "validate_plan", lambda path: plan)
    correct = np.zeros((len(labels), 18))
    correct[np.arange(len(labels)), [LABELS.index(label) for label in labels]] = 1.
    scores = normalized(correct)
    wrong = np.roll(correct, 1, axis=1)
    baseline, predictions = runner.evaluate(wrong, labels, folds)
    monkeypatch.setattr(runner, "load_incumbent", lambda plan, mode: (
        train, folds, groups, wrong, baseline, predictions, []))
    monkeypatch.setattr(runner, "load_embedding_cache", lambda *args, **kwargs: (
        values, {"provenance": {"runtime": {"synthetic": True}, "model_files": []}}))
    calls = []

    def evaluate(x, y, f, g):
        np.testing.assert_array_equal(x, values)
        np.testing.assert_array_equal(y, labels)
        np.testing.assert_array_equal(f, folds)
        np.testing.assert_array_equal(g, groups)
        calls.append("fixed_lr64")
        return scores.copy(), [{"synthetic": True}]

    monkeypatch.setattr(runner, "evaluate_oof", evaluate)

    def execute(mode):
        runner.run(mode, tmp_path / "configs/name-that-llm/v7.json", tmp_path / "cache", tmp_path / "model",
                   tmp_path / mode, tmp_path / "development/selected.json")
        return json.loads((tmp_path / mode / ("selected.json" if mode == "development" else "metrics.json")).read_text())

    return SimpleNamespace(runner=runner, root=tmp_path, execute=execute, calls=calls)


def test_driver_reuses_one_probe_for_two_recipes_then_evaluates_only_winner(runner_fixture):
    f = runner_fixture
    result = f.execute("development")
    assert list(result["all_results"]) == list(classifier.RECIPES)
    assert result["selected_name"] == "styledistance" and result["passes_development_gate"]
    robust = f.execute("robustness")
    assert list(robust["all_results"]) == ["styledistance"] and robust["passes"]
    assert f.calls == ["fixed_lr64", "fixed_lr64"]


@pytest.mark.parametrize("filename", ["embedding_oof_evidence.npy", "score_schema.json", "fit_audit.json", "selected.json"])
def test_driver_rejects_modified_development_artifacts_before_fitting(runner_fixture, filename):
    f = runner_fixture
    f.execute("development")
    with (f.root / "development" / filename).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError):
        f.execute("robustness")
    assert f.calls == ["fixed_lr64"]
    assert not (f.root / "robustness").exists()


@pytest.mark.parametrize("passed,delta", [(False, .02), (True, .006)])
def test_notebook_builder_refuses_failed_development_before_reading_robustness(tmp_path, monkeypatch, passed, delta):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("v7_builder_under_test", scripts / "build_notebook_v7.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builder, "validate_plan", lambda path: {"development": {"min_gain": .0075}})
    monkeypatch.setattr(builder, "verify_selection", lambda path: {
        "passes_development_gate": passed, "delta_from_v4": delta})
    output = tmp_path / "notebook"
    with pytest.raises(ValueError, match="development gate"):
        builder.build(output)
    assert not output.exists()
