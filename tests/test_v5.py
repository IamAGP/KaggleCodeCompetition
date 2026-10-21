"""Synthetic contracts for frozen LUAR v5; no private data or weights required."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from name_that_llm import embeddings_v5 as embeddings
from name_that_llm import classifier_v5 as classifier
from name_that_llm.baseline import LABELS
from name_that_llm import stacking_v4 as stacking


class LiteralTokenizer:
    """One Unicode character per content token, preserving whitespace and case."""

    pad_token_id = 1
    cls_token_id = 0
    sep_token_id = 2

    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        assert kwargs == {
            "add_special_tokens": False, "truncation": False,
            "return_attention_mask": False, "verbose": False,
        }
        self.calls.append(text)
        return {"input_ids": [ord(character) + 10 for character in text]}

    def build_inputs_with_special_tokens(self, content):
        return [0, *content, 2]


class NumpyTensor(np.ndarray):
    """Tiny inference-only test double so the base test suite needs no Torch."""

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(value):
    return np.asarray(value).view(NumpyTensor)


class RecordingEpisodeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, *, input_ids, attention_mask, document_batch_size):
        self.calls.append((input_ids.copy(), attention_mask.copy(), document_batch_size))
        assert input_ids.ndim == 3 and input_ids.shape == attention_mask.shape
        assert input_ids.shape[2] == 128
        # A row-dependent direction, sensitive to the number of episode windows.
        # It deliberately makes accidental merging/padding of episodes visible.
        visible = input_ids * attention_mask
        sums = visible.sum(axis=(1, 2), dtype=np.float64)
        squares = (visible.astype(np.float64) ** 2).sum(axis=(1, 2))
        features = np.arange(512, dtype=np.float64)[None, :]
        values = np.sin(sums[:, None] * (features + 1) * .000017)
        values += np.cos(squares[:, None] * (features + 7) * .000000001)
        values += input_ids.shape[1] / (features + 1)
        return tensor(values.astype(np.float32))


@pytest.fixture
def synthetic_encoder():
    instance = embeddings.FrozenLuarEncoder.__new__(embeddings.FrozenLuarEncoder)
    instance.settings = embeddings.EncoderSettings(max_batch_windows=8)
    instance.tokenizer = LiteralTokenizer()
    instance.model = RecordingEpisodeModel()
    instance.torch = SimpleNamespace(
        inference_mode=nullcontext,
        from_numpy=tensor,
        isfinite=np.isfinite,
        linalg=SimpleNamespace(vector_norm=lambda value, ord, dim, keepdim: tensor(
            np.linalg.norm(value, ord=ord, axis=dim, keepdims=keepdim)
        )),
    )
    instance.report = {}
    return instance


@pytest.mark.parametrize("length", [0, 1, 125, 126, 127, 251, 252, 253, 5000])
def test_windows_cover_every_content_token_once(synthetic_encoder, length):
    text = (" A\nZ\t🙂!?é" * (length // 9 + 1))[:length]
    episode = synthetic_encoder.tokenize(text)
    assert synthetic_encoder.tokenizer.calls == [text]
    assert episode.content_tokens == len(text)
    assert episode.windows == max(1, (len(text) + 125) // 126)
    assert episode.input_ids.shape == episode.attention_mask.shape == (episode.windows, 128)
    assert episode.input_ids.dtype == episode.attention_mask.dtype == np.int64
    observed = []
    for index, (ids, mask) in enumerate(zip(episode.input_ids, episode.attention_mask, strict=True)):
        count = min(126, max(0, length - index * 126))
        assert mask.tolist() == [1] * (count + 2) + [0] * (126 - count)
        assert ids[0] == 0 and ids[count + 1] == 2
        assert (ids[count + 2:] == 1).all()
        observed.extend(ids[1:count + 1])
    assert observed == [ord(character) + 10 for character in text]


def test_encoder_preserves_rows_and_exact_episode_lengths(synthetic_encoder):
    texts = ["a" * 253, "z", "b" * 127, " Raw Case!\n\t", "c" * 127, "long" * 300]
    expected = [synthetic_encoder.tokenize(text) for text in texts]
    synthetic_encoder.tokenizer.calls.clear()
    progress = []
    actual = synthetic_encoder.encode(texts, progress=progress.append)
    assert synthetic_encoder.tokenizer.calls == texts
    assert actual.shape == (len(texts), 512) and actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=2e-7)
    remaining = list(range(len(texts)))
    for ids, masks, batch_size in synthetic_encoder.model.calls:
        assert batch_size == synthetic_encoder.settings.document_batch_size
        assert len(ids) * ids.shape[1] <= 8 or len(ids) == 1
        for row_ids, row_mask in zip(ids, masks, strict=True):
            matched = [index for index in remaining
                       if np.array_equal(row_ids, expected[index].input_ids)
                       and np.array_equal(row_mask, expected[index].attention_mask)]
            assert len(matched) == 1, "Every episode must contain exactly one original row's windows"
            remaining.remove(matched[0])
    assert remaining == []
    assert synthetic_encoder.report["window_counts"] == [episode.windows for episode in expected]
    assert synthetic_encoder.report["total_windows"] == sum(episode.windows for episode in expected)
    assert progress[-1]["completed_rows"] == len(texts)


def test_encode_is_batch_and_order_independent(synthetic_encoder):
    texts = ["z" * 127, "Case\n ", "a" * 253, "b", "c" * 127]
    together = synthetic_encoder.encode(texts)
    alone = np.concatenate([synthetic_encoder.encode([text]) for text in texts])
    reverse = synthetic_encoder.encode(texts[::-1])[::-1]
    synthetic_encoder.settings = replace(synthetic_encoder.settings, max_batch_windows=1)
    small_batches = synthetic_encoder.encode(texts)
    np.testing.assert_array_equal(together, alone)
    np.testing.assert_array_equal(together, reverse)
    np.testing.assert_array_equal(together, small_batches)


@pytest.mark.parametrize("value", [None, 123, float("nan"), ["nested"]])
def test_nonstring_completion_is_rejected(synthetic_encoder, value):
    with pytest.raises(TypeError):
        synthetic_encoder.tokenize(value)


@pytest.mark.parametrize("kind", ["zero", "nan", "inf", "wrong_width"])
def test_bad_embeddings_are_rejected(synthetic_encoder, kind):
    def invalid_model(**kwargs):
        n = len(kwargs["input_ids"])
        result = np.ones((n, 511 if kind == "wrong_width" else 512), dtype=np.float32)
        if kind in {"zero", "nan", "inf"}:
            result[:] = {"zero": 0., "nan": np.nan, "inf": np.inf}[kind]
        return tensor(result)
    synthetic_encoder.model = invalid_model
    with pytest.raises(ValueError, match="Invalid LUAR"):
        synthetic_encoder.encode(["example"])


@pytest.mark.parametrize("kwargs", [
    {"content_tokens": 127}, {"sequence_length": 256}, {"max_batch_windows": 0},
    {"document_batch_size": 0}, {"cpu_threads": 0}, {"device": "cuda"},
])
def test_frozen_encoder_settings_reject_drift(kwargs):
    with pytest.raises(ValueError):
        embeddings.EncoderSettings(**kwargs)


def test_row_provenance_binds_order_ids_and_raw_utf8():
    ids = ["r1", "r2", "r3"]
    texts = [" Case\n", "case", "é🙂"]
    manifest = embeddings.build_row_manifest(ids, texts)
    assert manifest == [
        {"row_index": i, "id": identity, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        for i, (identity, text) in enumerate(zip(ids, texts, strict=True))
    ]
    assert manifest != embeddings.build_row_manifest(ids[::-1], texts[::-1])
    assert manifest != embeddings.build_row_manifest(ids, [text.strip() for text in texts])
    assert "text" not in manifest[0]
    with pytest.raises(ValueError):
        embeddings.build_row_manifest(["r1", "r1"], texts[:2])
    with pytest.raises(ValueError):
        embeddings.build_row_manifest(ids, texts[:2])
    with pytest.raises(ValueError):
        embeddings.build_row_manifest([""], ["valid"])


@pytest.fixture
def synthetic_model_manifest(tmp_path, monkeypatch):
    # Synthetic files only: no dependency on downloaded/private artifacts.
    reviewed = {}
    files = []
    for name in embeddings.REQUIRED_MODEL_FILES:
        path = tmp_path / name
        path.write_text(f"synthetic fixture for {name}\n")
        digest = embeddings.sha256_file(path)
        if name in embeddings.REVIEWED_SOURCE_SHA256:
            reviewed[name] = digest
        files.append({"path": name, "size_bytes": path.stat().st_size, "sha256": digest})
    manifest = {"model_id": embeddings.MODEL_ID, "revision": embeddings.MODEL_REVISION, "files": files}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(embeddings, "REVIEWED_SOURCE_SHA256", reviewed)
    return tmp_path, manifest


def test_manifest_accepts_complete_bound_package(synthetic_model_manifest):
    path, manifest = synthetic_model_manifest
    assert embeddings.validate_model_manifest(path) == manifest


@pytest.mark.parametrize("change", ["revision", "weight_bytes", "missing_file", "source_hash", "symlink"])
def test_manifest_rejects_unbound_or_changed_package(synthetic_model_manifest, change):
    path, manifest = synthetic_model_manifest
    manifest = deepcopy(manifest)
    if change == "revision":
        manifest["revision"] = "unreviewed-revision"
    elif change == "weight_bytes":
        (path / "model.safetensors").write_text("changed weights")
    elif change == "missing_file":
        (path / "tokenizer.json").unlink()
    elif change == "source_hash":
        (path / "model.py").write_text("changed source, with a self-consistent manifest")
        entry = next(item for item in manifest["files"] if item["path"] == "model.py")
        entry.update(size_bytes=(path / "model.py").stat().st_size,
                     sha256=embeddings.sha256_file(path / "model.py"))
    elif change == "symlink":
        (path / "model.safetensors").rename(path / "elsewhere")
        (path / "model.safetensors").symlink_to(path / "elsewhere")
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        embeddings.validate_model_manifest(path)


def test_constructor_loads_verified_local_safetensors_and_freezes_model(synthetic_model_manifest, monkeypatch):
    path, _ = synthetic_model_manifest
    calls = []
    parameter = SimpleNamespace(numel=lambda: 42, dtype="float32")

    class Model:
        transformer = SimpleNamespace(config=SimpleNamespace(_attn_implementation="eager"))

        def to(self, **kwargs):
            calls.append(("to", kwargs))
            return self

        def eval(self):
            calls.append(("eval", {}))
            return self

        def requires_grad_(self, value):
            calls.append(("requires_grad", value))

        def parameters(self):
            return iter([parameter])

    config = object()
    model = Model()

    def load_config(directory, **kwargs):
        calls.append(("config", directory, kwargs))
        return config

    def load_model(directory, **kwargs):
        calls.append(("model", directory, kwargs))
        return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}

    def load_tokenizer(directory, **kwargs):
        calls.append(("tokenizer", directory, kwargs))
        return SimpleNamespace(num_special_tokens_to_add=lambda pair: 2,
                               cls_token_id=0, sep_token_id=2, pad_token_id=1)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        float32="float32", set_num_threads=lambda value: calls.append(("threads", value)),
        use_deterministic_algorithms=lambda value: calls.append(("deterministic", value)),
        get_num_threads=lambda: 2,
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
    ))
    monkeypatch.setattr(embeddings.importlib.metadata, "version", lambda name: "synthetic")
    monkeypatch.setattr(embeddings.FrozenLuarEncoder, "_load_local_classes", lambda self: (
        SimpleNamespace(from_pretrained=load_config), SimpleNamespace(from_pretrained=load_model),
    ))
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        monkeypatch.delenv(variable, raising=False)
    instance = embeddings.FrozenLuarEncoder(path)
    assert ("config", str(path.resolve()), {"local_files_only": True}) in calls
    assert ("model", str(path.resolve()), {
        "config": config, "local_files_only": True, "use_safetensors": True, "output_loading_info": True,
    }) in calls
    assert ("tokenizer", str(path.resolve()), {
        "local_files_only": True, "trust_remote_code": False, "use_fast": True,
    }) in calls
    assert ("to", {"device": "cpu", "dtype": "float32"}) in calls
    assert ("eval", {}) in calls and ("requires_grad", False) in calls
    assert ("deterministic", True) in calls
    assert embeddings.os.environ["HF_HUB_OFFLINE"] == embeddings.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert instance.runtime["loading_info"]["missing_keys"] == []
    assert instance.provenance()["manifest_sha256"] == embeddings.sha256_file(path / "manifest.json")


def test_manifest_validation_precedes_any_local_source_execution(synthetic_model_manifest, monkeypatch):
    path, _ = synthetic_model_manifest
    (path / "model.py").write_text("unreviewed source")

    def forbidden_load(self):
        pytest.fail("Source must not execute before package validation")

    monkeypatch.setattr(embeddings.FrozenLuarEncoder, "_load_local_classes", forbidden_load)
    with pytest.raises(ValueError, match="manifest mismatch"):
        embeddings.FrozenLuarEncoder(path)


@pytest.fixture
def synthetic_cache(synthetic_model_manifest, tmp_path):
    model_dir, model_manifest = synthetic_model_manifest
    directory = tmp_path / "cache"
    directory.mkdir()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"encoder": asdict(embeddings.EncoderSettings())}))
    source_path = tmp_path / "train.csv"
    source_path.write_text("id,text,model\na,one,label-unused\nb,two,label-unused\n")
    ids, texts = ["a", "b"], [" One\n", "é" * 127]
    rows = embeddings.build_row_manifest(ids, texts)
    values = np.zeros((2, 512), dtype=np.float32)
    values[0, 0] = values[1, 1] = 1.
    arrays = {"embeddings": values, "ids": np.asarray(ids),
              "text_sha256": np.asarray([row["text_sha256"] for row in rows]),
              "content_tokens": np.asarray([5, 127]), "window_counts": np.asarray([1, 2])}
    metadata = {
        "status": "complete", "split": "train", "rows": rows, "shape": [2, 512], "dtype": "float32",
        "plan_sha256": embeddings.sha256_file(plan_path),
        "source_csv_sha256": embeddings.sha256_file(source_path),
        "script_sha256": embeddings.sha256_file(Path(__file__).resolve().parents[1] / "scripts/extract_embeddings_v5.py"),
        "provenance": {
            "model_id": embeddings.MODEL_ID, "revision": embeddings.MODEL_REVISION,
            "manifest_sha256": embeddings.sha256_file(model_dir / "manifest.json"),
            "model_files": model_manifest["files"],
            "runtime": {"settings": asdict(embeddings.EncoderSettings())},
            "encoder_source_sha256": embeddings.sha256_file(embeddings.__file__),
        },
        "report": {"content_tokens": [5, 127], "window_counts": [1, 2]},
    }

    def write():
        artifact = directory / "train-embeddings.npz"
        np.savez_compressed(artifact, **arrays)
        metadata["artifact"] = {"path": artifact.name, "size_bytes": artifact.stat().st_size,
                                "sha256": embeddings.sha256_file(artifact)}
        (directory / "train-embeddings.json").write_text(json.dumps(metadata))

    def load():
        return embeddings.load_embedding_cache(directory, "train", ids, texts,
            model_dir=model_dir, plan_path=plan_path, source_path=source_path)

    write()
    return SimpleNamespace(write=write, load=load, arrays=arrays, metadata=metadata,
                           directory=directory, ids=ids, texts=texts, plan_path=plan_path,
                           source_path=source_path, model_dir=model_dir)


def test_cache_roundtrip_is_self_contained(synthetic_cache):
    fixture = synthetic_cache
    values, metadata = fixture.load()
    np.testing.assert_array_equal(values, fixture.arrays["embeddings"])
    assert metadata == fixture.metadata


@pytest.mark.parametrize("change", [
    "row_order", "raw_text", "source_csv", "plan", "model_identity", "model_manifest",
    "model_files", "settings", "encoder_source", "extractor_source", "cache_bytes",
    "internal_ids", "internal_text", "windows", "negative_tokens", "count_metadata",
    "vector_norm", "vector_finite", "vector_dtype", "vector_width", "metadata_shape",
])
def test_cache_rejects_provenance_and_content_tampering(synthetic_cache, change):
    f = synthetic_cache
    p = f.metadata["provenance"]
    if change == "row_order":
        f.ids.reverse()
        f.texts.reverse()
    elif change == "raw_text":
        f.texts[0] = f.texts[0].strip()
    elif change == "source_csv":
        f.source_path.write_text("changed source\n")
    elif change == "plan":
        f.plan_path.write_text(f.plan_path.read_text() + "\n")
    elif change == "model_identity":
        p["revision"] = "other-revision"
    elif change == "model_manifest":
        p["manifest_sha256"] = "unbound"
    elif change == "model_files":
        p["model_files"] = []
    elif change == "settings":
        p["runtime"]["settings"]["content_tokens"] = 125
    elif change == "encoder_source":
        p["encoder_source_sha256"] = "unbound"
    elif change == "extractor_source":
        f.metadata["script_sha256"] = "unbound"
    elif change == "internal_ids":
        f.arrays["ids"] = f.arrays["ids"][::-1]
    elif change == "internal_text":
        f.arrays["text_sha256"] = f.arrays["text_sha256"][::-1]
    elif change == "windows":
        f.arrays["window_counts"][1] = 1
    elif change == "negative_tokens":
        f.arrays["content_tokens"][0] = -1
    elif change == "count_metadata":
        f.metadata["report"]["content_tokens"][0] = 6
    elif change == "vector_norm":
        f.arrays["embeddings"][0] *= 2
    elif change == "vector_finite":
        f.arrays["embeddings"][0, 0] = np.nan
    elif change == "vector_dtype":
        f.arrays["embeddings"] = f.arrays["embeddings"].astype(np.float64)
    elif change == "vector_width":
        f.arrays["embeddings"] = f.arrays["embeddings"][:, :511]
    elif change == "metadata_shape":
        f.metadata["shape"] = [2, 511]
    f.write()
    if change == "cache_bytes":
        with (f.directory / "train-embeddings.npz").open("ab") as handle:
            handle.write(b"tamper")
    with pytest.raises(ValueError):
        f.load()


def independent_evidence(values, *, probabilities=False):
    values = np.asarray(values, dtype=float)
    if probabilities:
        values = np.log(np.maximum(values, 1e-12))
    centered = values - np.mean(values, axis=1, keepdims=True)
    return centered / np.maximum(np.std(centered, axis=1, keepdims=True), 1e-8)


def test_incumbent_and_half_blend_match_preregistered_equations():
    rng = np.random.default_rng(810)
    stack_probs = rng.dirichlet(np.ones(18), size=7)
    luar_probs = rng.dirichlet(np.ones(18), size=7)
    base = rng.normal(size=(7, 4, 18))
    for component in range(4):
        base[:, component] = independent_evidence(base[:, component])
    # Component 2 must be entirely absent from the fixed v4 selected branch.
    base[:, 2] *= 99
    expected = .5 * independent_evidence(stack_probs, probabilities=True)
    expected += .5 * (base[:, 0] + base[:, 1] + base[:, 3]) / 3
    v4 = classifier.incumbent_evidence(stack_probs, base)
    np.testing.assert_allclose(v4, expected, atol=1e-14)
    expected_half = .5 * independent_evidence(luar_probs, probabilities=True) + .5 * independent_evidence(expected)
    actual_half = classifier.combine_evidence(luar_probs, v4, classifier.RECIPES["luar_v4_half"])
    np.testing.assert_allclose(actual_half, expected_half, atol=1e-14)
    np.testing.assert_allclose(classifier.combine_evidence(luar_probs, v4, classifier.RECIPES["luar"]),
                               independent_evidence(luar_probs, probabilities=True))
    with pytest.raises(ValueError, match="frozen candidates"):
        classifier.combine_evidence(luar_probs, v4, {"luar_weight": .6, "v4_weight": .4})


@pytest.fixture
def grouped_embedding_rows():
    labels = np.tile(np.asarray(LABELS), 10)
    groups = np.repeat(np.asarray([f"prompt-{i}" for i in range(10)]), 18)
    folds = np.repeat(np.arange(10) // 2, 18)
    values = np.zeros((len(labels), 512), dtype=np.float32)
    values[np.arange(len(labels)), np.tile(np.arange(18), 10)] = 1.
    values[:, 20] = np.arange(len(labels)) / 1000
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    order = np.random.default_rng(123).permutation(len(labels))
    return values[order], labels[order], groups[order], folds[order]


def test_fixed_logistic_probe_uses_no_scaler_or_warm_start(grouped_embedding_rows):
    values, labels, _, _ = grouped_embedding_rows
    first = classifier.fit_classifier(values, labels)
    second = classifier.fit_classifier(values, labels)
    assert first is not second
    parameters = first.get_params()
    assert parameters["C"] == 1.0 and parameters["class_weight"] == "balanced"
    assert parameters["solver"] == "lbfgs" and parameters["max_iter"] == 3000
    assert parameters["penalty"] == "l2" and parameters["warm_start"] is False
    assert tuple(first.classes_) == LABELS
    np.testing.assert_array_equal(first.predict(values), labels)
    np.testing.assert_allclose(first.predict_proba(values), second.predict_proba(values), atol=1e-14)


def test_oof_fits_fresh_models_on_exact_other_prompt_folds(grouped_embedding_rows, monkeypatch):
    values, labels, groups, folds = grouped_embedding_rows
    row_by_bytes = {row.tobytes(): index for index, row in enumerate(values)}
    observed = []

    class HeldOutOnlyModel:
        classes_ = np.asarray(LABELS)
        n_iter_ = np.asarray([1])

        def __init__(self, train_rows):
            self.train_rows = set(train_rows)

        def predict_proba(self, query):
            rows = [row_by_bytes[row.tobytes()] for row in query]
            assert self.train_rows.isdisjoint(rows)
            assert {groups[i] for i in self.train_rows}.isdisjoint(groups[rows])
            assert set(rows) == set(range(len(values))) - self.train_rows
            probabilities = np.full((len(rows), 18), .1 / 17)
            probabilities[np.arange(len(rows)), [LABELS.index(labels[i]) for i in rows]] = .9
            return probabilities

    def fit(train, target):
        rows = [row_by_bytes[row.tobytes()] for row in train]
        assert np.array_equal(target, labels[rows])
        assert len(set(folds[rows])) == 4
        model = HeldOutOnlyModel(rows)
        observed.append(model)
        return model

    monkeypatch.setattr(classifier, "fit_classifier", fit)
    actual, audit = classifier.evaluate_oof(values, labels, folds, groups)
    assert len(observed) == len({id(model) for model in observed}) == 5
    np.testing.assert_array_equal(np.asarray(LABELS)[actual.argmax(axis=1)], labels)
    for fold, record in enumerate(audit):
        assert record["held_out_fold"] == fold
        assert set(record["training_groups"]).isdisjoint(record["held_out_groups"])
        assert record["training_indexes_sha256"] == classifier.array_hash(np.flatnonzero(folds != fold))
        assert record["held_out_indexes_sha256"] == classifier.array_hash(np.flatnonzero(folds == fold))
    leaky = folds.copy()
    leaky[0] = (leaky[0] + 1) % 5
    with pytest.raises(ValueError, match="crosses folds"):
        classifier.evaluate_oof(values, labels, leaky, groups)


@pytest.mark.parametrize("bad", [
    np.ones((3, 511), dtype=np.float32), np.zeros((3, 512), dtype=np.float32),
    np.full((3, 512), np.nan, dtype=np.float32), np.ones((3, 512), dtype=np.int64),
])
def test_classifier_rejects_invalid_frozen_vectors(bad):
    with pytest.raises(ValueError):
        classifier.validate_embeddings(bad)


def load_v5_runner(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("v5_runner_under_test", scripts / "run_embeddings_v5.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_is_paired_at_prompt_group_level(monkeypatch):
    runner = load_v5_runner(monkeypatch)
    labels = np.tile(np.asarray(LABELS), 3)
    groups = np.repeat(["group-a", "group-b", "group-c"], 18)
    candidate = labels.copy()
    incumbent = labels.copy()
    # Paired score differences are exactly +1, -1, 0 per prompt group.
    incumbent[:18] = np.roll(labels[:18], 1)
    candidate[18:36] = np.roll(labels[18:36], 1)
    settings = {"seed": 31415, "bootstrap_draws": 1000}
    draws = np.random.default_rng(31415).integers(0, 3, (1000, 3))
    expected = np.quantile(np.asarray([1., -1., 0.])[draws].mean(axis=1), [.025, .975])
    np.testing.assert_allclose(runner.bootstrap_delta(labels, candidate, incumbent, groups, settings), expected)
    assert runner.bootstrap_delta(labels, candidate, candidate, groups, settings) == [0., 0.]


@pytest.fixture
def incumbent_producer(tmp_path, monkeypatch, grouped_embedding_rows):
    runner = load_v5_runner(monkeypatch)
    values, labels, groups, folds = grouped_embedding_rows
    train = pd.DataFrame({"id": [f"row-{index}" for index in range(len(values))],
                          "text": [f"synthetic passage {index}" for index in range(len(values))],
                          "model": labels, "prompt_id": groups})
    monkeypatch.setattr(sys.modules["select_v4"], "load_data", lambda _: (train, pd.DataFrame()))
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data/name-that-llm"
    data.mkdir(parents=True)
    train.to_csv(data / "train.csv", index=False)
    plan_dir = tmp_path / "configs/name-that-llm"
    plan_dir.mkdir(parents=True)
    (plan_dir / "v4.json").write_text('{"synthetic": true}\n')
    fold_path = tmp_path / "folds.csv"
    rows = train[["id", "prompt_id", "model"]].copy()
    rows["fold"] = folds
    rows.to_csv(fold_path, index=False)
    output = tmp_path / "incumbent"
    output.mkdir()
    rng = np.random.default_rng(99)
    probs = rng.dirichlet(np.ones(18), size=len(values))
    evidence = np.stack([independent_evidence(rng.normal(size=(len(values), 18))) for _ in range(4)], axis=1)
    np.save(output / "stack_oof_probabilities.npy", probs)
    np.save(output / "outer_oof_evidence.npy", evidence)
    rows["prediction"] = np.asarray(LABELS)[probs.argmax(axis=1)]
    rows.to_csv(output / "oof_predictions.csv", index=False)
    script = Path(sys.modules["select_v4"].__file__).with_name("run_nested_v4.py")
    definition = {"plan_sha256": runner.file_sha256(plan_dir / "v4.json"),
                  "source_hashes": stacking.source_hashes(), "script_sha256": runner.file_sha256(script),
                  "folds_sha256": runner.file_sha256(fold_path), "train_sha256": runner.file_sha256(data / "train.csv"),
                  "labels": list(LABELS), "meta_parameters": stacking.META_PARAMETERS,
                  "recipe": stacking.recipe_definition(), "outer_prediction_source": "audited_original_v2_v3_oof"}

    def write_json(name, value):
        (output / name).write_text(json.dumps(value))

    write_json("experiment_definition.json", definition)
    write_json("score_schema.json", {"labels": list(LABELS), "components": list(stacking.COMPONENTS),
                "source_hashes": definition["source_hashes"], "folds_sha256": definition["folds_sha256"]})
    write_json("metrics.json", {"synthetic": True})
    write_json("completion.json", {"all_checks_passed": True, "source_unchanged_during_run": True,
                                  "metrics": {"synthetic": True}})

    def seal():
        names = ("stack_oof_probabilities.npy", "outer_oof_evidence.npy", "oof_predictions.csv",
                 "experiment_definition.json", "completion.json", "score_schema.json", "metrics.json")
        write_json("artifacts.json", {"files": [{"path": name, "bytes": (output / name).stat().st_size,
                    "sha256": runner.file_sha256(output / name)} for name in names]})

    seal()
    incumbent = classifier.incumbent_evidence(probs, evidence)
    metric, prediction = runner.evaluate(incumbent, labels, folds)
    plan = {"development": {"incumbent_run": str(output), "folds": str(fold_path),
                             "folds_sha256": runner.file_sha256(fold_path)},
            "baseline": {"development_balanced_accuracy": metric["balanced_accuracy"]}}
    return SimpleNamespace(runner=runner, plan=plan, root=tmp_path, output=output, definition=definition,
                           fold_path=fold_path, train=train, folds=folds, groups=groups, probs=probs,
                           evidence=evidence, incumbent=incumbent, write_json=write_json, seal=seal)


def test_incumbent_producer_is_checked_and_reproduced(incumbent_producer):
    f = incumbent_producer
    # The producer validator performs the real source/output/label-order checks.
    actual = f.runner.verify_producer(f.output, f.fold_path, f.root / "configs/name-that-llm/v4.json")
    assert actual["outer_prediction_source"] == "audited_original_v2_v3_oof"
    train, folds, groups, stack, base = f.runner.load_outputs(f.output, f.fold_path)
    np.testing.assert_allclose(.5 * stack + .5 * base, f.incumbent)
    assert train.id.tolist() == f.train.id.tolist()
    np.testing.assert_array_equal(folds, f.folds)
    np.testing.assert_array_equal(groups, f.groups)


@pytest.mark.parametrize("change", ["output_bytes", "producer_source", "producer_labels", "source_script", "rows", "folds"])
def test_incumbent_producer_rejects_stale_outputs_or_sources(incumbent_producer, change):
    f = incumbent_producer
    if change == "output_bytes":
        np.save(f.output / "stack_oof_probabilities.npy", f.probs[:, ::-1])
    elif change in {"producer_source", "producer_labels", "source_script"}:
        if change == "producer_source":
            f.definition["source_hashes"]["stacking_v4.py"] = "changed-source"
        elif change == "producer_labels":
            f.definition["labels"].reverse()
        else:
            f.definition["script_sha256"] = "changed-script"
        f.write_json("experiment_definition.json", f.definition)
        f.seal()
    elif change == "rows":
        rows = pd.read_csv(f.output / "oof_predictions.csv")
        rows.iloc[::-1].to_csv(f.output / "oof_predictions.csv", index=False)
        f.seal()
    elif change == "folds":
        f.fold_path.write_text(f.fold_path.read_text() + "\n")
    with pytest.raises(ValueError):
        f.runner.verify_producer(f.output, f.fold_path, f.root / "configs/name-that-llm/v4.json")
        f.runner.load_outputs(f.output, f.fold_path)


@pytest.fixture
def runner_fixture(tmp_path, monkeypatch, grouped_embedding_rows):
    runner = load_v5_runner(monkeypatch)
    values, labels, groups, folds = grouped_embedding_rows
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    for folder in ("configs/name-that-llm", "artifacts/improvement-v5", "scripts", "data/name-that-llm", "cache", "model"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/name-that-llm/v4.json").write_text('{"synthetic": true}\n')
    for name in ("select_v4.py", "extract_embeddings_v5.py"):
        (tmp_path / "scripts" / name).write_bytes((repository / "scripts" / name).read_bytes())
    train = pd.DataFrame({"id": [f"r-{index}" for index in range(len(values))],
                          "text": [f"synthetic {index}" for index in range(len(values))],
                          "model": labels, "prompt_id": groups})
    train.to_csv(tmp_path / "data/name-that-llm/train.csv", index=False)
    for path in ("cache/train-embeddings.npz", "cache/train-embeddings.json", "model/manifest.json"):
        (tmp_path / path).write_text("Synthetic input, mocked cache loader only.\n")
    plan = {"classifier": classifier.CLASSIFIER_PARAMETERS, "candidates": classifier.RECIPES,
            "baseline": {"plan_sha256": runner.file_sha256(tmp_path / "configs/name-that-llm/v4.json")},
            "development": {"min_gain": .0075}, "robustness": {"min_gain": .005,
                "min_fold_wins": 4, "seed": 31415, "bootstrap_draws": 100},
            "limitations": ["Synthetic reused-split fixture."]}
    plan_path = tmp_path / "configs/name-that-llm/v5.json"
    plan_path.write_text(json.dumps(plan))
    (tmp_path / "artifacts/improvement-v5/frozen-plan.json").write_bytes(plan_path.read_bytes())
    probabilities = np.full((len(values), 18), .001 / 17)
    probabilities[np.arange(len(values)), [LABELS.index(label) for label in labels]] = .999
    called = []
    robust_error_folds = set(range(5))

    def incumbent(plan, mode):
        prediction = labels.copy()
        wrong = np.asarray([fold in (set(range(5)) if mode == "development" else robust_error_folds)
                            for fold in folds])
        prediction[wrong] = np.asarray([LABELS[(LABELS.index(label) + 1) % 18] for label in labels[wrong]])
        scores = np.full((len(labels), 18), -1.)
        scores[np.arange(len(labels)), [LABELS.index(label) for label in prediction]] = 1.
        metric, _ = runner.evaluate(scores, labels, folds)
        return train, folds, groups, scores, metric, prediction, []

    def oof(input_values, input_labels, input_folds, input_groups):
        np.testing.assert_array_equal(input_values, values)
        np.testing.assert_array_equal(input_labels, labels)
        np.testing.assert_array_equal(input_folds, folds)
        np.testing.assert_array_equal(input_groups, groups)
        called.append("fit-fixed-probe")
        return probabilities.copy(), [{"synthetic": True}]

    monkeypatch.setattr(runner, "load_incumbent", incumbent)
    monkeypatch.setattr(runner, "load_embedding_cache", lambda *args, **kwargs: (
        values, {"provenance": {"runtime": {"synthetic": True}}}))
    monkeypatch.setattr(runner, "evaluate_oof", oof)

    def execute(mode, name):
        output = tmp_path / name
        runner.run(mode, plan_path, tmp_path / "cache", tmp_path / "model", output,
                   tmp_path / "development/selected.json")
        return json.loads((output / ("selected.json" if mode == "development" else "metrics.json")).read_text())

    return SimpleNamespace(runner=runner, execute=execute, root=tmp_path, called=called,
                           robust_error_folds=robust_error_folds, plan_path=plan_path,
                           probabilities=probabilities)


def test_runner_selects_only_two_recipes_and_checks_only_winner(runner_fixture, capsys):
    f = runner_fixture
    development = f.execute("development", "development")
    assert set(development["all_results"]) == {"luar", "luar_v4_half"}
    assert development["selected_name"] == "luar" and development["passes_development_gate"]
    diagnostics = json.loads((f.root / "development/diagnostics.json").read_text())
    assert set(diagnostics) == {"luar", "luar_v4_half"}
    assert diagnostics["luar"]["per_class_recall"] == [1.] * 18
    assert np.asarray(diagnostics["luar"]["confusion_matrix"]).shape == (18, 18)
    assert diagnostics["luar"]["new_error_rows"] == 0
    robustness = f.execute("robustness", "robustness")
    assert list(robustness["all_results"]) == ["luar"]
    assert robustness["passes"] and robustness["fold_wins"] == 5
    assert f.called == ["fit-fixed-probe", "fit-fixed-probe"]


def test_robustness_requires_four_fold_wins_even_with_large_pooled_gain(runner_fixture, capsys):
    f = runner_fixture
    assert f.execute("development", "development")["passes_development_gate"]
    f.robust_error_folds.clear()
    f.robust_error_folds.update({0, 1, 2})
    result = f.execute("robustness", "robustness")
    assert result["delta_from_v4"] > .005 and result["fold_wins"] == 3
    assert not result["passes"]


@pytest.mark.parametrize("change", ["selected_result", "probabilities", "fit_audit", "source", "plan", "cache"])
def test_robustness_rejects_changed_development_before_fitting(runner_fixture, change, capsys):
    f = runner_fixture
    f.execute("development", "development")
    folder = f.root / "development"
    if change == "selected_result":
        path = folder / "selected.json"
    elif change == "probabilities":
        path = folder / "luar_oof_probabilities.npy"
    elif change == "fit_audit":
        path = folder / "fit_audit.json"
    elif change == "source":
        path = f.root / "scripts/extract_embeddings_v5.py"
    elif change == "plan":
        path = f.plan_path
    else:
        path = f.root / "cache/train-embeddings.npz"
    with path.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError):
        f.execute("robustness", "robustness")
    assert f.called == ["fit-fixed-probe"]
    assert not (f.root / "robustness").exists()
