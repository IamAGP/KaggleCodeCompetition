"""Frozen exploratory pilot: external-source confounding, with host prompts held out."""
import hashlib
import json
import logging
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().lower()


class PoolPilot:
    def __init__(self):
        self.out = Path("artifacts/improvement-v17")
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "metrics.json").exists():
            raise RuntimeError("Completed experiment exists; create a new experiment instead.")
        self.plan = json.loads(Path("configs/name-that-llm/v17.json").read_text())
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                            handlers=[logging.StreamHandler(), logging.FileHandler(self.out / "run.log")])

    def run(self):
        host_path = Path("data/name-that-llm/train.csv")
        pool_path = Path("data/name-that-llm/external/pool_v4.parquet")
        host = pd.read_csv(host_path, dtype={"id": str, "prompt_id": str})
        pool = pd.read_parquet(pool_path)
        folds = pd.read_csv(self.plan["fold_file"], dtype={"id": str, "prompt_id": str})
        assert folds.id.equals(host.id) and folds.prompt_id.equals(host.prompt_id)
        labels = np.sort(host.model.unique())
        assert set(pool.model) <= set(labels)
        frozen = dict(self.plan, input_sha256={str(p): digest(p) for p in
                      [host_path, pool_path, Path(self.plan["fold_file"]), Path(__file__)]})
        (self.out / "frozen-plan.json").write_text(json.dumps(frozen, indent=2))
        pool["normalized"] = pool.text.map(normalized)
        overlap = pool.normalized.isin(set(host.text.map(normalized)))
        conflicts = pool.groupby("normalized").model.nunique()
        ambiguous = set(conflicts[conflicts > 1].index)
        rejected = overlap | pool.normalized.isin(ambiguous)
        audit = {"rows": len(pool), "host_normalized_overlap": int(overlap.sum()),
                 "conflicting_texts": len(ambiguous), "duplicate_rows": int(pool.normalized.duplicated().sum()),
                 "sources": pool.groupby("source").agg(rows=("text", "size"), labels=("model", "nunique")).to_dict("index"),
                 "class_counts": pool.model.value_counts().to_dict()}
        pool = pool.loc[~rejected].drop_duplicates("normalized")
        pool = pool.groupby("model", group_keys=False).apply(
            lambda x: x.sample(min(len(x), self.plan["external_cap_per_class"]), random_state=self.plan["seed"]),
            include_groups=True).reset_index(drop=True)
        audit["selected_rows"] = len(pool)
        (self.out / "pool-audit.json").write_text(json.dumps(audit, indent=2))
        pool[["id", "source", "model"]].to_csv(self.out / "selected-pool-ids.csv", index=False)
        valid = folds.fold.to_numpy() == self.plan["pilot_fold"]
        train = host.loc[~valid].copy()
        test = host.loc[valid].copy()
        assert set(train.prompt_id).isdisjoint(test.prompt_id)
        all_train = pd.concat([train, pool], ignore_index=True)
        logging.info("Pilot host train=%s validation=%s external=%s", len(train), len(test), len(pool))
        results = {}
        for candidate in self.plan["candidates"]:
            started = time.monotonic()
            fitting = train if candidate == "host_only" else all_train
            # Host-only vocabulary is fitted separately; no validation text enters a vocabulary or IDF.
            if candidate in ("host_only", "uniform_pool"):
                blocks, held = [], []
                for analyzer, ngram, cap in [("char", (3, 5), 50000), ("word", (1, 2), 30000)]:
                    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=cap,
                                                min_df=2, lowercase=False, sublinear_tf=True, dtype=np.float32)
                    blocks.append(vectorizer.fit_transform(fitting.text))
                    held.append(vectorizer.transform(test.text))
                X, V = hstack(blocks).tocsr(), hstack(held).tocsr()
                logging.info("%s feature matrix=%s nnz=%s", candidate, X.shape, X.nnz)
            weights = np.ones(len(fitting), dtype=np.float64)
            if candidate != "host_only":
                external = np.ones(len(pool))
                if candidate in ("source_balanced", "source_and_length_balanced"):
                    counts = pool.groupby(["model", "source"]).text.transform("size").to_numpy()
                    external = 1.0 / np.sqrt(counts)
                if candidate == "source_and_length_balanced":
                    # Class-conditional length matching learned exclusively from the fitting host fold.
                    train_lengths = train.text.str.split().str.len()
                    pool_lengths = pool.text.str.split().str.len().to_numpy()
                    for label in labels:
                        lengths = train_lengths[train.model == label].to_numpy()
                        median = np.median(np.log1p(lengths))
                        scale = max(float(np.std(np.log1p(lengths))), 0.2)
                        mask = pool.model.to_numpy() == label
                        external[mask] *= np.clip(np.exp(-0.5 * ((np.log1p(pool_lengths[mask]) - median) / scale)**2), 0.1, 1.0)
                # Same total external weight per class across candidates isolates redistribution.
                for label in labels:
                    mask = pool.model.to_numpy() == label
                    external[mask] /= external[mask].mean()
                weights[len(train):] = external * self.plan["external_weight"]
            model = LinearSVC(C=self.plan["C"], class_weight="balanced", max_iter=5000,
                              random_state=self.plan["seed"], dual="auto")
            model.fit(X, fitting.model, sample_weight=weights)
            scores = model.decision_function(V)
            assert np.array_equal(model.classes_, labels) and np.isfinite(scores).all()
            pred = labels[scores.argmax(axis=1)]
            np.savez_compressed(self.out / f"{candidate}_pilot.npz", ids=test.id.to_numpy(dtype=str),
                                labels=labels.astype(str), scores=scores)
            results[candidate] = {"accuracy": accuracy_score(test.model, pred),
                                  "balanced_accuracy": balanced_accuracy_score(test.model, pred),
                                  "seconds": time.monotonic() - started, "iterations": int(model.n_iter_)}
            pd.DataFrame(confusion_matrix(test.model, pred, labels=labels), index=labels, columns=labels).to_csv(
                self.out / f"{candidate}_confusion.csv")
            (self.out / "partial-metrics.json").write_text(json.dumps(results, indent=2))
            logging.info("%s %s", candidate, results[candidate])
        (self.out / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    PoolPilot().run()
