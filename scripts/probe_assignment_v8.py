"""Diagnostic probe: does exploiting the one-response-per-model-per-prompt structure help v7?

Uses frozen v7 development OOF scores only; nothing is refit. Prints every step.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from name_that_llm.baseline import LABELS
from name_that_llm.classifier_v7 import RECIPES, combine_evidence
from run_embeddings_v5 import load_incumbent, read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("v8probe")
OUT = Path("artifacts/improvement-v8/probe")


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sinkhorn(p, iters=100):
    p = p.copy()
    target = np.full(p.shape[1], len(p) / p.shape[1])
    for _ in range(iters):
        p *= target / p.sum(0)
        p /= p.sum(1, keepdims=True)
    return p


def hungarian(p, groups):
    pred = p.argmax(1)
    for g in np.unique(groups):
        rows = np.where(groups == g)[0]
        if len(rows) > len(LABELS):
            continue
        _, cols = linear_sum_assignment(-np.log(np.clip(p[rows], 1e-12, 1)))
        pred[rows] = cols
    return pred


def cluster_prompts(texts, n_clusters, cap=None):
    x = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2).fit_transform(texts).toarray()
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average").fit_predict(x)
    if cap is None:
        return labels
    # split any cluster larger than one full prompt (18 responses) until every piece fits
    queue, nxt = list(np.unique(labels)), labels.max() + 1
    while queue:
        rows = np.where(labels == queue.pop())[0]
        if len(rows) <= cap:
            continue
        parts = AgglomerativeClustering(n_clusters=int(np.ceil(len(rows) / cap)), metric="cosine",
                                        linkage="average").fit_predict(x[rows])
        for k in range(1, parts.max() + 1):
            labels[rows[parts == k]] = nxt
            queue.append(nxt)
            nxt += 1
        queue.append(labels[rows[parts == 0][0]])
    return labels


def main():
    t0 = time.time()
    plan = read_json(Path("configs/name-that-llm/v7.json"))
    train, folds, groups, incumbent, baseline, _, _ = load_incumbent(plan, "development")
    sd = np.load("artifacts/improvement-v7/development/embedding_oof_evidence.npy")
    scores = combine_evidence(sd, incumbent, RECIPES["styledistance_v4_half"])
    y = train.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    log.info("rows=%d prompts=%d v4 baseline=%.6f", len(y), train.prompt_id.nunique(), baseline["balanced_accuracy"])

    # evidence is row-normalized log-prob-like; temperature matters for Sinkhorn/Hungarian, so scan it
    ba = lambda pred: balanced_accuracy_score(y, pred)
    res = {"v7_argmax": ba(scores.argmax(1))}
    log.info("v7 argmax reproduced=%.6f (recorded 0.404509)", res["v7_argmax"])
    true_groups = train.prompt_id.to_numpy()
    for t in (0.5, 1.0, 2.0, 4.0):
        p = softmax(scores * t)
        res[f"sinkhorn_T{t}"] = ba(sinkhorn(p).argmax(1))
        res[f"hungarian_oracle_T{t}"] = ba(hungarian(p, true_groups))
        res[f"sinkhorn+hungarian_oracle_T{t}"] = ba(hungarian(sinkhorn(p), true_groups))
        log.info("T=%.1f sinkhorn=%.4f hung_oracle=%.4f sk+hung_oracle=%.4f", t, res[f"sinkhorn_T{t}"],
                 res[f"hungarian_oracle_T{t}"], res[f"sinkhorn+hungarian_oracle_T{t}"])

    # realistic: test has no prompt_id, so recover prompts from text alone
    clusters = cluster_prompts(train.text.tolist(), train.prompt_id.nunique())
    sizes = np.bincount(clusters)
    res["cluster_ari"] = adjusted_rand_score(true_groups, clusters)
    log.info("text clusters ARI vs prompt_id=%.4f sizes min/med/max=%d/%d/%d", res["cluster_ari"],
             sizes.min(), np.median(sizes), sizes.max())
    capped = cluster_prompts(train.text.tolist(), round(len(train) / len(LABELS)), cap=len(LABELS))
    csz = np.bincount(capped)
    res["capped_cluster_ari"] = adjusted_rand_score(true_groups, capped)
    log.info("capped clusters (n=round(N/18)) ARI=%.4f count=%d sizes min/med/max=%d/%d/%d",
             res["capped_cluster_ari"], len(csz), csz.min(), np.median(csz), csz.max())
    for t in (1.0, 2.0):
        p = softmax(scores * t)
        res[f"hungarian_capped_T{t}"] = ba(hungarian(p, capped))
        log.info("T=%.1f hung_capped=%.4f", t, res[f"hungarian_capped_T{t}"])
        res[f"hungarian_clustered_T{t}"] = ba(hungarian(p, clusters))
        res[f"sinkhorn+hungarian_clustered_T{t}"] = ba(hungarian(sinkhorn(p), clusters))
        log.info("T=%.1f hung_clustered=%.4f sk+hung_clustered=%.4f", t, res[f"hungarian_clustered_T{t}"],
                 res[f"sinkhorn+hungarian_clustered_T{t}"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(res, indent=2) + "\n")
    log.info("done in %.1fs -> %s", time.time() - t0, OUT / "probe.json")


if __name__ == "__main__":
    main()
