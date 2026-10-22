"""Transductive prompt recovery and one-model-per-prompt assignment (v8).

Training data holds 150 prompts, each answered once by every one of the 18 models.
V8 assumes the hidden test set shares that structure: it recovers prompt groups from
passage text alone and assigns models within each group by minimum-cost matching.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

GROUP_SIZE = 18
ASSIGNMENT_PARAMETERS = {"tfidf": {"sublinear_tf": True, "stop_words": "english", "min_df": 2},
                         "linkage": "average", "metric": "cosine", "cap": GROUP_SIZE,
                         "n_clusters": "round(rows / 18)", "cost": "-log softmax(evidence)"}


def _cluster(distances, n_clusters):
    if n_clusters < 2:
        return np.zeros(len(distances), dtype=int)
    return AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed",
                                   linkage="average").fit_predict(distances)


def recover_prompts(texts, cap=GROUP_SIZE):
    """Cluster passages by topic, then split any cluster larger than one full prompt."""
    texts = list(texts)
    n_clusters = round(len(texts) / cap)
    if n_clusters < 2:
        return None
    x = TfidfVectorizer(**ASSIGNMENT_PARAMETERS["tfidf"]).fit_transform(texts)
    distances = np.clip(1.0 - (x @ x.T).toarray(), 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    labels = _cluster(distances, n_clusters)
    queue, nxt = list(np.unique(labels)), labels.max() + 1
    while queue:
        rows = np.where(labels == queue.pop())[0]
        if len(rows) <= cap:
            continue
        parts = _cluster(distances[np.ix_(rows, rows)], int(np.ceil(len(rows) / cap)))
        for k in range(1, parts.max() + 1):
            labels[rows[parts == k]] = nxt
            queue.append(nxt)
            nxt += 1
        queue.append(labels[rows[parts == 0][0]])
    return labels


def assign_within_prompts(scores, groups):
    """Return class indices; within each recovered group no class is used twice."""
    scores = np.asarray(scores, dtype=float)
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    prediction = log_p.argmax(axis=1)
    if groups is None:
        return prediction
    for group in np.unique(groups):
        rows = np.where(groups == group)[0]
        if len(rows) > scores.shape[1]:
            raise ValueError("A recovered prompt group exceeds the number of models")
        _, cols = linear_sum_assignment(-log_p[rows])
        prediction[rows] = cols
    return prediction
