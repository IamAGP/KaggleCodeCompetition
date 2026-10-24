"""Frozen Qwen feature probes; each prediction is independent of other test rows."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LAYERS = (9, 18, 24, 30, 36)
V10_MEMBERS = ((24, .02), (9, .02), (24, .2))


def row_z(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('Scores must be a finite matrix')
    return (scores - scores.mean(1, keepdims=True)) / np.maximum(scores.std(1, keepdims=True), 1e-8)


class QwenKernelProbe:
    def __init__(self, C=10.):
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[1] != len(LAYERS) or not np.isfinite(X).all():
            raise ValueError('Expected finite rows x five layers x hidden dimension')
        self.classes_ = np.unique(y)
        self.members_ = []
        for layer in (9, 24):
            scaler = StandardScaler().fit(X[:, LAYERS.index(layer)])
            reference = scaler.transform(X[:, LAYERS.index(layer)]).astype(np.float64)
            gamma = 1. / reference.shape[1]
            gram = rbf_kernel(reference, gamma=gamma)
            model = OneVsRestClassifier(SVC(C=self.C, kernel='precomputed',
                class_weight='balanced', probability=False, tol=.001, cache_size=256), n_jobs=1).fit(gram, y)
            if not np.array_equal(model.classes_, self.classes_):
                raise ValueError('Class alignment changed')
            self.members_.append((layer, scaler, reference, gamma, model))
        return self

    def decision_function(self, X):
        result = []
        for layer, scaler, reference, gamma, model in self.members_:
            values = scaler.transform(np.asarray(X)[:, LAYERS.index(layer)]).astype(np.float64)
            result.append(row_z(model.decision_function(rbf_kernel(values, reference, gamma=gamma))))
        return row_z(np.mean(result, axis=0))


class FrozenV10Probe:
    """Historical three selected members, without repeating their selection grid."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.members_ = []
        for layer, C in V10_MEMBERS:
            scaler = StandardScaler().fit(X[:, LAYERS.index(layer)])
            model = LogisticRegression(C=C, max_iter=500).fit(scaler.transform(X[:, LAYERS.index(layer)]), y)
            self.members_.append((layer, scaler, model))
        return self

    def decision_function(self, X):
        return np.mean([np.log(np.clip(model.predict_proba(scaler.transform(X[:, LAYERS.index(layer)])), 1e-12, 1))
                        for layer, scaler, model in self.members_], axis=0)
