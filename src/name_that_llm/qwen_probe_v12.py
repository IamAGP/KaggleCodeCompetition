"""Fixed shared-covariance discriminant probes on frozen Qwen layers."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from name_that_llm.qwen_probe_v11 import LAYERS, row_z


class QwenDiscriminantProbe:
    def __init__(self, shrinkage=.1):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.members_ = []
        for layer in (9, 24):
            scaler = StandardScaler().fit(X[:, LAYERS.index(layer)])
            model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=self.shrinkage,
                        priors=np.full(len(self.classes_), 1/len(self.classes_)))
            model.fit(scaler.transform(X[:, LAYERS.index(layer)]).astype(np.float64), y)
            self.members_.append((layer, scaler, model))
        return self

    def decision_function(self, X):
        return row_z(np.mean([row_z(model.decision_function(scaler.transform(X[:, LAYERS.index(layer)])))
                             for layer, scaler, model in self.members_], axis=0))
