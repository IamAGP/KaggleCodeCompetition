"""Fixed two-layer linear probe for frozen Qwen3-8B features."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Qwen8BProbe:
    def __init__(self, pooling=0):
        self.pooling = pooling

    def _values(self, X):
        X = np.asarray(X)
        if X.ndim != 4 or X.shape[1:3] != (2, 2) or not np.isfinite(X).all():
            raise ValueError('Expected finite rows x two pools x two layers x hidden dimension')
        return X[:, self.pooling]

    def fit(self, X, y):
        X = self._values(X)
        self.classes_ = np.unique(y)
        self.members_ = []
        for layer in range(2):
            scaler = StandardScaler().fit(X[:, layer])
            model = LogisticRegression(C=.02, max_iter=1000).fit(scaler.transform(X[:, layer]), y)
            assert np.array_equal(model.classes_, self.classes_)
            self.members_.append((scaler, model))
        return self

    def decision_function(self, X):
        X = self._values(X)
        return np.mean([np.log(np.clip(model.predict_proba(scaler.transform(X[:, j])),1e-12,1))
                        for j,(scaler,model) in enumerate(self.members_)],axis=0)
