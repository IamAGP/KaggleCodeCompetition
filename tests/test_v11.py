import numpy as np
import pytest
from name_that_llm.qwen_probe_v11 import QwenKernelProbe
from name_that_llm.qwen_probe_v12 import QwenDiscriminantProbe


@pytest.mark.parametrize('probe', [QwenKernelProbe(1), QwenDiscriminantProbe(.1)])
def test_independent_test_rows_and_training_only_scaler(probe):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(45, 5, 7))
    y = np.arange(45) % 3
    probe.fit(x, y)
    query = rng.normal(size=(4, 5, 7))
    before = probe.decision_function(query[:1])
    unrelated = np.full((3, 5, 7), 1e6)
    after = probe.decision_function(np.concatenate([query[:1], unrelated]))[:1]
    np.testing.assert_allclose(before, after, atol=1e-12)
    np.testing.assert_allclose(probe.members_[0][1].mean_, x[:, 0].mean(0))


def test_precomputed_kernel_matches_native_svc():
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    rng = np.random.default_rng(31)
    x, q = rng.normal(size=(45, 7)), rng.normal(size=(4, 7))
    y = np.arange(45) % 3
    native = OneVsRestClassifier(SVC(C=10, gamma=1/7, class_weight='balanced')).fit(x,y)
    pre = OneVsRestClassifier(SVC(C=10, kernel='precomputed', class_weight='balanced')).fit(rbf_kernel(x,gamma=1/7),y)
    np.testing.assert_allclose(native.decision_function(q),pre.decision_function(rbf_kernel(q,x,gamma=1/7)),atol=1e-10)
