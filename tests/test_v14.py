import numpy as np
import pytest
from name_that_llm.qwen_probe_v14 import Qwen8BProbe


@pytest.mark.parametrize('pool', [0, 1])
def test_probe_uses_training_scaler_and_predicts_rows_independently(pool):
    rng = np.random.default_rng(49)
    x = rng.normal(size=(45, 2, 2, 9))
    y = np.arange(45) % 3
    q = rng.normal(size=(3, 2, 2, 9))
    probe = Qwen8BProbe(pool).fit(x, y)
    before = probe.decision_function(q[:1])
    after = probe.decision_function(np.concatenate([q[:1], np.full_like(q[1:], 1e5)]))[:1]
    np.testing.assert_allclose(before, after, atol=1e-12)
    np.testing.assert_allclose(probe.members_[0][0].mean_, x[:, pool, 0].mean(0))


def test_probe_rejects_wrong_pool_shape_and_nonfinite_values():
    probe = Qwen8BProbe()
    with pytest.raises(ValueError):
        probe.fit(np.zeros((4, 3, 2, 9)), np.arange(4))
    with pytest.raises(ValueError):
        probe.fit(np.full((4, 2, 2, 9), np.nan), np.arange(4))
