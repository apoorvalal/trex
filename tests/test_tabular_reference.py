import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import wasserstein_distance
from trex.simdgp import (
    TabularTransformer,
    distribution_metrics,
    sliced_wasserstein_distance,
)

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("n,m", [(2, 5), (20, 300), (30, 30)])
def test_sliced_w1_unequal_samples_scipy(n, m):
    rng = np.random.default_rng(5)
    a = rng.normal(size=(n, 3))
    b = rng.normal(size=(m, 3)) + 2
    directions = np.random.default_rng(11).normal(size=(17, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    reference = np.mean([wasserstein_distance(a @ v, b @ v) for v in directions])
    assert_allclose(
        sliced_wasserstein_distance(a, b, seed=11, n_projections=17),
        reference,
        atol=1e-14,
    )
    assert_allclose(
        sliced_wasserstein_distance(b, a, seed=11, n_projections=17),
        reference,
        atol=1e-14,
    )


def test_transform_rejects_swapped_columns_and_invalid_values():
    d = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0, 1, 0]})
    t = TabularTransformer(binary_columns=["b"]).fit(d)
    assert_allclose(t.inverse_transform(t.transform(d)), d, atol=1e-6)
    with pytest.raises(ValueError, match="order"):
        t.transform(d[["b", "a"]])
    with pytest.raises(ValueError):
        t.inverse_transform(np.ones((3, 1)))
    with pytest.raises(ValueError):
        TabularTransformer(binary_columns=[5]).fit(d)
    with pytest.raises(ValueError):
        TabularTransformer(binary_columns=["a"]).fit(d)
    with pytest.raises(ValueError):
        distribution_metrics(np.empty((0, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        sliced_wasserstein_distance(np.ones((2, 1)), np.ones((3, 1)), n_projections=0)
