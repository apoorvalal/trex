"""GMM tests against linearmodels and independent derivative/covariance oracles."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from linearmodels.iv import IVGMM
from statsmodels.stats.sandwich_covariance import S_hac_simple
from trex.gmm.gmm import (
    GMMEstimator,
    GMMEstimatorScipy,
    GMMEstimatorTorch,
    moment_covariance,
)

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("backend", ["scipy", "torch"])
@pytest.mark.parametrize("optimal", [True, False])
def test_iv_gmm_reference(backend, optimal):
    rng = np.random.default_rng(40)
    z = rng.normal(size=(500, 3))
    x = z @ np.array([[0.7, 0.2], [0.2, 0.5], [0.3, -0.2]]) + rng.normal(size=(500, 2))
    y = x @ [1.2, -0.6] + rng.normal(size=500) * (1 + abs(z[:, 0]))
    cls = GMMEstimatorTorch if backend == "torch" else GMMEstimatorScipy
    w = np.diag([0.7, 1.2, 2.0])
    spec = "optimal" if optimal else w
    model = GMMEstimator(
        cls.iv_moment, weighting_matrix=spec, backend=backend, device="cpu"
    ).fit(z, y, x)
    reference = IVGMM(y, None, x, z, weight_type="robust").fit(
        iter_limit=2 if optimal else 1,
        initial_weight=np.eye(3) if optimal else w,
        cov_type="robust",
        debiased=False,
    )
    assert_allclose(model.theta_, reference.params, atol=2e-6)
    assert_allclose(model.covariance_, reference.cov, atol=2e-7)
    assert_allclose(model.jacobian_moment_cond(), -z.T @ x / len(x), atol=1e-8)


@pytest.mark.parametrize("backend", ["scipy", "torch"])
def test_nonlinear_one_moment_and_refit(backend):
    # Estimate log E[y]; derivative is -exp(theta), not -z'x/n.
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    z = x = np.ones((len(y), 1))

    def moment(z, y, x, b):
        exp = torch.exp(b[0]) if isinstance(b, torch.Tensor) else np.exp(b[0])
        return (y - exp)[:, None]

    m = GMMEstimator(moment, backend=backend, device="cpu").fit(z, y, x)
    assert_allclose(m.theta_, [np.log(y.mean())], atol=1e-6)
    assert_allclose(m.Gamma_, [[-y.mean()]], atol=1e-5)
    assert_allclose(
        m.covariance_,
        [[np.mean((y - y.mean()) ** 2) / len(y) / y.mean() ** 2]],
        atol=1e-7,
    )
    m.fit(z, y * 2, x)
    assert_allclose(m.theta_, [np.log(2 * y.mean())], atol=1e-6)


def test_hac_matches_statsmodels_and_factory():
    g = np.random.default_rng(9).normal(size=(50, 3))
    assert_allclose(
        moment_covariance(g, 4), S_hac_simple(g, nlags=4) / len(g), atol=1e-15
    )
    assert isinstance(GMMEstimatorTorch(GMMEstimatorTorch.iv_moment), GMMEstimatorTorch)
    with pytest.raises(ValueError, match="fitted"):
        GMMEstimatorScipy(GMMEstimatorScipy.iv_moment).summary()
