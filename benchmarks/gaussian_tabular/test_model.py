import numpy as np
import pytest
import torch
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from model import GaussianSplats


def test_density_matches_independent_scipy_and_normalizes():
    x = np.random.default_rng(2).normal(size=(100, 2))
    model = GaussianSplats().initialize(x).split()
    state = model.snapshot()
    reference = logsumexp([np.log(w) + multivariate_normal.logpdf(x, mean=m, cov=c)
                           for w, m, c in zip(state['weights'], state['means'], state['covariances'])], axis=0)
    np.testing.assert_allclose(model.log_prob(x).detach(), reference, atol=1e-10)
    assert np.allclose(state['weights'].sum(), 1)


def test_split_preserves_moments_and_variance_floor():
    x = np.random.default_rng(7).normal(size=(100, 3))
    model = GaussianSplats().initialize(x)
    mean, cov = model.means[0].detach().clone(), model.covariances[0].detach().clone()
    model.split().split()
    difference = model.means - mean
    marginal = (model.weights[:, None, None] * (model.covariances + difference[:, :, None] * difference[:, None, :])).sum(0)
    torch.testing.assert_close((model.weights[:, None] * model.means).sum(0), mean)
    torch.testing.assert_close(marginal, cov)
    assert torch.linalg.eigvalsh(model.covariances).min() >= model.variance_floor


def test_sampling_matches_mixture_moments_and_is_reproducible():
    x = np.random.default_rng(3).normal(size=(100, 2))
    model = GaussianSplats().initialize(x).split()
    a = model.sample(40000, seed=99).numpy()
    np.testing.assert_allclose(a.mean(0), x.mean(0), atol=.025)
    np.testing.assert_allclose(np.cov(a, rowvar=False), np.cov(x, rowvar=False, ddof=0) + .0025*np.eye(2), atol=.04)
    torch.testing.assert_close(model.sample(10, seed=1), model.sample(10, seed=1))
    assert model.sample(0).shape == (0, 2)


def test_covariance_gradients_against_finite_differences():
    model = GaussianSplats().initialize(np.array([[0., 1.], [2., 0.], [-1., -1.]]))
    x = np.array([[.3, .7], [.5, .9]])
    loss = model.log_prob(x).sum()
    loss.backward()
    analytic = model.raw_chol.grad[0, 1, 0].item()
    eps = 1e-5
    with torch.no_grad():
        model.raw_chol[0, 1, 0] += eps
        plus = model.log_prob(x).sum().item()
        model.raw_chol[0, 1, 0] -= 2*eps
        minus = model.log_prob(x).sum().item()
        model.raw_chol[0, 1, 0] += eps
    assert abs(analytic - (plus-minus)/(2*eps)) < 1e-7


def test_refinement_reduces_loss_on_bimodal_data():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(-2., .2, (100, 1)), rng.normal(2., .2, (100, 1))]
    model = GaussianSplats().initialize(x).split()
    before = float(-model.log_prob(x).mean().detach())
    model.refine(x, steps=200, lr=.05)
    after = float(-model.log_prob(x).mean().detach())
    assert after < before - .5


def test_rejects_nonfinite_and_wrong_dimensions():
    with pytest.raises(ValueError): GaussianSplats().initialize([[float('nan')], [0.]])
    with pytest.raises(ValueError): GaussianSplats(variance_floor=0)
    model = GaussianSplats().initialize(np.eye(2))
    with pytest.raises(ValueError): model.log_prob(np.ones((3, 3)))
