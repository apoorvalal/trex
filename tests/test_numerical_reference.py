"""Reference checks for the remaining numerical primitives and limiting cases."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.spatial.distance import cdist
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from trex.score_matching import _pava_decreasing
from trex.cmr import rbf_kernel, hsic, mmr_loss, SieveMinimumDistance
from trex.grouped_fe import chunked_knn_indices
from trex.panel import NuclearNormMatrixCompletion
from trex import LatentFactorGLM
import statsmodels.api as sm

pytestmark = pytest.mark.reference


def test_pava_against_sklearn():
    rng = np.random.default_rng(91)
    y = rng.normal(size=90)
    w = np.exp(rng.normal(size=90))
    expected = IsotonicRegression(increasing=False).fit_transform(
        np.arange(len(y)), y, sample_weight=w
    )
    assert_allclose(_pava_decreasing(y, w), expected, atol=1e-13)


def test_kernel_moments_against_scipy_numpy():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(25, 3))
    y = rng.normal(size=(25, 2))
    k = np.exp(-cdist(x, x, "sqeuclidean") / (2 * 0.7**2))
    l = np.exp(-cdist(y, y, "sqeuclidean") / (2 * 1.2**2))
    xt, yt = torch.tensor(x), torch.tensor(y)
    assert_allclose(rbf_kernel(xt, bandwidth=0.7)[0], k, atol=1e-13)
    assert_allclose(
        mmr_loss(yt, xt, bandwidth=0.7), np.trace(y.T @ k @ y) / 25**2, atol=1e-13
    )
    h = np.eye(25) - np.ones((25, 25)) / 25
    assert_allclose(
        hsic(xt, yt, bandwidth_x=0.7, bandwidth_y=1.2),
        np.trace(h @ k @ h @ l) / 24**2,
        atol=1e-13,
    )


def test_knn_against_sklearn_without_ties():
    x = np.random.default_rng(8).normal(size=(30, 4))
    expected = NearestNeighbors(n_neighbors=4).fit(x).kneighbors(return_distance=False)
    actual, distances = chunked_knn_indices(torch.tensor(x), 4, block_size=7)
    assert_allclose(actual, expected)


def test_matrix_completion_full_mask_matches_numpy_svt():
    y = np.random.default_rng(13).normal(size=(12, 9))
    lam = 0.018
    u, s, v = np.linalg.svd(y, full_matrices=False)
    expected = (u * np.maximum(s - lam * y.size / 2, 0)) @ v
    m = NuclearNormMatrixCompletion(
        lambda_L=lam, fit_unit_effects=False, fit_time_effects=False, device="cpu"
    ).fit(torch.tensor(y))
    assert_allclose(m.predict(), expected, atol=1e-12)


@pytest.mark.parametrize("family", ["gaussian", "bernoulli", "poisson"])
def test_rank_zero_latent_glm_matches_statsmodels(family):
    rng = np.random.default_rng(47)
    x = np.column_stack([np.ones(300), rng.normal(size=(300, 2))])
    index = x @ np.array([0.2, 0.4, -0.3])
    if family == "gaussian":
        y = index + rng.normal(size=300)
        link = sm.families.Gaussian()
    elif family == "bernoulli":
        y = rng.binomial(1, 1 / (1 + np.exp(-index)))
        link = sm.families.Binomial()
    else:
        y = rng.poisson(np.exp(index))
        link = sm.families.Poisson()
    reference = sm.GLM(y, x, family=link).fit(tol=1e-12)
    m = LatentFactorGLM(
        family=family,
        rank=0,
        penalty=0,
        beta_penalty=0,
        optimizer=torch.optim.LBFGS,
        optimizer_kwargs={"line_search_fn": "strong_wolfe"},
        maxiter=150,
        tol=1e-10,
        device="cpu",
    )
    m.fit(
        torch.tensor(x),
        torch.tensor(y, dtype=torch.float64),
        unit_ids=torch.arange(30).repeat_interleave(10),
        time_ids=torch.arange(10).repeat(30),
    )
    assert_allclose(m.params["coef"], reference.params, atol=5e-5)


def test_panel_and_embedding_input_guards():
    from trex.grouped_fe import build_panel_embeddings

    with pytest.raises(ValueError, match="Observed"):
        NuclearNormMatrixCompletion(device="cpu").fit(
            torch.tensor([[1.0, float("nan")]]),
            mask=torch.ones((1, 2), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="observed cells"):
        build_panel_embeddings(
            torch.ones((3, 1)),
            torch.ones(3),
            torch.tensor([0, 0, 1]),
            torch.tensor([0, 1, 0]),
        )
    y = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    m = NuclearNormMatrixCompletion(
        lambda_L=1, fit_unit_effects=False, device="cpu"
    ).fit(y)
    assert torch.all(m.result_.unit_effects == 0)
    with pytest.raises(ValueError):
        hsic(torch.ones((1, 1)), torch.ones((1, 1)))


def test_linear_sieve_matches_linearmodels_2sls():
    from linearmodels.iv import IV2SLS

    rng = np.random.default_rng(25)
    z = rng.normal(size=(400, 2))
    noise = rng.normal(size=400)
    x = z @ np.array([0.8, 0.4]) + 0.5 * noise
    y = 1.4 + 2.1 * x + noise
    oracle = IV2SLS(y, np.ones((400, 1)), x, z).fit()
    regression = torch.nn.Linear(1, 1).double()
    m = SieveMinimumDistance(
        model=regression,
        moment_function=lambda pred, target: pred - target,
        degree=1,
        ridge=0,
        maxiter=100,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
        dtype=torch.float64,
        device="cpu",
    )
    m.fit(torch.tensor(x[:, None]), torch.tensor(y[:, None]), torch.tensor(z))
    actual = np.array([regression.bias.item(), regression.weight.item()])
    assert_allclose(actual, oracle.params, atol=1e-5)


def test_glm_rejects_invalid_response():
    from trex import LogisticRegression, PoissonRegression

    for cls, y in [
        (LogisticRegression, torch.tensor([0.0, 2.0])),
        (PoissonRegression, torch.tensor([0.0, -1.0])),
    ]:
        with pytest.raises(ValueError):
            cls(device="cpu").fit(torch.ones((2, 1)), y)
