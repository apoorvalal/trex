"""Independent checks of the scalar polynomial CUE prototype."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

from trex.gmm import GELEstimator, ScalarIVCUE, rho_cue


def data(seed=9, k=4, n=400):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    v, e = rng.normal(size=(2, n))
    x = 0.3 * z[:, 0] + v
    y = 2 * x + 0.6 * v + e * (0.2 + abs(z[:, 0]))
    return z, y, x


def test_just_identified_and_remote_root():
    for beta in [2.3, 1e10, -1e10]:
        problem = ScalarIVCUE([beta], [1], np.eye(2))
        fit = problem.minimize_polynomial()
        assert_allclose(fit.beta, beta, rtol=1e-6)
        assert fit.fun < 1e-10
        assert not fit.at_infinity and not fit.certified


def test_infinity_and_constant():
    fit = ScalarIVCUE([1], [0], np.eye(2)).minimize_polynomial()
    assert fit.at_infinity and fit.fun == 0
    fit = ScalarIVCUE([1, 0], [0, 1], np.eye(4)).minimize_polynomial()
    assert fit.constant and np.isfinite(fit.beta)
    assert_allclose(fit.fun, 1)
    fit = ScalarIVCUE([0], [0], np.eye(2)).minimize_polynomial()
    assert fit.constant and fit.fun == 0


@pytest.mark.parametrize("k", [1, 2, 4, 10])
def test_homoskedastic_generalized_eigen_oracle(k):
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=(2, k))
    H = rng.normal(size=(k, k))
    H = H @ H.T + np.eye(k)
    error_cov = np.array([[2.0, 0.6], [0.6, 1.0]])
    problem = ScalarIVCUE(a, b, np.kron(error_cov, H))
    means = np.column_stack([a, -b])
    sign = np.diag([1, -1])
    expected, vectors = eigh(
        means.T @ np.linalg.solve(H, means), sign @ error_cov @ sign
    )
    result = problem.minimize_polynomial()
    assert_allclose(result.fun, expected[0], atol=1e-8)
    assert_allclose(result.beta, vectors[1, 0] / vectors[0, 0], rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_arbitrary_pd_against_angular_grid_and_local_refinement(k):
    rng = np.random.default_rng(321 + k)
    a, b = rng.normal(size=(2, k))
    L = rng.normal(size=(2 * k, 2 * k))
    sigma = L @ L.T + np.eye(2 * k)
    problem = ScalarIVCUE(a, b, sigma)

    # Independent compactified direct matrix objective, no interpolation.
    def angle_value(angle):
        c, s = np.cos(angle), np.sin(angle)
        g = c * a - s * b
        omega = (
            c * c * sigma[:k, :k]
            - c * s * (sigma[:k, k:] + sigma[k:, :k])
            + s * s * sigma[k:, k:]
        )
        return g @ np.linalg.solve(omega, g)

    grid = np.linspace(-np.pi / 2, np.pi / 2, 2001)
    values = np.array([angle_value(t) for t in grid])
    j = np.argmin(values)
    reference = minimize_scalar(
        angle_value,
        bounds=(grid[max(0, j - 1)], grid[min(len(grid) - 1, j + 1)]),
        method="bounded",
        options={"xatol": 1e-14},
    )
    result = problem.minimize_polynomial()
    assert_allclose(result.fun, reference.fun, atol=1e-8)
    # Analytical derivatives against central differences in original beta units.
    for beta in [-20, -0.3, 2.0, 30]:
        q, d, dd = problem.value_derivatives(beta)
        step = 1e-4
        assert_allclose(
            d,
            (problem.value(beta + step) - problem.value(beta - step)) / (2 * step),
            atol=1e-8,
        )
        assert_allclose(
            dd,
            (
                problem.value_derivatives(beta + step)[1]
                - problem.value_derivatives(beta - step)[1]
            )
            / (2 * step),
            atol=1e-7,
        )


@pytest.mark.parametrize("kind", ["hc0", "hac", "cluster"])
def test_covariance_builders_against_direct_scores(kind):
    z, y, x = data()
    Y = np.column_stack([y - y.mean(), x - x.mean()])
    z = z - z.mean(0)
    V = Y - z @ np.linalg.lstsq(z, Y, rcond=None)[0]
    scores = np.column_stack([z * V[:, 0, None], z * V[:, 1, None]])
    kwargs = {}
    expected = scores.T @ scores / len(z)
    if kind == "hac":
        kwargs["max_lags"] = 3
        for lag in range(1, 4):
            cross = scores[lag:].T @ scores[:-lag] / len(z)
            expected += (1 - lag / 4) * (cross + cross.T)
    if kind == "cluster":
        kwargs["clusters"] = np.repeat(np.arange(len(z) // 5), 5)
        sums = scores.reshape(-1, 5, 8).sum(1)
        expected = sums.T @ sums / len(z)
    problem = ScalarIVCUE.from_data(z, y, x, covariance=kind, **kwargs)
    assert_allclose(problem.sigma, expected, atol=2e-14)
    beta = 1.7
    g = z.T @ (y - beta * x) / np.sqrt(len(z))
    R = np.column_stack([np.eye(4), -beta * np.eye(4)])
    assert_allclose(
        problem.value(beta), g @ np.linalg.solve(R @ expected @ R.T, g), rtol=1e-12
    )


def test_raw_criterion_is_existing_trex_gel_profile():
    z, y, x = data(k=3)
    D = np.column_stack([y, x, z])
    gel = GELEstimator(
        lambda d, b: d[:, 2:] * (d[:, 0] - b[0] * d[:, 1])[:, None], rho=rho_cue
    )
    problem = ScalarIVCUE.from_data(z, y, x, weighting="raw", demean=False)
    for beta in [-3.0, 0.0, 2.0]:
        value, gradient = gel._profile_value_gradient(np.array([beta]), D, np.zeros(3))
        q, dq, _ = problem.value_derivatives(beta)
        assert_allclose(value * 2 * len(D), q, rtol=1e-12)
        assert_allclose(gradient[0] * 2 * len(D), dq, rtol=1e-7)


def test_invariance_to_instrument_basis_and_parameter_units():
    z, y, x = data(k=4)
    result = ScalarIVCUE.from_data(z, y, x).minimize_polynomial()
    matrix = np.array([[1, 2, 0, 0], [0, 0.1, 0, 0], [0, 0, 20, 1], [0, 0, 0, 2]])
    changed = ScalarIVCUE.from_data(
        z @ matrix, 1000 * y + 70 * x, x
    ).minimize_polynomial()
    assert_allclose(changed.beta, 1000 * result.beta + 70, rtol=1e-6)
    assert_allclose(changed.fun, result.fun, rtol=1e-8)


def test_validation_no_silent_regularization():
    with pytest.raises(ValueError, match="positive definite"):
        ScalarIVCUE([1], [1], np.ones((2, 2)))
    with pytest.raises(ValueError, match="Expected finite"):
        ScalarIVCUE.from_data(*data()[:2], np.ones((400, 2)))
    with pytest.raises(ValueError, match="full column rank"):
        z, y, x = data()
        ScalarIVCUE.from_data(np.column_stack([z, z]), y, x)
    with pytest.raises(ValueError, match="at most 12"):
        ScalarIVCUE(np.ones(13), np.ones(13), np.eye(26)).minimize_polynomial()
    with pytest.raises(ValueError, match="cluster label"):
        ScalarIVCUE.from_data(*data(), covariance="cluster")
