import json
from pathlib import Path
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from trex.gmm.gel import GELEstimator, rho_el, rho_exponential, rho_cue

DATA = Path(__file__).parent / "data"
pytestmark = pytest.mark.reference


@pytest.mark.parametrize("rho", [rho_el, rho_exponential, rho_cue])
def test_gel_mean_covariance_closed_form(rho):
    y = np.linspace(-1, 3, 51)[:, None]
    model = GELEstimator(lambda d, b: d - b[0], rho=rho)
    model.fit(y, np.array([1.0]))
    assert_allclose(model.est, [1.0], atol=1e-6)
    assert_allclose(model.Sigma, [[np.var(y) / len(y)]], atol=1e-6)
    assert model.J_stat is None


@pytest.mark.parametrize(
    "name,rho", [("EL", rho_el), ("ET", rho_exponential), ("CUE", rho_cue)]
)
def test_gel_overidentified_matches_r(name, rho):
    y = np.loadtxt(DATA / "r_gel_input.csv", delimiter=",", skiprows=1)
    ref = json.loads((DATA / "r_gel_reference.json").read_text())[name]

    def moments(d, b):
        e = d - b[0]
        return np.column_stack([e, e**2 - b[1], e**3])

    m = GELEstimator(moments, rho=rho)
    m.fit(y, np.array([y.mean(), y.var()]))
    assert_allclose(m.est, ref["coef"], atol=1e-4)
    assert m.Sigma.shape == (2, 2)
    assert np.linalg.eigvalsh(m.Sigma).min() > 0
    assert m.J_stat >= 0 and 0 <= m.J_pvalue <= 1
    # Correct-specification first-order covariance, independent analytic G.
    e = y - m.est[0]
    g = moments(y, m.est)
    G = np.array([[-1, 0], [-2 * e.mean(), -1], [-3 * np.mean(e**2), 0]])
    cov = np.linalg.inv(G.T @ np.linalg.solve(g.T @ g / len(y), G)) / len(y)
    assert_allclose(m.Sigma, cov, atol=1e-8)
