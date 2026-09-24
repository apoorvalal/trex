"""GLM and discrete-choice inference checked outside Trex."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy.special import log_ndtr
from trex import LogisticRegression, PoissonRegression
from trex.choice import BinaryProbit, MultinomialLogit

DATA = Path(__file__).parent / "data"
pytestmark = pytest.mark.reference


@pytest.mark.parametrize(
    "kind,cls,response,family",
    [
        ("logistic", LogisticRegression, "binary", sm.families.Binomial()),
        ("poisson", PoissonRegression, "count", sm.families.Poisson()),
    ],
)
def test_glm_r_and_python(kind, cls, response, family):
    d = pd.read_csv(DATA / "r_regression_input.csv")
    x = sm.add_constant(d[["x1", "x2"]]).to_numpy()
    y = d[response].to_numpy(dtype=float)
    m = cls(device="cpu", tol=1e-10, maxiter=100).fit(torch.tensor(x), torch.tensor(y))
    reference = sm.GLM(y, x, family=family).fit(tol=1e-12)
    r = json.loads((DATA / "r_regression_reference.json").read_text())[kind]
    assert_allclose(m.params["coef"], reference.params, atol=2e-7)
    assert_allclose(m.params["coef"], r["coef"], atol=2e-7)
    assert_allclose(m.params["vcov"], r["covariance"], atol=1e-6)


def test_multinomial_coefs_and_information_against_statsmodels():
    rng = np.random.default_rng(719)
    x = np.column_stack([np.ones(1000), rng.normal(size=(1000, 2))])
    beta = np.array([[0.2, -0.3], [0.7, 0.1], [-0.2, 0.5]])
    from scipy.special import softmax

    p = softmax(np.column_stack([x @ beta, np.zeros(len(x))]), axis=1)
    y = np.array([rng.choice(3, p=row) for row in p])
    # Statsmodels base is its first category; Trex base is its last.
    sm_y = (y + 1) % 3
    r = sm.MNLogit(sm_y, x).fit(disp=0, tol=1e-12)
    m = MultinomialLogit(device="cpu", tol=1e-10, maxiter=100).fit(
        torch.tensor(x), torch.nn.functional.one_hot(torch.tensor(y), 3).double()
    )
    assert_allclose(m.params["coef"], r.params, atol=2e-6)
    permutation = np.arange(6).reshape(2, 3).T.ravel()
    cov = r.cov_params()[np.ix_(permutation, permutation)]
    assert_allclose(m.params["vcov"], cov, atol=1e-7)
    m.summary()


def test_probit_tail_likelihood_and_gradient():
    m = BinaryProbit(device="cpu")
    b = torch.tensor([40.0], dtype=torch.float64, requires_grad=True)
    x = torch.ones((1, 1), dtype=torch.float64)
    y = torch.zeros(1, dtype=torch.float64)
    loss = m._negative_log_likelihood(b, x, y)
    assert_allclose(loss.detach(), -log_ndtr(-40.0), atol=1e-10)
    loss.backward()
    assert b.grad.item() > 40
    assert torch.isfinite(m._compute_fisher_information(b, x, y)).all()


def test_probit_coefs_expected_information_against_glm():
    d = pd.read_csv(DATA / "r_regression_input.csv")
    x = sm.add_constant(d[["x1", "x2"]]).to_numpy()
    y = d.binary.to_numpy(dtype=float)
    r = sm.GLM(y, x, family=sm.families.Binomial(link=sm.families.links.Probit())).fit(
        tol=1e-12
    )
    m = BinaryProbit(device="cpu", tol=1e-10, maxiter=100).fit(
        torch.tensor(x), torch.tensor(y)
    )
    assert_allclose(m.params["coef"], r.params, atol=1e-6)
    assert_allclose(m.params["vcov"], r.cov_params(), atol=1e-7)


def test_low_rank_logit_does_not_fabricate_information():
    from trex.choice import LowRankLogit

    m = LowRankLogit(1, 3, 2, device="cpu")
    with pytest.raises(NotImplementedError, match="standard errors"):
        m._compute_fisher_information(torch.zeros(5), torch.arange(3), torch.zeros(3))
