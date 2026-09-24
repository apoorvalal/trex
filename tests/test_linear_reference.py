"""Independent dense-dummy/WLS and R regression oracles."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from trex import LinearRegression, demean_torch

DATA = Path(__file__).parent / "data"
pytestmark = pytest.mark.reference


@pytest.mark.parametrize("weighted,fe", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("se", ["classical", "HC0", "HC1"])
def test_linear_against_r_and_statsmodels(weighted, fe, se):
    d = pd.read_csv(DATA / "r_regression_input.csv")
    ref = json.loads((DATA / "r_regression_reference.json").read_text())
    x = sm.add_constant(d[["x1", "x2"]]).to_numpy()
    w = d.w.to_numpy() if weighted else np.ones(len(d))
    dense = x
    if fe:
        dense = np.column_stack(
            [
                x,
                pd.get_dummies(d.unit, drop_first=True),
                pd.get_dummies(d.period, drop_first=True),
            ]
        )
    oracle = sm.WLS(d.y, dense, weights=w).fit()
    if se != "classical":
        oracle = oracle.get_robustcov_results(cov_type=se)
    model = LinearRegression(device="cpu").fit(
        torch.tensor(x),
        torch.tensor(d.y.to_numpy()),
        weights=torch.tensor(w),
        fe=[d.unit.to_numpy(), d.period.to_numpy()] if fe else None,
        se=se,
    )
    sl = slice(1, 3) if fe else slice(None)
    assert_allclose(model.params["coef"][sl], oracle.params[sl], atol=1e-8)
    assert_allclose(model.params["se"][sl], oracle.bse[sl], atol=1e-8)
    name = "fe" if fe else "wls" if weighted else "ols"
    covname = "covariance" if se == "classical" else se
    assert_allclose(
        model.params["se"][sl], np.sqrt(np.diag(ref[name][covname]))[sl], atol=1e-8
    )
    if fe and se == "HC1":
        assert_allclose(
            model.params["se"][1:], np.sqrt(np.diag(ref["fixest"]["HC1"])), atol=1e-8
        )
    assert_allclose(model.fitted_values_, oracle.fittedvalues, atol=1e-8)


def test_no_intercept_classical_variance():
    d = pd.read_csv(DATA / "r_regression_input.csv")
    x = torch.tensor(d[["x1", "x2"]].to_numpy())
    m = LinearRegression(device="cpu").fit(
        x, torch.tensor(d.y.to_numpy()), se="classical"
    )
    r = json.loads((DATA / "r_regression_reference.json").read_text())["no_intercept"]
    assert_allclose(m.params["se"], np.sqrt(np.diag(r["covariance"])), atol=1e-9)


def test_demeaning_returns_final_iterate_and_handles_sparse_labels():
    x = torch.tensor([1e-10, 2e-10], dtype=torch.float64)
    actual, ok = demean_torch(x, torch.tensor([-100, -100]), tol=1e-8)
    assert ok
    assert_allclose(actual[:, 0], [-5e-11, 5e-11], atol=1e-20)


def test_invalid_linear_inputs_and_refit():
    x = torch.arange(20, dtype=torch.float64).reshape(10, 2)
    y = torch.arange(10, dtype=torch.float64)
    for weights in [torch.zeros(10), -torch.ones(10), torch.ones(9)]:
        with pytest.raises(ValueError):
            LinearRegression(device="cpu").fit(x, y, weights=weights)
    with pytest.raises(ValueError, match="solver"):
        LinearRegression(solver="typo")
    with pytest.raises(ValueError, match="fitted"):
        LinearRegression().predict(x)
    model = LinearRegression(device="cpu").fit(
        torch.ones((10, 1)), y, fe=[np.arange(10) // 2]
    )
    model.fit(torch.ones((10, 1)), y, se="HC1")
    assert_allclose(model.params["coef"], [4.5])


def test_disconnected_fe_rank_and_numpy_solver():
    from trex.demean import fixed_effect_rank

    f = torch.tensor([[0, 0], [0, 1], [1, 1], [2, 2], [2, 3], [3, 3]])
    assert fixed_effect_rank(f) == 6
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 2))
    y = x @ [2.0, -3.0] + rng.normal(size=30)
    a = LinearRegression(solver="numpy", device="cpu").fit(x, y)
    assert_allclose(a.params["coef"], sm.OLS(y, x).fit().params)
