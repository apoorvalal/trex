"""CPU/CUDA parity for corrected numerical paths; CPU CI skips this module."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from trex import LinearRegression
from trex.choice import MultinomialLogit
from trex.choice.dynamic import HotzMillerCCP, ReplacementUtility
from trex.choice.ccp_estimators import estimate_ccps
from trex.gmm.gmm import GMMEstimator, GMMEstimatorTorch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_weighted_fe_cpu_cuda():
    rng = np.random.default_rng(61)
    x = rng.normal(size=(120, 2))
    y = x @ [0.4, 1.2] + rng.normal(size=120)
    w = np.exp(x[:, 0])
    fe = [np.arange(120) // 6, np.arange(120) % 6]
    cpu = LinearRegression(device="cpu").fit(x, y, weights=w, fe=fe, se="HC1")
    gpu = LinearRegression(device="cuda").fit(x, y, weights=w, fe=fe, se="HC1")
    for key in ("coef", "vcov"):
        assert_allclose(gpu.params[key].cpu(), cpu.params[key], atol=1e-9)


def test_multinomial_cpu_cuda():
    rng = np.random.default_rng(42)
    x = torch.tensor(np.column_stack([np.ones(120), rng.normal(size=120)]))
    y = torch.nn.functional.one_hot(torch.tensor(rng.integers(0, 3, 120)), 3).double()
    a = MultinomialLogit(device="cpu", tol=1e-10, maxiter=60).fit(x, y)
    b = MultinomialLogit(device="cuda", tol=1e-10, maxiter=60).fit(x, y)
    assert_allclose(a.params["coef"], b.params["coef"].cpu(), atol=1e-6)
    assert_allclose(a.params["vcov"], b.params["vcov"].cpu(), atol=1e-7)


def test_hm_first_stage_gpu_and_gmm():
    states = torch.tensor([0, 0, 1, 1, 1], device="cuda")
    actions = torch.tensor([0, 1, 1, 0, 1], device="cuda")
    c = estimate_ccps(states, actions, 3, 2)
    assert c.device.type == "cuda"
    assert_allclose(c.cpu(), [[0.5, 0.5], [1 / 3, 2 / 3], [0.5, 0.5]], atol=1e-7)
    m = HotzMillerCCP(3, 2, 0.9, device="cuda")
    m.set_transition_probabilities(
        torch.eye(3, dtype=torch.float64)[:, None, :].repeat(1, 2, 1)
    )
    m.set_flow_utility(ReplacementUtility(device="cuda"))
    m.ccp_hat = c
    m._precompute_inversion_matrices()
    assert torch.isfinite(
        m.invert_ccps(torch.zeros((3, 2), device="cuda", dtype=torch.float64))
    ).all()
    x = np.column_stack([np.ones(100), np.linspace(-1, 1, 100)])
    y = x @ [1.0, 2.0] + np.sin(np.arange(100)) * 0.1
    r = GMMEstimator(GMMEstimatorTorch.iv_moment, backend="torch", device="cuda").fit(
        x, y, x
    )
    assert_allclose(r.theta_, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-6)
