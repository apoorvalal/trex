"""Independent NumPy/SciPy Bellman and likelihood references."""

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from scipy.special import logsumexp
from scipy.optimize import minimize
from trex.choice.dynamic import (
    RustNFP,
    HotzMillerCCP,
    ReplacementUtility,
    LinearFlowUtility,
    DynamicChoiceData,
)

pytestmark = pytest.mark.reference


def setup_model(cls=RustNFP):
    p = np.zeros((8, 2, 8))
    for s in range(8):
        p[s, 0, s] += 0.25
        p[s, 0, min(s + 1, 7)] += 0.75
        p[s, 1, 0] = 0.8
        p[s, 1, 1] = 0.2
    m = cls(8, 2, 0.9, device="cpu", tol=1e-9, maxiter=80)
    m.set_transition_probabilities(torch.tensor(p))
    m.set_flow_utility(
        ReplacementUtility(
            theta_maintenance=0.01, theta_replacement_cost=8.0, device="cpu"
        )
    )
    return m, p


def numpy_log_probs(b, p):
    u = np.column_stack([-b[0] * np.arange(len(p)), np.full(len(p), -b[1])])
    v = np.zeros(len(p))
    for _ in range(1000):
        q = u + 0.9 * np.einsum("iaj,j->ia", p, v)
        updated = logsumexp(q, axis=1)
        if np.max(abs(v - updated)) < 1e-12:
            break
        v = updated
    return q - logsumexp(q, axis=1, keepdims=True)


@pytest.mark.parametrize("seed", [11, 22, 33])
def test_rust_fits_scipy_reference(seed):
    model, p = setup_model()
    rng = np.random.default_rng(seed)
    probabilities = np.exp(numpy_log_probs([0.1, 2.0], p))
    current = rng.integers(8, size=200)
    states, actions = [], []
    for t in range(100):
        chosen = (rng.random(200) > probabilities[current, 0]).astype(int)
        states.extend(current)
        actions.extend(chosen)
        current = np.array([rng.choice(8, p=p[s, a]) for s, a in zip(current, chosen)])
    states, actions = np.array(states), np.array(actions)
    counts = np.zeros((8, 2))
    np.add.at(counts, (states, actions), 1)
    objective = lambda b: -np.sum(counts * numpy_log_probs(b, p)) / len(states)
    reference = minimize(objective, [0.05, 1.0], method="BFGS", tol=1e-9)
    model.fit({"states": torch.tensor(states), "actions": torch.tensor(actions)})
    assert_allclose(model.params["coef"], reference.x, atol=2e-4)
    assert_allclose(
        model.predict_proba(torch.arange(8)),
        np.exp(numpy_log_probs(reference.x, p)),
        atol=2e-5,
    )


def test_rare_action_gradient_is_not_floored():
    m, p = setup_model()
    b = torch.tensor([0.1, 100.0], dtype=torch.float64, requires_grad=True)
    loss = m._negative_log_likelihood(
        b, {"states": torch.tensor([0]), "actions": torch.tensor([1])}
    )
    loss.backward()
    assert loss.item() > 90
    assert b.grad[1] > 0.9


@pytest.mark.parametrize("cls", [RustNFP, HotzMillerCCP])
def test_counterfactual_uses_changed_policy_without_mutation(cls):
    m, p = setup_model(cls)
    m.params = {"coef": torch.tensor([0.1, 2.0], dtype=torch.float64)}
    before = m.predict_proba(torch.arange(8)).clone()
    data = DynamicChoiceData(
        states=torch.tensor([4, 1, 2, 3]),
        actions=torch.tensor([0, 0, 1, 0]),
        next_states=torch.tensor([5, 2, 0, 4]),
        individual_ids=torch.tensor([10, 20, 10, 20]),
        time_periods=torch.tensor([2002, 2001, 2001, 2002]),
    )
    result = m.counterfactual(
        data, {"theta_replacement_cost": -100.0}, rng=torch.Generator().manual_seed(42)
    )
    assert result["simulated_actions"].shape == (2, 2)
    assert torch.all(result["simulated_actions"] == 1)
    assert_allclose(result["simulated_states"][:, 0], [2, 1])
    assert_allclose(m.predict_proba(torch.arange(8)), before)
    assert m.utility_fn.get_params()["theta_replacement_cost"] == 8.0
    with pytest.raises(ValueError, match="Unknown"):
        m.counterfactual(data, {"typo": 1})


def test_hm_float32_ccps_float64_transitions_matches_linear_solve():
    m, p = setup_model(HotzMillerCCP)
    m.ccp_hat = torch.tensor(
        np.exp(numpy_log_probs([0.1, 2.0], p)), dtype=torch.float32
    )
    m._precompute_inversion_matrices()
    u = m._flow_utility(torch.tensor([0.1, 2.0], dtype=torch.float64))
    c = m.ccp_hat.numpy()
    rhs = np.sum(c * u.numpy(), axis=1) - np.sum(
        np.where(c > 0, c * np.log(c), 0), axis=1
    )
    ref = np.linalg.solve(np.eye(8) - 0.9 * np.einsum("ia,iaj->ij", c, p), rhs)
    assert_allclose(m.invert_ccps(u), ref, atol=1e-10)


def test_linear_flow_features_retained_after_fit():
    m, p = setup_model()
    m.set_flow_utility(LinearFlowUtility(1, 2, device="cpu"))
    data = {
        "states": torch.arange(8).repeat(30),
        "actions": torch.tensor([0, 1]).repeat(120),
        "all_states_features": torch.arange(8).float()[:, None],
    }
    original = data.copy()
    m.fit(data)
    assert_allclose(m.predict_proba(torch.arange(8)).sum(1), np.ones(8), atol=1e-10)
    assert data["all_states_features"] is original["all_states_features"]
    m.to("cpu")
    assert m.simulate(torch.tensor([0]), 2)[0].shape == (1, 3)
