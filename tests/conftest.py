"""
Pytest configuration and fixtures for trex tests
"""

import torch
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def seed_torch():
    """Set random seed for reproducible tests"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def simple_regression_data():
    """Simple regression dataset for testing"""
    torch.manual_seed(42)
    n, p = 100, 3
    X = torch.randn(n, p)
    true_coef = torch.tensor([1.0, -0.5, 0.8])
    y = X @ true_coef + 0.1 * torch.randn(n)
    return X, y, true_coef


@pytest.fixture
def panel_data():
    """Panel data with fixed effects for testing"""
    torch.manual_seed(42)
    n_firms, n_years = 10, 5
    n_obs = n_firms * n_years

    X = torch.randn(n_obs, 2)
    firm_ids = torch.repeat_interleave(torch.arange(n_firms), n_years)
    year_ids = torch.tile(torch.arange(n_years), (n_firms,))

    # Add intercept
    X_with_intercept = torch.cat([torch.ones(n_obs, 1), X], dim=1)
    true_coef = torch.tensor([2.0, 1.0, -0.5])

    # Generate fixed effects
    firm_effects = torch.randn(n_firms)[firm_ids]
    year_effects = torch.randn(n_years)[year_ids]

    y = (
        X_with_intercept @ true_coef
        + firm_effects
        + year_effects
        + 0.1 * torch.randn(n_obs)
    )

    return X_with_intercept, y, firm_ids, year_ids, true_coef


@pytest.fixture
def binary_classification_data():
    """Binary classification dataset for logistic regression"""
    torch.manual_seed(42)
    n, p = 200, 3
    X = torch.randn(n, p)
    X_with_intercept = torch.cat([torch.ones(n, 1), X], dim=1)
    true_coef = torch.tensor([0.5, 1.0, -0.8, 0.3])

    logits = X_with_intercept @ true_coef
    probs = torch.sigmoid(logits)
    y = torch.bernoulli(probs).to(torch.float32)

    return X_with_intercept, y, true_coef
