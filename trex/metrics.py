"""Distribution diagnostics; raw-scale distances are sensitive to units.

Standardize with training-set scales before comparing heterogeneous columns.
These checks do not establish causal/econometric fidelity of generated samples.
"""

from typing import Any
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def _validate_pair(real, fake, min_rows=1):
    real, fake = np.asarray(real, dtype=float), np.asarray(fake, dtype=float)
    if (
        real.ndim != 2
        or fake.ndim != 2
        or real.shape[1] != fake.shape[1]
        or real.shape[1] == 0
    ):
        raise ValueError("Expected two matrices with matching, nonzero column counts")
    if (
        min(len(real), len(fake)) < min_rows
        or not np.isfinite(real).all()
        or not np.isfinite(fake).all()
    ):
        raise ValueError(f"Each matrix needs at least {min_rows} finite rows")
    return real, fake


def distribution_metrics(
    real: Any,
    fake: Any,
    *,
    n_projections: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Compute marginal and joint distribution discrepancy metrics."""

    real_array, fake_array = _validate_pair(real, fake, min_rows=2)

    marginal_w1 = [
        wasserstein_distance(real_array[:, j], fake_array[:, j])
        for j in range(real_array.shape[1])
    ]
    marginal_ks = [
        ks_2samp(real_array[:, j], fake_array[:, j]).statistic
        for j in range(real_array.shape[1])
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_real = np.corrcoef(real_array, rowvar=False)
        corr_fake = np.corrcoef(fake_array, rowvar=False)
    corr_real = np.nan_to_num(corr_real)
    corr_fake = np.nan_to_num(corr_fake)
    return {
        "marginal_w1_mean": float(np.mean(marginal_w1)),
        "marginal_w1_max": float(np.max(marginal_w1)),
        "marginal_ks_mean": float(np.mean(marginal_ks)),
        "marginal_ks_max": float(np.max(marginal_ks)),
        "mean_l2": float(
            np.linalg.norm(real_array.mean(axis=0) - fake_array.mean(axis=0))
        ),
        "cov_frobenius": float(
            np.linalg.norm(
                np.cov(real_array, rowvar=False) - np.cov(fake_array, rowvar=False)
            )
        ),
        "corr_frobenius": float(np.linalg.norm(corr_real - corr_fake)),
        "sliced_wasserstein": sliced_wasserstein_distance(
            real_array,
            fake_array,
            n_projections=n_projections,
            seed=seed,
        ),
    }


def sliced_wasserstein_distance(
    real: np.ndarray,
    fake: np.ndarray,
    *,
    n_projections: int = 128,
    seed: int = 0,
) -> float:
    """Monte Carlo directional average of exact empirical 1-Wasserstein.

    Unequal sample sizes are integrated using empirical CDF masses, not matched
    by truncating the sorted arrays (which discards one distribution's tail).
    """
    real, fake = _validate_pair(real, fake)
    if not isinstance(n_projections, (int, np.integer)) or n_projections < 1:
        raise ValueError("n_projections must be a positive integer")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_projections, real.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    a, b = real @ directions.T, fake @ directions.T
    if len(a) == len(b):
        return float(np.mean(abs(np.sort(a, axis=0) - np.sort(b, axis=0))))
    return float(
        np.mean([wasserstein_distance(a[:, j], b[:, j]) for j in range(n_projections)])
    )
