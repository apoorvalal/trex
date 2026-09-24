"""Weighted alternating projections for additive fixed effects."""

from typing import Optional, Union

import numpy as np
import torch


def prepare_fixed_effects(fe_vars: list) -> Optional[torch.Tensor]:
    """Encode each factor independently, including string or sparse numeric IDs."""
    if not fe_vars:
        return None
    arrays = []
    device = next((v.device for v in fe_vars if isinstance(v, torch.Tensor)), "cpu")
    for value in fe_vars:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim != 1:
            raise ValueError("Each fixed-effect factor must be one-dimensional")
        # pandas handles numeric/string IDs and detects missing labels uniformly.
        from pandas import factorize

        codes, _ = factorize(value, sort=False)
        if np.any(codes < 0):
            raise ValueError("Fixed-effect labels must not be missing")
        arrays.append(torch.as_tensor(codes, device=device))
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Fixed-effect factors must have the same length")
    return torch.stack(arrays, dim=1)


def demean_torch(
    x: Union[np.ndarray, torch.Tensor],
    flist: Union[np.ndarray, torch.Tensor],
    weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    tol: float = 1e-8,
    maxiter: int = 100_000,
) -> tuple[torch.Tensor, bool]:
    """Return weighted within residuals and a convergence flag.

    Weights must be finite and strictly positive. Each factor is re-encoded, so
    allocation depends on the number of levels, not the largest label. All
    columns share one projection pass; the returned array includes the last
    (converged) update. Computation uses float64, as in the original API.
    """
    x = torch.as_tensor(x).to(dtype=torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] == 0 or not torch.isfinite(x).all():
        raise ValueError("x must be a nonempty, finite matrix")
    f = torch.as_tensor(flist, device=x.device)
    if f.ndim == 1:
        f = f[:, None]
    if f.ndim != 2 or len(f) != len(x) or not torch.isfinite(f).all():
        raise ValueError("Fixed effects must be finite and match the rows of x")
    if tol <= 0 or not np.isfinite(tol) or maxiter < 1:
        raise ValueError("tol and maxiter must be positive")
    w = (
        torch.ones(len(x), dtype=x.dtype, device=x.device)
        if weights is None
        else torch.as_tensor(weights, dtype=x.dtype, device=x.device)
    )
    if w.shape != (len(x),) or not torch.isfinite(w).all() or torch.any(w <= 0):
        raise ValueError("weights must be finite, strictly positive, and match x")
    factors = []
    for j in range(f.shape[1]):
        levels, codes = torch.unique(f[:, j], return_inverse=True)
        totals = torch.zeros(len(levels), dtype=x.dtype, device=x.device).index_add_(
            0, codes, w
        )
        factors.append((codes, totals))
    current = x.clone()
    for _ in range(maxiter):
        previous = current
        for codes, totals in factors:
            sums = torch.zeros(len(totals), x.shape[1], dtype=x.dtype, device=x.device)
            sums.index_add_(0, codes, current * w[:, None])
            current = current - (sums / totals[:, None])[codes]
        if current.numel() == 0 or torch.max(torch.abs(current - previous)) < tol:
            return current, True
    return current, False


def fixed_effect_rank(codes: torch.Tensor) -> int:
    """Exact rank of additive FE incidence, without dense n-by-level dummies.

    One/two-factor rank uses level counts / graph connected components. For
    three or more factors a bounded level-by-level Gram calculation is used.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    f = codes.detach().cpu().numpy()
    encoded = [np.unique(f[:, j], return_inverse=True)[1] for j in range(f.shape[1])]
    sizes = [int(c.max()) + 1 for c in encoded]
    if len(sizes) == 1:
        return sizes[0]
    if len(sizes) == 2:
        edge = sparse.coo_matrix(
            (np.ones(len(f)), (encoded[0], encoded[1] + sizes[0])),
            shape=(sum(sizes), sum(sizes)),
        )
        return sum(sizes) - connected_components(
            edge, directed=False, return_labels=False
        )
    if sum(sizes) > 2048:
        raise ValueError(
            "Exact rank for 3+ FE factors with >2048 levels is expensive; supply df_absorbed explicitly"
        )
    offsets = np.cumsum([0] + sizes[:-1])
    columns = np.column_stack([c + o for c, o in zip(encoded, offsets)]).ravel()
    dummy = sparse.coo_matrix(
        (np.ones(len(columns)), (np.repeat(np.arange(len(f)), len(sizes)), columns)),
        shape=(len(f), sum(sizes)),
    ).tocsr()
    return int(np.linalg.matrix_rank((dummy.T @ dummy).toarray(), hermitian=True))
