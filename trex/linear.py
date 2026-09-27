"""OLS/WLS with weighted fixed-effect absorption and sandwich inference."""

from typing import Optional, Union, List

import numpy as np
import torch

from .base import BaseEstimator
from .demean import demean_torch, fixed_effect_rank, prepare_fixed_effects


class LinearRegression(BaseEstimator):
    """Least squares with optional additive fixed effects.

    ``X`` must include a constant if desired. ``weights`` are strictly positive
    analytic/WLS weights, not frequency counts. ``classical`` uses weighted SSE
    divided by residual degrees of freedom; HC0/HC1 use the weighted sandwich.
    HC1 counts the exact rank of the absorbed effects by default.

    Absorbed columns (including an intercept) retain zero coefficient/SE slots.
    ``predict(X)`` returns the slope component only, **not** FE level predictions.
    ``fitted_values_`` includes the fitted effects for the training observations.
    """

    def __init__(self, solver="torch", device=None):
        super().__init__(device=device)
        if solver not in {"torch", "numpy"}:
            raise ValueError("solver must be 'torch' or 'numpy'")
        self.solver = solver

    def fit(
        self,
        X,
        y,
        se=None,
        fe: Optional[Union[List, torch.Tensor]] = None,
        weights=None,
        *,
        tol=1e-10,
        maxiter=100_000,
        df_absorbed=None,
    ):
        """Fit; optionally override absorbed rank for large multi-way FE designs.

        Rank-deficient non-absorbed columns raise instead of returning arbitrary
        coefficients. Failed absorption raises rather than returning a partial fit.
        """
        X = torch.as_tensor(X, device=self.device)
        dtype = X.dtype if X.is_floating_point() else torch.float64
        if fe is not None:
            dtype = torch.float64
        X = X.to(dtype=dtype)
        y = torch.as_tensor(y, device=self.device, dtype=dtype)
        if X.ndim != 2 or y.shape != (X.shape[0],) or len(X) == 0:
            raise ValueError("Expected nonempty X (n,p) and y (n,)")
        if not torch.isfinite(X).all() or not torch.isfinite(y).all():
            raise ValueError("X and y must be finite; handle missing rows explicitly")
        if se not in {None, "classical", "HC0", "HC1"}:
            raise ValueError("se must be classical, HC0, HC1 or None")
        w = (
            torch.ones(len(X), dtype=dtype, device=self.device)
            if weights is None
            else torch.as_tensor(weights, dtype=dtype, device=self.device)
        )
        if w.shape != y.shape or not torch.isfinite(w).all() or torch.any(w <= 0):
            raise ValueError("weights must be finite, strictly positive and match y")
        self.params = None
        xw, yw = X, y
        f = None
        if fe is not None:
            if isinstance(fe, list):
                f = prepare_fixed_effects(fe)
            else:
                f = torch.as_tensor(fe, device=self.device)
                if f.ndim == 1:
                    f = f[:, None]
            if f is None or f.shape[1] == 0:
                raise ValueError("fe must contain at least one factor")
            within, converged = demean_torch(
                torch.column_stack([X, y]), f, w, tol=tol, maxiter=maxiter
            )
            if not converged:
                raise RuntimeError("Fixed-effect absorption did not converge")
            xw, yw = within[:, :-1], within[:, -1]
        keep = torch.linalg.vector_norm(xw, dim=0) > tol * torch.linalg.vector_norm(
            X, dim=0
        ).clamp_min(1)
        # Only FE absorption may remove columns; otherwise rank deficiency is an error.
        if f is None:
            keep = torch.ones(X.shape[1], dtype=torch.bool, device=self.device)
        design = xw[:, keep] * w.sqrt()[:, None]
        target = yw * w.sqrt()
        k = design.shape[1]
        if k and int(torch.linalg.matrix_rank(design)) != k:
            raise ValueError("Design is rank deficient after absorption")
        if k == 0:
            beta = torch.empty(0, dtype=dtype, device=self.device)
        elif self.solver == "torch":
            beta = torch.linalg.lstsq(design, target).solution
        else:
            beta = torch.as_tensor(
                np.linalg.lstsq(design.cpu().numpy(), target.cpu().numpy(), rcond=None)[
                    0
                ],
                dtype=dtype,
                device=self.device,
            )
        coef = torch.zeros(X.shape[1], dtype=dtype, device=self.device)
        coef[keep] = beta
        residual = yw - xw[:, keep] @ beta
        self.params = {"coef": coef}
        self.residuals_ = residual.detach()
        self.fitted_values_ = (y - residual).detach()
        self._dropped_cols = ~keep
        if se:
            absorbed = (
                0
                if f is None
                else fixed_effect_rank(f) if df_absorbed is None else df_absorbed
            )
            if not isinstance(absorbed, (int, np.integer)) or absorbed < 0:
                raise ValueError("df_absorbed must be a nonnegative integer")
            self.df_resid_ = len(X) - k - absorbed
            if self.df_resid_ <= 0:
                raise ValueError("No residual degrees of freedom for standard errors")
            bread = (
                torch.linalg.inv(design.T @ design)
                if k
                else torch.empty((0, 0), dtype=dtype, device=self.device)
            )
            wr = residual * w.sqrt()
            if se == "classical":
                covariance = bread * (wr.square().sum() / self.df_resid_)
            else:
                score = design * wr[:, None]
                covariance = bread @ (score.T @ score) @ bread
                if se == "HC1":
                    covariance *= len(X) / self.df_resid_
            full_cov = torch.zeros(
                X.shape[1], X.shape[1], dtype=dtype, device=self.device
            )
            idx = torch.where(keep)[0]
            full_cov[idx[:, None], idx[None, :]] = covariance
            self.params.update(vcov=full_cov, se=full_cov.diag().clamp_min(0).sqrt())
        return self

    def predict(self, X):
        """Slope-only predictions; additive FE contributions are not included."""
        if self.params is None:
            raise ValueError("Model must be fitted before prediction")
        X = torch.as_tensor(X, dtype=self.params["coef"].dtype, device=self.device)
        if X.ndim != 2 or X.shape[1] != len(self.params["coef"]):
            raise ValueError("Prediction features do not match the fitted model")
        return X @ self.params["coef"]
