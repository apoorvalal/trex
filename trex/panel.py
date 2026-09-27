"""Panel estimators for matrix completion and synthetic difference-in-differences.

This module contains pure-torch implementations of two workhorse panel tools:

* nuclear-norm penalized matrix completion with optional additive unit/time
  effects, following the soft-impute/SVT update used in MCPanel; and
* synthetic difference-in-differences weights via Frank-Wolfe simplex least
  squares, following the synthdid reference solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .base import BaseEstimator
from ._utils import _to_tensor


def _svt(
    matrix: torch.Tensor, threshold: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Singular-value thresholding operator."""
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    S_shrunk = torch.clamp(S - threshold, min=0.0)
    keep = S_shrunk > 0
    if not torch.any(keep):
        return torch.zeros_like(matrix), S_shrunk
    return (U[:, keep] * S_shrunk[keep]) @ Vh[keep, :], S_shrunk


def _center_effects(
    row_effects: torch.Tensor, col_effects: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize additive effects without changing their sum."""
    row_mean = row_effects.mean()
    row_effects = row_effects - row_mean
    col_effects = col_effects + row_mean
    col_mean = col_effects.mean()
    col_effects = col_effects - col_mean
    row_effects = row_effects + col_mean
    return row_effects, col_effects


def matrix_completion_lambda_max(
    Y: torch.Tensor,
    mask: torch.Tensor,
    fit_unit_effects: bool = True,
    fit_time_effects: bool = True,
    n_effect_iters: int = 20,
) -> torch.Tensor:
    """Largest lambda for which the SVT update returns a zero low-rank matrix."""
    Y = torch.as_tensor(Y)
    mask = torch.as_tensor(mask, device=Y.device).to(dtype=torch.bool)
    work = torch.where(mask, Y, torch.zeros_like(Y))
    row_effects = torch.zeros(Y.shape[0], device=Y.device, dtype=Y.dtype)
    col_effects = torch.zeros(Y.shape[1], device=Y.device, dtype=Y.dtype)

    for _ in range(n_effect_iters):
        if fit_unit_effects:
            resid = Y - col_effects[None, :]
            denom = mask.sum(dim=1).clamp_min(1).to(Y.dtype)
            row_effects = torch.where(
                mask.any(dim=1),
                torch.where(mask, resid, torch.zeros_like(resid)).sum(dim=1) / denom,
                row_effects,
            )
        if fit_time_effects:
            resid = Y - row_effects[:, None]
            denom = mask.sum(dim=0).clamp_min(1).to(Y.dtype)
            col_effects = torch.where(
                mask.any(dim=0),
                torch.where(mask, resid, torch.zeros_like(resid)).sum(dim=0) / denom,
                col_effects,
            )
        if fit_unit_effects and fit_time_effects:
            row_effects, col_effects = _center_effects(row_effects, col_effects)

    residual = Y - row_effects[:, None] - col_effects[None, :]
    work = torch.where(mask, residual, torch.zeros_like(Y))
    n_obs = mask.sum().to(Y.dtype).clamp_min(1)
    return 2.0 * torch.linalg.svdvals(work).max() / n_obs


@dataclass
class MatrixCompletionResult:
    completed: torch.Tensor
    low_rank: torch.Tensor
    unit_effects: torch.Tensor
    time_effects: torch.Tensor
    singular_values: torch.Tensor
    lambda_L: float
    objective: float
    iterations: int


class NuclearNormMatrixCompletion(BaseEstimator):
    """Nuclear-norm penalized matrix completion with additive fixed effects.

    Solves

        min_{L,u,v} |Omega|^{-1} ||P_Omega(Y - L - u1' - 1v')||_F^2
                   + lambda_L ||L||_*

    by alternating least-squares updates for the additive effects with the SVT
    projection update from MCPanel.  The implementation is dense and pure torch,
    so it runs on CPU or GPU and is intended as a clean scalable baseline.
    """

    def __init__(
        self,
        lambda_L: Optional[float] = None,
        lambda_fraction: float = 0.25,
        fit_unit_effects: bool = True,
        fit_time_effects: bool = True,
        maxiter: int = 500,
        effect_iters: int = 2,
        tol: float = 1e-6,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__(device=device)
        self.lambda_L = lambda_L
        self.lambda_fraction = float(lambda_fraction)
        self.fit_unit_effects = bool(fit_unit_effects)
        self.fit_time_effects = bool(fit_time_effects)
        self.maxiter = int(maxiter)
        self.effect_iters = int(effect_iters)
        self.tol = float(tol)
        if self.maxiter < 1 or self.effect_iters < 1 or self.tol <= 0:
            raise ValueError("Iteration counts and tolerance must be positive")
        self.history: dict[str, list[float]] = {"objective": [], "rmse": []}

    def _update_effects(
        self,
        Y: torch.Tensor,
        mask: torch.Tensor,
        L: torch.Tensor,
        row_effects: torch.Tensor,
        col_effects: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.effect_iters):
            if self.fit_unit_effects:
                resid = Y - L - col_effects[None, :]
                denom = mask.sum(dim=1).clamp_min(1).to(Y.dtype)
                updated = (
                    torch.where(mask, resid, torch.zeros_like(resid)).sum(dim=1) / denom
                )
                row_effects = torch.where(mask.any(dim=1), updated, row_effects)
            if self.fit_time_effects:
                resid = Y - L - row_effects[:, None]
                denom = mask.sum(dim=0).clamp_min(1).to(Y.dtype)
                updated = (
                    torch.where(mask, resid, torch.zeros_like(resid)).sum(dim=0) / denom
                )
                col_effects = torch.where(mask.any(dim=0), updated, col_effects)
            if self.fit_unit_effects and self.fit_time_effects:
                row_effects, col_effects = _center_effects(row_effects, col_effects)
        return row_effects, col_effects

    def _objective(
        self,
        Y: torch.Tensor,
        mask: torch.Tensor,
        L: torch.Tensor,
        row_effects: torch.Tensor,
        col_effects: torch.Tensor,
        singular_values: torch.Tensor,
        lambda_L: float,
    ) -> torch.Tensor:
        fitted = L + row_effects[:, None] + col_effects[None, :]
        residual = torch.where(mask, fitted - Y, torch.zeros_like(Y))
        n_obs = mask.sum().to(Y.dtype).clamp_min(1)
        return residual.pow(2).sum() / n_obs + lambda_L * singular_values.sum()

    def fit(
        self, Y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> "NuclearNormMatrixCompletion":
        Y = _to_tensor(Y, self.device)
        if not Y.is_floating_point():
            Y = Y.to(torch.float64)
        if Y.ndim != 2:
            raise ValueError("Y must be a 2D panel matrix.")
        if mask is None:
            mask_t = torch.isfinite(Y)
        else:
            mask_t = _to_tensor(mask, self.device).to(dtype=torch.bool)
            if mask_t.shape != Y.shape:
                raise ValueError("mask must have the same shape as Y.")
        if not torch.any(mask_t):
            raise ValueError("mask contains no observed entries.")
        if not torch.isfinite(Y[mask_t]).all():
            raise ValueError("Observed panel entries must be finite")
        Y_work = torch.where(mask_t, Y, torch.zeros_like(Y))

        if self.lambda_L is None:
            lambda_max = matrix_completion_lambda_max(
                Y_work,
                mask_t,
                fit_unit_effects=self.fit_unit_effects,
                fit_time_effects=self.fit_time_effects,
            ).to(dtype=Y.dtype, device=Y.device)
            lambda_L = float((self.lambda_fraction * lambda_max).item())
        else:
            lambda_L = float(self.lambda_L)
        if lambda_L < 0:
            raise ValueError("lambda_L must be nonnegative.")

        L = torch.zeros_like(Y_work)
        row_effects = torch.zeros(Y.shape[0], device=self.device, dtype=Y.dtype)
        col_effects = torch.zeros(Y.shape[1], device=self.device, dtype=Y.dtype)
        singular_values = torch.zeros(min(Y.shape), device=self.device, dtype=Y.dtype)
        n_obs = mask_t.sum().to(Y.dtype).clamp_min(1)
        threshold = lambda_L * n_obs / 2.0
        previous_obj: Optional[float] = None
        self.history = {"objective": [], "rmse": []}

        for iteration in range(self.maxiter):
            row_effects, col_effects = self._update_effects(
                Y_work, mask_t, L, row_effects, col_effects
            )
            fitted = L + row_effects[:, None] + col_effects[None, :]
            projected = L + torch.where(
                mask_t, Y_work - fitted, torch.zeros_like(Y_work)
            )
            L, singular_values = _svt(projected, threshold)
            obj = self._objective(
                Y_work, mask_t, L, row_effects, col_effects, singular_values, lambda_L
            )
            rmse = torch.sqrt(
                torch.where(
                    mask_t,
                    (L + row_effects[:, None] + col_effects[None, :] - Y_work) ** 2,
                    torch.zeros_like(Y_work),
                ).sum()
                / n_obs
            )
            obj_item = float(obj.item())
            self.history["objective"].append(obj_item)
            self.history["rmse"].append(float(rmse.item()))
            if previous_obj is not None:
                rel = abs(previous_obj - obj_item) / (abs(previous_obj) + 1e-12)
                if rel < self.tol:
                    break
            previous_obj = obj_item

        completed = L + row_effects[:, None] + col_effects[None, :]
        self.result_ = MatrixCompletionResult(
            completed=completed.detach(),
            low_rank=L.detach(),
            unit_effects=row_effects.detach(),
            time_effects=col_effects.detach(),
            singular_values=singular_values.detach(),
            lambda_L=lambda_L,
            objective=self.history["objective"][-1],
            iterations=iteration + 1,
        )
        self.params = {
            "completed": self.result_.completed,
            "low_rank": self.result_.low_rank,
            "unit_effects": self.result_.unit_effects,
            "time_effects": self.result_.time_effects,
        }
        return self

    def predict(self) -> torch.Tensor:
        if not hasattr(self, "result_"):
            raise ValueError("Model has not been fitted yet.")
        return self.result_.completed


def collapsed_form(Y: torch.Tensor, N0: int, T0: int) -> torch.Tensor:
    """Collapse treated units and post-treatment periods as in synthdid."""
    N, T = Y.shape
    top = torch.cat([Y[:N0, :T0], Y[:N0, T0:T].mean(dim=1, keepdim=True)], dim=1)
    bottom = torch.cat(
        [
            Y[N0:N, :T0].mean(dim=0, keepdim=True),
            Y[N0:N, T0:T].mean().reshape(1, 1),
        ],
        dim=1,
    )
    return torch.cat([top, bottom], dim=0)


def _demean_columns(Y: torch.Tensor) -> torch.Tensor:
    return Y - Y.mean(dim=0, keepdim=True)


def frank_wolfe_step(
    A: torch.Tensor, x: torch.Tensor, b: torch.Tensor, eta: float | torch.Tensor
) -> torch.Tensor:
    """One exact-line-search Frank-Wolfe step over the probability simplex."""
    Ax = A @ x
    half_grad = (Ax - b) @ A + eta * x
    i = torch.argmin(half_grad)
    direction = -x.clone()
    direction[i] = 1.0 - x[i]
    if torch.all(direction == 0):
        return x
    d_err = A[:, i] - Ax
    denom = d_err.pow(2).sum() + eta * direction.pow(2).sum()
    if denom <= 0:
        return x
    step = -torch.dot(half_grad, direction) / denom
    step = torch.clamp(step, min=0.0, max=1.0)
    return x + step * direction


def simplex_least_squares_fw(
    A: torch.Tensor,
    b: torch.Tensor,
    zeta: float = 0.0,
    intercept: bool = True,
    x0: Optional[torch.Tensor] = None,
    min_decrease: float = 1e-8,
    maxiter: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimize ||Ax-b||^2 + zeta^2 n ||x||^2 over simplex by Frank-Wolfe."""
    A = torch.as_tensor(A)
    b = torch.as_tensor(b, device=A.device, dtype=A.dtype)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("A must be 2D and b must be a compatible vector.")
    if x0 is None:
        x = torch.full((A.shape[1],), 1.0 / A.shape[1], device=A.device, dtype=A.dtype)
    else:
        x = torch.as_tensor(x0, device=A.device, dtype=A.dtype)
        x = torch.clamp(x, min=0)
        total = x.sum()
        x = torch.full_like(x, 1.0 / x.numel()) if total <= 0 else x / total
    if intercept:
        A = _demean_columns(A)
        b = b - b.mean()
    eta = A.shape[0] * float(zeta) ** 2
    vals = []
    prev: Optional[float] = None
    for _ in range(maxiter):
        x = frank_wolfe_step(A, x, b, eta)
        residual = A @ x - b
        val = float(
            (
                residual.pow(2).sum() / A.shape[0] + float(zeta) ** 2 * x.pow(2).sum()
            ).item()
        )
        vals.append(val)
        if prev is not None and prev - val <= min_decrease**2:
            break
        prev = val
    return x, torch.tensor(vals, device=A.device, dtype=A.dtype)


def _noise_level(Y: torch.Tensor, N0: int, T0: int) -> torch.Tensor:
    diffs = Y[:N0, 1:T0] - Y[:N0, : T0 - 1]
    return diffs.flatten().std(unbiased=True)


@dataclass
class SyntheticDIDResult:
    estimate: torch.Tensor
    omega: torch.Tensor
    lambda_: torch.Tensor
    omega_values: torch.Tensor
    lambda_values: torch.Tensor


class SyntheticDID(BaseEstimator):
    """Synthetic difference-in-differences with two-way balancing weights."""

    def __init__(
        self,
        eta_omega: Optional[float] = None,
        eta_lambda: float = 1e-6,
        zeta_omega: Optional[float] = None,
        zeta_lambda: Optional[float] = None,
        omega_intercept: bool = True,
        lambda_intercept: bool = True,
        min_decrease: float = 1e-8,
        maxiter: int = 1000,
        sparsify: bool = True,
        omega: Optional[torch.Tensor] = None,
        lambda_weights: Optional[torch.Tensor] = None,
        update_omega: Optional[bool] = None,
        update_lambda: Optional[bool] = None,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__(device=device)
        self.eta_omega = eta_omega
        self.eta_lambda = float(eta_lambda)
        self.zeta_omega = zeta_omega
        self.zeta_lambda = zeta_lambda
        self.omega_intercept = bool(omega_intercept)
        self.lambda_intercept = bool(lambda_intercept)
        self.min_decrease = float(min_decrease)
        self.maxiter = int(maxiter)
        self.sparsify = bool(sparsify)
        self.omega = omega
        self.lambda_weights = lambda_weights
        self.update_omega = update_omega
        self.update_lambda = update_lambda

    @staticmethod
    def sparsify_weights(weights: torch.Tensor) -> torch.Tensor:
        out = weights.clone()
        out[out <= out.max() / 4.0] = 0
        total = out.sum()
        return torch.full_like(out, 1.0 / out.numel()) if total <= 0 else out / total

    def fit(self, Y: torch.Tensor, N0: int, T0: int) -> "SyntheticDID":
        Y = _to_tensor(Y, self.device)
        if Y.ndim != 2:
            raise ValueError("Y must be a 2D panel matrix.")
        N, T = Y.shape
        if not (0 < N0 < N and 0 < T0 < T):
            raise ValueError(
                "N0 and T0 must define nonempty control/treated and pre/post blocks."
            )
        sigma = _noise_level(Y, N0, T0)
        eta_omega = (
            self.eta_omega
            if self.eta_omega is not None
            else ((N - N0) * (T - T0)) ** 0.25
        )
        zeta_omega = (
            float(self.zeta_omega)
            if self.zeta_omega is not None
            else float(eta_omega * sigma)
        )
        zeta_lambda = (
            float(self.zeta_lambda)
            if self.zeta_lambda is not None
            else float(self.eta_lambda * sigma)
        )

        Yc = collapsed_form(Y, N0, T0)
        lambda_init = (
            None
            if self.lambda_weights is None
            else _to_tensor(self.lambda_weights, self.device, dtype=Y.dtype)
        )
        omega_init = (
            None
            if self.omega is None
            else _to_tensor(self.omega, self.device, dtype=Y.dtype)
        )
        update_lambda = (
            self.update_lambda
            if self.update_lambda is not None
            else lambda_init is None
        )
        update_omega = (
            self.update_omega if self.update_omega is not None else omega_init is None
        )

        if lambda_init is not None:
            if lambda_init.shape != (T0,):
                raise ValueError("lambda_weights must have shape (T0,).")
            lambda_init = torch.clamp(lambda_init, min=0)
            lambda_init = lambda_init / lambda_init.sum().clamp_min(
                torch.finfo(Y.dtype).eps
            )
        if omega_init is not None:
            if omega_init.shape != (N0,):
                raise ValueError("omega must have shape (N0,).")
            omega_init = torch.clamp(omega_init, min=0)
            omega_init = omega_init / omega_init.sum().clamp_min(
                torch.finfo(Y.dtype).eps
            )

        if update_lambda:
            lambda_, lambda_vals = simplex_least_squares_fw(
                Yc[:N0, :T0],
                Yc[:N0, T0],
                zeta=zeta_lambda,
                intercept=self.lambda_intercept,
                x0=lambda_init,
                min_decrease=self.min_decrease,
                maxiter=min(100, self.maxiter) if self.sparsify else self.maxiter,
            )
            if self.sparsify:
                lambda_, lambda_vals = simplex_least_squares_fw(
                    Yc[:N0, :T0],
                    Yc[:N0, T0],
                    zeta=zeta_lambda,
                    intercept=self.lambda_intercept,
                    x0=self.sparsify_weights(lambda_),
                    min_decrease=self.min_decrease,
                    maxiter=self.maxiter,
                )
        else:
            if lambda_init is None:
                raise ValueError(
                    "Fixed lambda requested but lambda_weights was not provided."
                )
            lambda_ = lambda_init
            lambda_vals = torch.empty(0, device=Y.device, dtype=Y.dtype)

        if update_omega:
            omega, omega_vals = simplex_least_squares_fw(
                Yc[:N0, :T0].T,
                Yc[N0, :T0],
                zeta=zeta_omega,
                intercept=self.omega_intercept,
                x0=omega_init,
                min_decrease=self.min_decrease,
                maxiter=min(100, self.maxiter) if self.sparsify else self.maxiter,
            )
            if self.sparsify:
                omega, omega_vals = simplex_least_squares_fw(
                    Yc[:N0, :T0].T,
                    Yc[N0, :T0],
                    zeta=zeta_omega,
                    intercept=self.omega_intercept,
                    x0=self.sparsify_weights(omega),
                    min_decrease=self.min_decrease,
                    maxiter=self.maxiter,
                )
        else:
            if omega_init is None:
                raise ValueError("Fixed omega requested but omega was not provided.")
            omega = omega_init
            omega_vals = torch.empty(0, device=Y.device, dtype=Y.dtype)

        post_weights = torch.full(
            (T - T0,), 1.0 / (T - T0), device=Y.device, dtype=Y.dtype
        )
        treated_weights = torch.full(
            (N - N0,), 1.0 / (N - N0), device=Y.device, dtype=Y.dtype
        )
        row_contrast = torch.cat([-omega, treated_weights])
        col_contrast = torch.cat([-lambda_, post_weights])
        estimate = row_contrast @ Y @ col_contrast
        self.result_ = SyntheticDIDResult(
            estimate=estimate.detach(),
            omega=omega.detach(),
            lambda_=lambda_.detach(),
            omega_values=omega_vals.detach(),
            lambda_values=lambda_vals.detach(),
        )
        self.params = {
            "estimate": self.result_.estimate,
            "omega": self.result_.omega,
            "lambda": self.result_.lambda_,
        }
        self.N0_, self.T0_ = int(N0), int(T0)
        return self

    def predict(self) -> torch.Tensor:
        if not hasattr(self, "result_"):
            raise ValueError("Model has not been fitted yet.")
        return self.result_.estimate


def synthdid_estimate(
    Y: torch.Tensor, N0: int, T0: int, **kwargs
) -> SyntheticDIDResult:
    """Convenience function returning a fitted :class:`SyntheticDIDResult`."""
    return SyntheticDID(**kwargs).fit(Y, N0, T0).result_


def did_estimate(Y: torch.Tensor, N0: int, T0: int) -> torch.Tensor:
    """Classical two-way DID estimate for a block treatment design."""
    Y = torch.as_tensor(Y)
    N, T = Y.shape
    return (
        Y[N0:N, T0:T].mean()
        - Y[N0:N, :T0].mean()
        - Y[:N0, T0:T].mean()
        + Y[:N0, :T0].mean()
    )


def sc_estimate(
    Y: torch.Tensor, N0: int, T0: int, eta_omega: float = 1e-6, **kwargs
) -> SyntheticDIDResult:
    """Synthetic-control special case: no time weights and no omega intercept."""
    Yt = torch.as_tensor(Y)
    return (
        SyntheticDID(
            eta_omega=eta_omega,
            lambda_weights=torch.zeros(T0, dtype=Yt.dtype, device=Yt.device),
            update_lambda=False,
            omega_intercept=False,
            **kwargs,
        )
        .fit(Yt, N0, T0)
        .result_
    )


def matrix_completion_estimate(
    Y: torch.Tensor, N0: int, T0: int, **kwargs
) -> tuple[torch.Tensor, NuclearNormMatrixCompletion]:
    """Estimate ATT by imputing treated post cells with matrix completion."""
    Y = torch.as_tensor(Y)
    mask = torch.ones_like(Y, dtype=torch.bool)
    mask[N0:, T0:] = False
    model = NuclearNormMatrixCompletion(**kwargs).fit(Y, mask=mask)
    tau = (Y[N0:, T0:] - model.predict()[N0:, T0:]).mean()
    return tau, model


def panel_estimates(
    Y: torch.Tensor,
    N0: int,
    T0: int,
    methods: Optional[list[str]] = None,
    mc_kwargs: Optional[dict] = None,
    sdid_kwargs: Optional[dict] = None,
) -> dict[str, torch.Tensor]:
    """Compute the point-estimate row from the synthdid README estimator table."""
    Y = torch.as_tensor(Y)
    N, T = Y.shape
    default_methods = [
        "DID",
        "Synthetic Control (SC)",
        "Synthetic DID (SDID)",
        "Time Weighted DID",
        "SDID (No Intercept)",
        "SC with FEs (DIFP)",
        "Matrix Completion",
        "SC (Regularized)",
        "DIFP (Regularized)",
    ]
    methods = default_methods if methods is None else methods
    mc_kwargs = dict(mc_kwargs or {})
    sdid_kwargs = dict(sdid_kwargs or {})
    uniform_lambda = torch.full((T0,), 1.0 / T0, dtype=Y.dtype, device=Y.device)
    uniform_omega = torch.full((N0,), 1.0 / N0, dtype=Y.dtype, device=Y.device)
    reg_eta = ((N - N0) * (T - T0)) ** 0.25
    out: dict[str, torch.Tensor] = {}
    for method in methods:
        if method == "DID":
            out[method] = did_estimate(Y, N0, T0)
        elif method == "Synthetic Control (SC)":
            out[method] = sc_estimate(Y, N0, T0, eta_omega=1e-6, **sdid_kwargs).estimate
        elif method == "Synthetic DID (SDID)":
            out[method] = synthdid_estimate(Y, N0, T0, **sdid_kwargs).estimate
        elif method == "Time Weighted DID":
            out[method] = synthdid_estimate(
                Y, N0, T0, omega=uniform_omega, update_omega=False, **sdid_kwargs
            ).estimate
        elif method == "SDID (No Intercept)":
            out[method] = synthdid_estimate(
                Y, N0, T0, omega_intercept=False, **sdid_kwargs
            ).estimate
        elif method == "SC with FEs (DIFP)":
            out[method] = synthdid_estimate(
                Y,
                N0,
                T0,
                lambda_weights=uniform_lambda,
                update_lambda=False,
                eta_omega=1e-6,
                **sdid_kwargs,
            ).estimate
        elif method == "Matrix Completion":
            out[method] = matrix_completion_estimate(Y, N0, T0, **mc_kwargs)[0]
        elif method == "SC (Regularized)":
            out[method] = sc_estimate(
                Y, N0, T0, eta_omega=reg_eta, **sdid_kwargs
            ).estimate
        elif method == "DIFP (Regularized)":
            out[method] = synthdid_estimate(
                Y,
                N0,
                T0,
                lambda_weights=uniform_lambda,
                update_lambda=False,
                **sdid_kwargs,
            ).estimate
        else:
            raise ValueError(f"Unknown panel estimator method: {method}")
    return out
