"""
Latent-factor GLM estimators for panel prediction and counterfactual analysis.

Supports Gaussian, Bernoulli, and Poisson latent-factor models with factorized
optimization, masked training support, and prediction on held-out cells.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import BaseEstimator
from ._utils import (
    _to_tensor,
    _optimizer_display_name,
    _is_lbfgs_optimizer,
    _encode_ids,
)


def _validate_levels(values: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Map prediction-time IDs into fitted codes, raising on unseen levels."""
    codes = torch.searchsorted(levels, values)
    clamped = codes.clamp_max(levels.numel() - 1)
    valid = (codes < levels.numel()) & (levels[clamped] == values)
    if not torch.all(valid):
        raise ValueError("Prediction includes unit or time levels not seen in fitting.")
    return codes.to(dtype=torch.int64)


class LatentFactorGLM(BaseEstimator):
    """
    Latent-factor GLM with factorized optimization.

    Supported families are:

    - `"gaussian"`: y_it = x_it' beta + a_i' b_t + offset_it + eps_it
    - `"bernoulli"`: Pr(y_it = 1 | X) = sigmoid(x_it' beta + a_i' b_t + offset_it)
    - `"poisson"`: E[y_it | X] = exp(x_it' beta + a_i' b_t + offset_it)

    Parameters
    ----------
    family : str, default="gaussian"
        Outcome family. Supported values are `"gaussian"`, `"bernoulli"`,
        and `"poisson"`.
    rank : int, default=2
        Latent factor rank.
    penalty : float, default=1e-2
        Ridge penalty applied to unit and time factors.
    beta_penalty : float, default=0.0
        Optional ridge penalty for observed-effect coefficients.
    optimizer : torch optimizer class, default=torch.optim.AdamW
        Optimizer used for factorized estimation.
    optimizer_kwargs : dict, optional
        Keyword arguments forwarded to the optimizer constructor.
    maxiter : int, default=2000
        Maximum optimizer iterations.
    tol : float, default=1e-6
        Relative convergence tolerance on the loss history.
    device : torch.device | str, optional
        Computation device.
    """

    SUPPORTED_FAMILIES = {"gaussian", "bernoulli", "poisson"}
    POISSON_ETA_MAX = 20.0

    def __init__(
        self,
        family: str = "gaussian",
        rank: int = 2,
        penalty: float = 1e-2,
        beta_penalty: float = 0.0,
        optimizer: Any = torch.optim.AdamW,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        maxiter: int = 2000,
        tol: float = 1e-6,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__(device=device)
        if family not in self.SUPPORTED_FAMILIES:
            raise ValueError(
                "`family` must be one of 'gaussian', 'bernoulli', or 'poisson'."
            )
        if rank < 0:
            raise ValueError("`rank` must be non-negative.")

        self.family = family
        self.rank = int(rank)
        self.penalty = float(penalty)
        self.beta_penalty = float(beta_penalty)
        self.optimizer_class = optimizer
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.history: dict[str, list[float]] = {"loss": []}
        self._fit_context: Optional[dict[str, Any]] = None
        self._panel_shape: Optional[tuple[int, int]] = None

    def _validate_response(self, y_obs: torch.Tensor) -> None:
        """Validate outcome support for the configured family."""
        if self.family == "bernoulli":
            if torch.any((y_obs < 0) | (y_obs > 1)):
                raise ValueError("Bernoulli outcomes must lie in [0, 1].")
        elif self.family == "poisson":
            if torch.any(y_obs < 0):
                raise ValueError("Poisson outcomes must be non-negative.")

    def _initial_target(
        self,
        y_obs: torch.Tensor,
        offset_obs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Construct a working response for parameter initialization."""
        if self.family == "gaussian":
            target = y_obs
        elif self.family == "bernoulli":
            smoothed_prob = torch.clamp(0.8 * y_obs + 0.1, min=1e-4, max=1 - 1e-4)
            target = torch.logit(smoothed_prob)
        else:
            target = torch.log(torch.clamp(y_obs + 0.1, min=1e-4))

        if offset_obs is not None:
            target = target - offset_obs
        return target

    def _mean_from_index(self, index: torch.Tensor) -> torch.Tensor:
        """Map the latent index into the conditional mean."""
        if self.family == "gaussian":
            return index
        if self.family == "bernoulli":
            return torch.sigmoid(index)
        return torch.exp(torch.clamp(index, max=self.POISSON_ETA_MAX))

    def _linear_predictor(
        self,
        beta: torch.Tensor,
        unit_factors: torch.Tensor,
        time_factors: torch.Tensor,
        X_obs: torch.Tensor,
        unit_obs: torch.Tensor,
        time_obs: torch.Tensor,
        offset_obs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute x'beta + a_i'b_t + offset on the observation layout."""
        linear_term = X_obs @ beta
        if self.rank > 0:
            linear_term = linear_term + (
                unit_factors[unit_obs] * time_factors[time_obs]
            ).sum(dim=1)
        if offset_obs is not None:
            linear_term = linear_term + offset_obs
        return linear_term

    def _coerce_training_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        unit_ids: Optional[torch.Tensor],
        time_ids: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert dense panel or observation-list input into training arrays."""
        X = _to_tensor(X, self.device)
        y = _to_tensor(y, self.device, dtype=X.dtype)

        if X.ndim == 3:
            if y.ndim != 2:
                raise ValueError(
                    "Dense panel fitting requires y with shape (n_units, n_times)."
                )
            if X.shape[:2] != y.shape:
                raise ValueError(
                    "Dense X and y must agree on the first two dimensions."
                )

            n_units, n_times, n_features = X.shape
            if unit_ids is not None or time_ids is not None:
                raise ValueError(
                    "Do not pass unit_ids or time_ids with dense panel inputs."
                )

            if mask is None:
                mask_tensor = torch.ones(
                    (n_units, n_times), device=self.device, dtype=torch.bool
                )
            else:
                mask_tensor = _to_tensor(mask, self.device).to(dtype=torch.bool)
                if mask_tensor.shape != (n_units, n_times):
                    raise ValueError("Dense masks must have shape (n_units, n_times).")

            if offset is None:
                offset_tensor = None
            else:
                offset_tensor = _to_tensor(offset, self.device, dtype=X.dtype)
                if offset_tensor.shape != (n_units, n_times):
                    raise ValueError(
                        "Dense offsets must have shape (n_units, n_times)."
                    )

            unit_grid = torch.arange(n_units, device=self.device)[:, None].expand(
                n_units, n_times
            )
            time_grid = torch.arange(n_times, device=self.device)[None, :].expand(
                n_units, n_times
            )
            flat_mask = mask_tensor.reshape(-1)

            return {
                "X_obs": X.reshape(-1, n_features)[flat_mask],
                "y_obs": y.reshape(-1)[flat_mask],
                "unit_obs": unit_grid.reshape(-1)[flat_mask].to(dtype=torch.int64),
                "time_obs": time_grid.reshape(-1)[flat_mask].to(dtype=torch.int64),
                "offset_obs": (
                    None
                    if offset_tensor is None
                    else offset_tensor.reshape(-1)[flat_mask]
                ),
                "unit_levels": torch.arange(
                    n_units, device=self.device, dtype=torch.int64
                ),
                "time_levels": torch.arange(
                    n_times, device=self.device, dtype=torch.int64
                ),
                "panel_shape": (n_units, n_times),
            }

        if X.ndim != 2:
            raise ValueError(
                "Observation-list fitting requires X with shape (n_obs, n_features)."
            )
        if y.ndim != 1:
            raise ValueError("Observation-list fitting requires y with shape (n_obs,).")
        if unit_ids is None or time_ids is None:
            raise ValueError("Observation-list fitting requires unit_ids and time_ids.")

        unit_ids = _to_tensor(unit_ids, self.device).to(dtype=torch.int64)
        time_ids = _to_tensor(time_ids, self.device).to(dtype=torch.int64)
        if unit_ids.shape[0] != X.shape[0] or time_ids.shape[0] != X.shape[0]:
            raise ValueError(
                "unit_ids and time_ids must match the number of observations."
            )

        if mask is None:
            mask_tensor = torch.ones(X.shape[0], device=self.device, dtype=torch.bool)
        else:
            mask_tensor = _to_tensor(mask, self.device).to(dtype=torch.bool)
            if mask_tensor.shape != (X.shape[0],):
                raise ValueError("Observation-list masks must have shape (n_obs,).")

        if offset is None:
            offset_tensor = None
        else:
            offset_tensor = _to_tensor(offset, self.device, dtype=X.dtype)
            if offset_tensor.shape != (X.shape[0],):
                raise ValueError("Observation-list offsets must have shape (n_obs,).")

        unit_levels, unit_codes = _encode_ids(unit_ids)
        time_levels, time_codes = _encode_ids(time_ids)

        return {
            "X_obs": X[mask_tensor],
            "y_obs": y[mask_tensor],
            "unit_obs": unit_codes[mask_tensor],
            "time_obs": time_codes[mask_tensor],
            "offset_obs": None if offset_tensor is None else offset_tensor[mask_tensor],
            "unit_levels": unit_levels,
            "time_levels": time_levels,
            "panel_shape": None,
        }

    def _initialize_parameters(
        self,
        X_obs: torch.Tensor,
        y_obs: torch.Tensor,
        unit_obs: torch.Tensor,
        time_obs: torch.Tensor,
        offset_obs: Optional[torch.Tensor],
        n_units: int,
        n_times: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize beta, unit factors, and time factors."""
        working_target = self._initial_target(y_obs=y_obs, offset_obs=offset_obs)
        beta_init = torch.linalg.lstsq(X_obs, working_target).solution

        if self.rank == 0:
            unit_factors = torch.zeros(
                (n_units, 0), device=self.device, dtype=X_obs.dtype
            )
            time_factors = torch.zeros(
                (n_times, 0), device=self.device, dtype=X_obs.dtype
            )
            return beta_init, unit_factors, time_factors

        residual_matrix = torch.zeros(
            (n_units, n_times), device=self.device, dtype=X_obs.dtype
        )
        counts = torch.zeros((n_units, n_times), device=self.device, dtype=X_obs.dtype)
        residual = working_target - X_obs @ beta_init
        residual_matrix.index_put_((unit_obs, time_obs), residual, accumulate=True)
        counts.index_put_(
            (unit_obs, time_obs),
            torch.ones_like(residual),
            accumulate=True,
        )
        residual_matrix = residual_matrix / torch.clamp(counts, min=1.0)

        U, S, Vh = torch.linalg.svd(residual_matrix, full_matrices=False)
        rank = min(self.rank, S.numel())
        sqrt_s = torch.sqrt(torch.clamp(S[:rank], min=0.0))
        unit_factors = U[:, :rank] * sqrt_s
        time_factors = Vh[:rank, :].T * sqrt_s

        if rank < self.rank:
            pad_u = torch.zeros(
                (n_units, self.rank - rank), device=self.device, dtype=X_obs.dtype
            )
            pad_t = torch.zeros(
                (n_times, self.rank - rank), device=self.device, dtype=X_obs.dtype
            )
            unit_factors = torch.cat([unit_factors, pad_u], dim=1)
            time_factors = torch.cat([time_factors, pad_t], dim=1)

        return beta_init, unit_factors, time_factors

    def _objective(
        self,
        beta: torch.Tensor,
        unit_factors: torch.Tensor,
        time_factors: torch.Tensor,
        X_obs: torch.Tensor,
        y_obs: torch.Tensor,
        unit_obs: torch.Tensor,
        time_obs: torch.Tensor,
        offset_obs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Family-specific latent-factor loss with ridge penalties."""
        linear_term = self._linear_predictor(
            beta=beta,
            unit_factors=unit_factors,
            time_factors=time_factors,
            X_obs=X_obs,
            unit_obs=unit_obs,
            time_obs=time_obs,
            offset_obs=offset_obs,
        )

        if self.family == "gaussian":
            residual = y_obs - linear_term
            loss = 0.5 * torch.mean(residual**2)
        elif self.family == "bernoulli":
            loss = F.binary_cross_entropy_with_logits(
                linear_term, y_obs, reduction="mean"
            )
        else:
            loss = F.poisson_nll_loss(
                linear_term,
                y_obs,
                log_input=True,
                full=False,
                reduction="mean",
            )

        if self.rank > 0 and self.penalty > 0:
            loss = loss + 0.5 * self.penalty * (
                unit_factors.pow(2).mean() + time_factors.pow(2).mean()
            )
        if self.beta_penalty > 0:
            loss = loss + 0.5 * self.beta_penalty * beta.pow(2).mean()
        return loss

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> "LatentFactorGLM":
        """
        Fit a latent-factor GLM on observed cells.

        Parameters
        ----------
        X : torch.Tensor
            Either `(n_obs, n_features)` with explicit unit/time IDs or
            `(n_units, n_times, n_features)` for dense panels.
        y : torch.Tensor
            Either `(n_obs,)` or `(n_units, n_times)`.
        unit_ids, time_ids : torch.Tensor, optional
            Observation-level unit and time identifiers for sparse layouts.
        mask : torch.Tensor, optional
            Boolean training mask over observations or dense panel cells.
        offset : torch.Tensor, optional
            Known additive offset with the same shape as `y`.
        verbose : bool, default=False
            Whether to print convergence messages.
        """
        data = self._coerce_training_data(
            X=X,
            y=y,
            unit_ids=unit_ids,
            time_ids=time_ids,
            mask=mask,
            offset=offset,
        )
        X_obs = data["X_obs"]
        y_obs = data["y_obs"]
        unit_obs = data["unit_obs"]
        time_obs = data["time_obs"]
        offset_obs = data["offset_obs"]
        unit_levels = data["unit_levels"]
        time_levels = data["time_levels"]
        self._panel_shape = data["panel_shape"]

        if X_obs.shape[0] == 0:
            raise ValueError(
                "Training data contains no observed cells after applying the mask."
            )
        self._validate_response(y_obs)

        beta_init, unit_init, time_init = self._initialize_parameters(
            X_obs=X_obs,
            y_obs=y_obs,
            unit_obs=unit_obs,
            time_obs=time_obs,
            offset_obs=offset_obs,
            n_units=int(unit_levels.numel()),
            n_times=int(time_levels.numel()),
        )

        beta = beta_init.clone().requires_grad_(True)
        unit_factors = unit_init.clone().requires_grad_(True)
        time_factors = time_init.clone().requires_grad_(True)
        params = [beta, unit_factors, time_factors]

        if _is_lbfgs_optimizer(self.optimizer_class):
            optimizer = self.optimizer_class(
                params, max_iter=20, **self.optimizer_kwargs
            )
        else:
            optimizer = self.optimizer_class(params, **self.optimizer_kwargs)

        self.history["loss"] = []

        for iteration in range(self.maxiter):
            if _is_lbfgs_optimizer(self.optimizer_class):

                def closure():
                    optimizer.zero_grad()
                    loss = self._objective(
                        beta=beta,
                        unit_factors=unit_factors,
                        time_factors=time_factors,
                        X_obs=X_obs,
                        y_obs=y_obs,
                        unit_obs=unit_obs,
                        time_obs=time_obs,
                        offset_obs=offset_obs,
                    )
                    loss.backward()
                    return loss

                loss_val = optimizer.step(closure)
                loss_item = float(loss_val.item())
            else:
                optimizer.zero_grad()
                loss = self._objective(
                    beta=beta,
                    unit_factors=unit_factors,
                    time_factors=time_factors,
                    X_obs=X_obs,
                    y_obs=y_obs,
                    unit_obs=unit_obs,
                    time_obs=time_obs,
                    offset_obs=offset_obs,
                )
                loss.backward()
                optimizer.step()
                loss_item = float(loss.item())

            self.history["loss"].append(loss_item)
            if iteration > 10 and self.tol > 0:
                prev = self.history["loss"][-2]
                rel_change = abs(prev - loss_item) / (abs(prev) + 1e-8)
                if rel_change < self.tol:
                    if verbose:
                        print(
                            f"Convergence tolerance {self.tol} met at iteration {iteration}."
                        )
                    break

        self.iterations_run = iteration + 1
        self.params = {
            "coef": beta.detach(),
            "unit_factors": unit_factors.detach(),
            "time_factors": time_factors.detach(),
        }
        self._fit_context = {
            "unit_levels": unit_levels.detach(),
            "time_levels": time_levels.detach(),
            "n_features": X_obs.shape[1],
            "dtype": X_obs.dtype,
            "layout": "dense" if self._panel_shape is not None else "sparse",
        }
        return self

    def _coerce_prediction_inputs(
        self,
        X: torch.Tensor,
        unit_ids: Optional[torch.Tensor],
        time_ids: Optional[torch.Tensor],
        offset: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert dense or sparse prediction inputs into observation lists."""
        if self._fit_context is None:
            raise ValueError("Model has not been fitted yet.")

        dtype = self._fit_context["dtype"]
        X = _to_tensor(X, self.device, dtype=dtype)

        if X.ndim == 3:
            if self._panel_shape is None:
                raise ValueError(
                    "Dense prediction after sparse fitting is ambiguous: rows/columns "
                    "would have to be ordered by the fitted sorted unit/time levels. "
                    "Use observation-list prediction with explicit unit_ids/time_ids, "
                    "or fit with dense panel inputs."
                )
            panel_shape = X.shape[:2]
            if panel_shape != self._panel_shape:
                raise ValueError(
                    "Dense prediction panels must match the fitted panel dimensions."
                )
            n_units, n_times, n_features = X.shape
            if n_features != self._fit_context["n_features"]:
                raise ValueError(
                    "Prediction feature dimension does not match the fitted model."
                )
            if unit_ids is not None or time_ids is not None:
                raise ValueError(
                    "Do not pass unit_ids or time_ids with dense panel prediction."
                )

            unit_grid = torch.arange(n_units, device=self.device)[:, None].expand(
                n_units, n_times
            )
            time_grid = torch.arange(n_times, device=self.device)[None, :].expand(
                n_units, n_times
            )
            if offset is None:
                offset_obs = None
            else:
                offset_tensor = _to_tensor(offset, self.device, dtype=dtype)
                if offset_tensor.shape != (n_units, n_times):
                    raise ValueError(
                        "Dense prediction offsets must match the panel shape."
                    )
                offset_obs = offset_tensor.reshape(-1)

            return {
                "X_obs": X.reshape(-1, n_features),
                "unit_obs": unit_grid.reshape(-1).to(dtype=torch.int64),
                "time_obs": time_grid.reshape(-1).to(dtype=torch.int64),
                "offset_obs": offset_obs,
                "dense_shape": (n_units, n_times),
            }

        if X.ndim != 2:
            raise ValueError("Prediction X must be 2D or 3D.")
        if unit_ids is None or time_ids is None:
            raise ValueError(
                "Observation-list prediction requires unit_ids and time_ids."
            )
        if X.shape[1] != self._fit_context["n_features"]:
            raise ValueError(
                "Prediction feature dimension does not match the fitted model."
            )

        unit_ids = _to_tensor(unit_ids, self.device).to(dtype=torch.int64)
        time_ids = _to_tensor(time_ids, self.device).to(dtype=torch.int64)
        if unit_ids.shape[0] != X.shape[0] or time_ids.shape[0] != X.shape[0]:
            raise ValueError(
                "unit_ids and time_ids must match the number of prediction rows."
            )

        if offset is None:
            offset_obs = None
        else:
            offset_obs = _to_tensor(offset, self.device, dtype=dtype)
            if offset_obs.shape != (X.shape[0],):
                raise ValueError(
                    "Observation-list offsets must match the number of rows."
                )

        unit_obs = _validate_levels(unit_ids, self._fit_context["unit_levels"])
        time_obs = _validate_levels(time_ids, self._fit_context["time_levels"])
        return {
            "X_obs": X,
            "unit_obs": unit_obs,
            "time_obs": time_obs,
            "offset_obs": offset_obs,
            "dense_shape": None,
        }

    def predict_index(
        self,
        X: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the latent index x'beta + a_i'b_t + offset."""
        if not self.params:
            raise ValueError("Model has not been fitted yet.")

        data = self._coerce_prediction_inputs(
            X=X,
            unit_ids=unit_ids,
            time_ids=time_ids,
            offset=offset,
        )
        X_obs = data["X_obs"]
        unit_obs = data["unit_obs"]
        time_obs = data["time_obs"]
        offset_obs = data["offset_obs"]
        dense_shape = data["dense_shape"]

        pred = self._linear_predictor(
            beta=self.params["coef"],
            unit_factors=self.params["unit_factors"],
            time_factors=self.params["time_factors"],
            X_obs=X_obs,
            unit_obs=unit_obs,
            time_obs=time_obs,
            offset_obs=offset_obs,
        )

        if dense_shape is not None:
            return pred.reshape(dense_shape)
        return pred

    def predict_mean(
        self,
        X: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the conditional mean under the configured family."""
        index = self.predict_index(
            X=X,
            unit_ids=unit_ids,
            time_ids=time_ids,
            offset=offset,
        )
        return self._mean_from_index(index)

    def predict_proba(
        self,
        X: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        time_ids: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict Bernoulli success probabilities."""
        if self.family != "bernoulli":
            raise ValueError(
                "`predict_proba` is only available for family='bernoulli'."
            )
        return self.predict_mean(
            X=X,
            unit_ids=unit_ids,
            time_ids=time_ids,
            offset=offset,
        )

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict for dense panels only, matching the BaseEstimator interface.
        """
        return self.predict_mean(X)

    def to(self, device):
        """Move model parameters and fitted metadata to a new device."""
        super().to(device)
        if self._fit_context is not None:
            self._fit_context = {
                **self._fit_context,
                "unit_levels": self._fit_context["unit_levels"].to(device),
                "time_levels": self._fit_context["time_levels"].to(device),
            }
        return self

    def summary(self) -> None:
        """Print a concise summary of the fitted latent-factor model."""
        if not self.params:
            print("Model has not been fitted yet.")
            return

        print(f"{self.__class__.__name__} Results")
        print("=" * 40)
        print(f"Family: {self.family}")
        print(f"Rank: {self.rank}")
        print(f"Optimizer: {_optimizer_display_name(self.optimizer_class)}")
        print(f"Penalty: {self.penalty}")
        if hasattr(self, "iterations_run"):
            print(f"Optimization: {self.iterations_run}/{self.maxiter} iterations")
        if self.history["loss"]:
            print(f"Final Loss: {self.history['loss'][-1]:.6f}")
        print(f"Coefficients: {self.params['coef']}")
        print("=" * 40)
