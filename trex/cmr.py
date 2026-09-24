"""
Conditional moment restriction estimators.

This module keeps the useful, lightweight pieces of the
`conditional-moment-restrictions` project in Trex style: PyTorch tensors,
explicit estimator objects, and no solver stack beyond `torch.optim`.
"""

from itertools import product
from typing import Any, Callable, Optional

import torch

from .base import BaseEstimator
from ._utils import _to_tensor

MomentFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _as_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor[:, None]
    if tensor.ndim != 2:
        raise ValueError("Expected a 1D or 2D tensor.")
    return tensor


def squared_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared Euclidean distance.

    Parameters
    ----------
    x, y : torch.Tensor
        Two matrices with the same number of columns.
    """
    x = _as_2d(x)
    y = _as_2d(y).to(device=x.device, dtype=x.dtype)
    return torch.cdist(x, y, p=2).square()


def median_bandwidth(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Median-heuristic bandwidth for an RBF kernel.

    Zero distances are ignored when possible so repeated rows do not collapse
    the bandwidth to zero.
    """
    y = x if y is None else y
    sq_dist = squared_distance(x, y)
    positive = sq_dist[sq_dist > 0]
    if positive.numel() == 0:
        return torch.ones((), device=x.device, dtype=x.dtype)
    bandwidth = torch.sqrt(0.5 * torch.median(positive))
    return torch.clamp(bandwidth, min=torch.finfo(x.dtype).eps)


def rbf_kernel(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    bandwidth: Optional[float | torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    RBF kernel matrix and bandwidth.

    If `bandwidth` is omitted, this uses the median heuristic.
    """
    y = x if y is None else y
    x = _as_2d(x)
    y = _as_2d(y).to(device=x.device, dtype=x.dtype)

    if bandwidth is None:
        bw = median_bandwidth(x, y)
    else:
        bw = _to_tensor(bandwidth, x.device, x.dtype)
        bw = torch.clamp(bw, min=torch.finfo(x.dtype).eps)

    kernel = torch.exp(-squared_distance(x, y) / (2.0 * bw.square()))
    return kernel, bw


def hsic(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth_x: Optional[float | torch.Tensor] = None,
    bandwidth_y: Optional[float | torch.Tensor] = None,
) -> torch.Tensor:
    """
    Biased HSIC statistic using RBF kernels.
    """
    x = _as_2d(x)
    y = _as_2d(y).to(device=x.device, dtype=x.dtype)
    kx, _ = rbf_kernel(x, bandwidth=bandwidth_x)
    ky, _ = rbf_kernel(y, bandwidth=bandwidth_y)
    n = x.shape[0]
    if n < 2 or y.shape[0] != n:
        raise ValueError("HSIC requires matching samples with at least two rows")
    center = torch.eye(n, device=x.device, dtype=x.dtype)
    center = center - torch.full((n, n), 1.0 / n, device=x.device, dtype=x.dtype)
    return torch.trace(center @ kx @ center @ ky) / ((n - 1) ** 2)


def mmr_loss(
    moments: torch.Tensor,
    z: torch.Tensor,
    bandwidth: Optional[float | torch.Tensor] = None,
) -> torch.Tensor:
    """
    Maximum moment restriction loss.

    Computes `n^-2 sum_{i,j} psi_i' psi_j k(z_i, z_j)`.
    """
    moments = _as_2d(moments)
    z = _as_2d(z).to(device=moments.device, dtype=moments.dtype)
    kernel_z, _ = rbf_kernel(z, bandwidth=bandwidth)
    n = moments.shape[0]
    return torch.einsum("ir,ij,jr->", moments, kernel_z, moments) / (n**2)


def polynomial_sieve(
    z: torch.Tensor,
    degree: int = 3,
    include_interactions: bool = True,
    include_intercept: bool = True,
) -> torch.Tensor:
    """
    Polynomial basis expansion for instruments.

    The basis is intentionally simple and PyTorch-native. It is a clean
    replacement for the old SciPy B-spline helper when the goal is a small,
    dependency-light conditional moment estimator.
    """
    if degree < 1:
        raise ValueError("degree must be at least 1.")

    z = _as_2d(z)
    n_obs, dim_z = z.shape
    columns = []
    if include_intercept:
        columns.append(torch.ones(n_obs, 1, device=z.device, dtype=z.dtype))

    if include_interactions:
        powers = product(range(degree + 1), repeat=dim_z)
        for multi_index in powers:
            total_degree = sum(multi_index)
            if total_degree == 0 or total_degree > degree:
                continue
            term = torch.ones(n_obs, device=z.device, dtype=z.dtype)
            for col, power in enumerate(multi_index):
                if power:
                    term = term * z[:, col].pow(power)
            columns.append(term[:, None])
    else:
        for col in range(dim_z):
            for power in range(1, degree + 1):
                columns.append(z[:, col : col + 1].pow(power))

    return torch.cat(columns, dim=1)


class ConditionalMomentEstimator(BaseEstimator):
    """
    Base class for neural conditional moment estimators.

    Parameters
    ----------
    model : torch.nn.Module
        Structural model mapping treatments/features `t` to predictions.
    moment_function : callable
        Function `moment_function(prediction, y)` returning an `(n, q)` moment
        matrix. For NPIV this is often `prediction - y`.
    optimizer : torch optimizer class, default=torch.optim.LBFGS
        Optimizer used to update `model.parameters()`.
    maxiter : int, default=100
        Number of optimizer iterations. For LBFGS this is passed as `max_iter`.
    lr : float, default=1.0
        Optimizer learning rate.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        moment_function: MomentFunction,
        optimizer: type[torch.optim.Optimizer] = torch.optim.LBFGS,
        maxiter: int = 100,
        lr: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
        **optimizer_kwargs: Any,
    ):
        super().__init__(device=device)
        self.model = model.to(device=self.device, dtype=dtype)
        self.moment_function = moment_function
        self.optimizer_class = optimizer
        self.maxiter = maxiter
        self.lr = lr
        self.dtype = dtype
        self.optimizer_kwargs = optimizer_kwargs
        self.history: dict[str, list[float]] = {"loss": []}
        self.params = {}

    def _prepare_inputs(
        self,
        t: Any,
        y: Any,
        z: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_tensor = _as_2d(_to_tensor(t, self.device, self.dtype))
        y_tensor = _as_2d(_to_tensor(y, self.device, self.dtype))
        z_tensor = _as_2d(_to_tensor(z, self.device, self.dtype))
        if not (t_tensor.shape[0] == y_tensor.shape[0] == z_tensor.shape[0]):
            raise ValueError("t, y, and z must have the same number of rows.")
        return t_tensor, y_tensor, z_tensor

    def moments(self, t: Any, y: Any) -> torch.Tensor:
        """
        Evaluate sample moments at the current model parameters.
        """
        t_tensor = _as_2d(_to_tensor(t, self.device, self.dtype))
        y_tensor = _as_2d(_to_tensor(y, self.device, self.dtype))
        return _as_2d(self.moment_function(self.model(t_tensor), y_tensor))

    def predict(self, t: Any) -> torch.Tensor:
        """
        Predict with the fitted structural model.
        """
        t_tensor = _as_2d(_to_tensor(t, self.device, self.dtype))
        return self.model(t_tensor)

    def _fit_objective(
        self,
        objective: Callable[[], torch.Tensor],
        verbose: bool,
    ) -> None:
        params = list(self.model.parameters())
        if not params:
            raise ValueError("model must have trainable parameters.")

        if self.optimizer_class is torch.optim.LBFGS:
            options = {"max_iter": self.maxiter, "line_search_fn": "strong_wolfe"}
            options.update(self.optimizer_kwargs)
            optimizer = self.optimizer_class(params, lr=self.lr, **options)

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = objective()
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            final_loss = float(objective().detach().cpu())
            self.history["loss"].append(float(loss.detach().cpu()))
            self.history["loss"].append(final_loss)
        else:
            optimizer = self.optimizer_class(
                params,
                lr=self.lr,
                **self.optimizer_kwargs,
            )
            for _ in range(self.maxiter):
                optimizer.zero_grad()
                loss = objective()
                loss.backward()
                optimizer.step()
                self.history["loss"].append(float(loss.detach().cpu()))
            final_loss = self.history["loss"][-1]

        if verbose:
            print(f"Final loss: {final_loss:.6g}")

    def to(self, device: torch.device | str) -> "ConditionalMomentEstimator":
        super().to(device)
        self.model = self.model.to(self.device)
        return self


class MaximumMomentRestriction(ConditionalMomentEstimator):
    """
    Kernel maximum moment restriction estimator.

    This is the most direct, useful CMR component: minimize a kernelized
    conditional moment violation over the structural model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        moment_function: MomentFunction,
        bandwidth: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, moment_function=moment_function, **kwargs)
        self.bandwidth = bandwidth
        self.bandwidth_: Optional[torch.Tensor] = None

    def fit(
        self, t: Any, y: Any, z: Any, verbose: bool = False
    ) -> "MaximumMomentRestriction":
        t_tensor, y_tensor, z_tensor = self._prepare_inputs(t, y, z)
        _, self.bandwidth_ = rbf_kernel(z_tensor, bandwidth=self.bandwidth)

        def objective() -> torch.Tensor:
            moments = _as_2d(self.moment_function(self.model(t_tensor), y_tensor))
            return mmr_loss(moments, z_tensor, bandwidth=self.bandwidth_)

        self._fit_objective(objective, verbose=verbose)
        with torch.no_grad():
            final_moments = _as_2d(self.moment_function(self.model(t_tensor), y_tensor))
            final_loss = mmr_loss(final_moments, z_tensor, bandwidth=self.bandwidth_)
        self.params = {
            "mmr_loss": final_loss.detach(),
            "bandwidth": self.bandwidth_.detach(),
        }
        return self


class SieveMinimumDistance(ConditionalMomentEstimator):
    """
    Polynomial-sieve minimum distance estimator for conditional moments.

    The criterion is the quadratic norm of sample moments interacted with a
    polynomial instrument basis:

    `trace(M(theta)' W M(theta))`, where `M = F(z)' psi(theta) / n`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        moment_function: MomentFunction,
        degree: int = 3,
        include_interactions: bool = True,
        include_intercept: bool = True,
        ridge: float = 1e-6,
        **kwargs: Any,
    ):
        super().__init__(model=model, moment_function=moment_function, **kwargs)
        self.degree = degree
        self.include_interactions = include_interactions
        self.include_intercept = include_intercept
        self.ridge = ridge
        self.basis_: Optional[torch.Tensor] = None
        self.weighting_matrix_: Optional[torch.Tensor] = None
        self.z_mean_: Optional[torch.Tensor] = None
        self.z_scale_: Optional[torch.Tensor] = None

    def _standardize_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.z_mean_ is None or self.z_scale_ is None:
            self.z_mean_ = z.mean(dim=0, keepdim=True)
            self.z_scale_ = z.std(dim=0, correction=0, keepdim=True).clamp_min(1e-8)
        return (z - self.z_mean_) / self.z_scale_

    def fit(
        self, t: Any, y: Any, z: Any, verbose: bool = False
    ) -> "SieveMinimumDistance":
        t_tensor, y_tensor, z_tensor = self._prepare_inputs(t, y, z)
        self.z_mean_ = self.z_scale_ = None
        z_std = self._standardize_z(z_tensor)
        basis = polynomial_sieve(
            z_std,
            degree=self.degree,
            include_interactions=self.include_interactions,
            include_intercept=self.include_intercept,
        )
        basis_cov = basis.T @ basis / basis.shape[0]
        eye = torch.eye(basis_cov.shape[0], device=self.device, dtype=self.dtype)
        weighting_matrix = torch.linalg.pinv(basis_cov + self.ridge * eye)
        self.basis_ = basis
        self.weighting_matrix_ = weighting_matrix

        def objective() -> torch.Tensor:
            moments = _as_2d(self.moment_function(self.model(t_tensor), y_tensor))
            projected = basis.T @ moments / t_tensor.shape[0]
            return torch.trace(projected.T @ weighting_matrix @ projected)

        self._fit_objective(objective, verbose=verbose)
        with torch.no_grad():
            final_moments = _as_2d(self.moment_function(self.model(t_tensor), y_tensor))
            final_projected = basis.T @ final_moments / t_tensor.shape[0]
            final_loss = torch.trace(
                final_projected.T @ weighting_matrix @ final_projected
            )
        self.params = {
            "smd_loss": final_loss.detach(),
            "basis": basis.detach(),
            "weighting_matrix": weighting_matrix.detach(),
        }
        return self


__all__ = [
    "ConditionalMomentEstimator",
    "MaximumMomentRestriction",
    "SieveMinimumDistance",
    "hsic",
    "median_bandwidth",
    "mmr_loss",
    "polynomial_sieve",
    "rbf_kernel",
    "squared_distance",
]
