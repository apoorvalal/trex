"""Two-step GMM with SciPy/Torch objectives and shared sandwich inference.

Moment functions return an (n, q) array, not necessarily linear-IV moments.
Covariances use the uncentered second moment with divisor n (HC0 convention).
``iid=False`` selects Bartlett/Newey-West HAC for both weighting and inference.
``vtheta_`` retains the legacy asymptotic scale; ``covariance_ = vtheta_ / n``.
"""

from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
import torch
import torchmin


def moment_covariance(moments, max_lags=0):
    """Uncentered long-run covariance of observation-level moment vectors."""
    g = np.asarray(moments, dtype=float)
    if g.ndim != 2 or len(g) < 2 or g.shape[1] == 0 or not np.isfinite(g).all():
        raise ValueError("moments must be finite with shape (n>=2, q>=1)")
    if not isinstance(max_lags, (int, np.integer)) or not 0 <= max_lags < len(g):
        raise ValueError("max_lags must be an integer in [0, n)")
    omega = g.T @ g / len(g)
    for lag in range(1, max_lags + 1):
        cross = g[lag:].T @ g[:-lag] / len(g)
        omega += (1 - lag / (max_lags + 1)) * (cross + cross.T)
    return omega


def numerical_jacobian(function, theta):
    """Central finite differences with scale-aware steps, for NumPy callbacks."""
    step = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(theta))
    eye = np.eye(len(theta))
    return np.column_stack(
        [
            (function(theta + eye[j] * step[j]) - function(theta - eye[j] * step[j]))
            / (2 * step[j])
            for j in range(len(theta))
        ]
    )


class GMMEstimator:
    """Dispatch to a SciPy or Torch GMM implementation.

    ``weighting_matrix`` is 'optimal' (two-step), 'identity', or a fixed
    symmetric positive-definite matrix. ``init_params`` in fit defaults to
    zeros with x.shape[1] entries; pass it for other parameter dimensions.
    """

    def __new__(
        cls,
        moment_cond: Callable,
        weighting_matrix="optimal",
        backend="scipy",
        **kwargs,
    ):
        if cls is GMMEstimator:
            if backend.lower() not in _BACKENDS:
                raise ValueError(f"Unsupported backend: {backend}")
            cls = _BACKENDS[backend.lower()]
        return object.__new__(cls)

    def __init__(
        self, moment_cond, weighting_matrix="optimal", backend="scipy", device=None
    ):
        self.moment_cond = moment_cond
        if isinstance(weighting_matrix, str) and weighting_matrix not in {
            "optimal",
            "identity",
        }:
            raise ValueError(
                "weighting_matrix must be 'optimal', 'identity', or a matrix"
            )
        self.weighting_matrix = weighting_matrix
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.theta_ = self.std_errors_ = self.W_ = None

    def _numpy(self, value):
        return (
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )

    def _convert(self, value):
        if isinstance(self, GMMEstimatorTorch):
            return torch.as_tensor(value, dtype=torch.float64, device=self.device)
        return np.asarray(value, dtype=float)

    def _moments(self, theta):
        return self.moment_cond(self.z_, self.y_, self.x_, theta)

    def gmm_objective(self, beta):
        moments = self._moments(beta)
        avg = (
            moments.mean(dim=0)
            if isinstance(moments, torch.Tensor)
            else moments.mean(axis=0)
        )
        return avg @ self.W_ @ avg

    def _compute_hac_covariance(self, moments, max_lags=None):
        if max_lags is None:
            max_lags = min(
                len(moments) - 1, int(np.floor(4 * (len(moments) / 100) ** (2 / 9)))
            )
        return moment_covariance(moments, max_lags)

    def optimal_weighting_matrix(self, moments):
        omega = moment_covariance(self._numpy(moments), getattr(self, "max_lags_", 0))
        # A rank-deficient moment set is not repaired by silently inventing a ridge.
        if np.linalg.matrix_rank(omega) < len(omega):
            raise ValueError("Moment covariance is singular; remove redundant moments")
        return self._convert(np.linalg.inv(omega))

    def fit(
        self,
        z,
        y,
        x,
        verbose=False,
        fit_method=None,
        iid=True,
        two_step=True,
        *,
        init_params=None,
        max_lags=None,
        tol=1e-9,
        maxiter=2000,
    ):
        z, y, x = [np.asarray(self._numpy(a), dtype=float) for a in (z, y, x)]
        if (
            x.ndim != 2
            or z.ndim != 2
            or y.shape != (len(x),)
            or len(z) != len(x)
            or len(x) < 2
        ):
            raise ValueError("Expected x (n,p), z (n,q), and y (n,) with n>=2")
        if not all(np.isfinite(a).all() for a in (x, y, z)):
            raise ValueError("Inputs must be finite")
        self.n_ = len(x)
        start = (
            np.zeros(x.shape[1])
            if init_params is None
            else np.asarray(self._numpy(init_params), dtype=float)
        )
        if start.ndim != 1 or not len(start) or not np.isfinite(start).all():
            raise ValueError("init_params must be a nonempty finite vector")
        self.k_ = len(start)
        self.z_, self.y_, self.x_ = [self._convert(a) for a in (z, y, x)]
        self.max_lags_ = (
            0
            if iid
            else (
                min(len(x) - 1, int(np.floor(4 * (len(x) / 100) ** (2 / 9))))
                if max_lags is None
                else max_lags
            )
        )
        initial = self._numpy(self._moments(self._convert(start)))
        moment_covariance(initial, self.max_lags_)  # validate dimensions and lags
        if len(initial) != self.n_ or initial.shape[1] < self.k_:
            raise ValueError("Need at least as many moments as parameters, with n rows")
        q = initial.shape[1]
        mode = (
            self.weighting_matrix if isinstance(self.weighting_matrix, str) else "fixed"
        )
        weight = (
            np.eye(q)
            if mode != "fixed"
            else np.asarray(self._numpy(self.weighting_matrix), dtype=float)
        )
        if (
            weight.shape != (q, q)
            or not np.isfinite(weight).all()
            or not np.allclose(weight, weight.T)
            or np.linalg.eigvalsh(weight).min() <= 0
        ):
            raise ValueError(
                "Weight matrix must be finite, symmetric positive definite, and q-by-q"
            )
        self.W_ = self._convert(weight)
        self.theta_ = self.std_errors_ = None
        result = self._optimize(start, fit_method, verbose, tol, maxiter)
        if mode == "optimal" and two_step:
            self.W_ = self.optimal_weighting_matrix(
                self._moments(self._convert(self._numpy(result.x)))
            )
            result = self._optimize(
                self._numpy(result.x), fit_method, verbose, tol, maxiter
            )
        self.result_ = result
        self.theta_ = self._numpy(result.x).copy()
        moments = self._numpy(self._moments(self._convert(self.theta_)))
        self.Omega_ = moment_covariance(moments, self.max_lags_)
        self.Gamma_ = self.jacobian_moment_cond()
        w = self._numpy(self.W_)
        bread_inv = self.Gamma_.T @ w @ self.Gamma_
        if np.linalg.matrix_rank(bread_inv) < self.k_:
            raise ValueError("Parameters are not locally identified by these moments")
        bread = np.linalg.inv(bread_inv)
        middle = self.Gamma_.T @ w @ self.Omega_ @ w @ self.Gamma_
        self.vtheta_ = bread @ middle @ bread
        self.covariance_ = self.vtheta_ / self.n_
        self.std_errors_ = np.sqrt(np.maximum(0.0, np.diag(self.covariance_)))
        return self

    def jacobian_moment_cond(self):
        if self.theta_ is None:
            raise ValueError("Model must be fitted first")
        if isinstance(self, GMMEstimatorTorch):
            theta = self._convert(self.theta_).requires_grad_(True)
            jac = torch.autograd.functional.jacobian(
                lambda b: self._moments(b).mean(dim=0), theta
            )
            self.jac_est_ = self._numpy(jac)
        else:
            self.jac_est_ = numerical_jacobian(
                lambda b: self._moments(b).mean(axis=0), self.theta_
            )
        return self.jac_est_

    def summary(self, prec=4, alpha=0.05):
        if self.theta_ is None or self.std_errors_ is None:
            raise ValueError("Estimator not fitted; call fit() first")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        z = self.theta_ / self.std_errors_
        bound = stats.norm.ppf(1 - alpha / 2) * self.std_errors_
        return pd.DataFrame(
            {
                "coef": self.theta_,
                "std err": self.std_errors_,
                "t": z,
                "p-value": 2 * stats.norm.sf(np.abs(z)),
                f"[{alpha/2}": self.theta_ - bound,
                f"{1-alpha/2}]": self.theta_ + bound,
            }
        ).round(prec)


class GMMEstimatorScipy(GMMEstimator):
    def _optimize(self, start, method, verbose, tol, maxiter):
        result = optimize.minimize(
            self.gmm_objective,
            start,
            method=method or "BFGS",
            jac=lambda b: numerical_jacobian(
                lambda p: np.atleast_1d(self.gmm_objective(p)), b
            ).ravel(),
            tol=tol,
            options={"maxiter": maxiter},
        )
        gradient = numerical_jacobian(
            lambda p: np.atleast_1d(self.gmm_objective(p)), result.x
        )
        if not np.isfinite(result.fun) or (
            not result.success
            and np.linalg.norm(gradient, ord=np.inf) > max(1e-6, 10 * tol)
        ):
            raise RuntimeError(f"GMM optimization failed: {result.message}")
        return result

    @staticmethod
    def iv_moment(z, y, x, beta):
        return z * (y - x @ beta)[:, None]


class GMMEstimatorTorch(GMMEstimator):
    def _optimize(self, start, method, verbose, tol, maxiter):
        if method is None or method == "l-bfgs":
            from types import SimpleNamespace

            p = self._convert(start).clone().requires_grad_(True)
            optimizer = torch.optim.LBFGS(
                [p],
                max_iter=maxiter,
                tolerance_grad=tol,
                tolerance_change=np.finfo(float).eps,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad()
                loss = self.gmm_objective(p)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure()
            result = SimpleNamespace(
                x=p.detach(),
                fun=loss.detach(),
                success=bool(p.grad.abs().max() <= max(1e-6, 10 * tol)),
                message="LBFGS gradient tolerance not met",
            )
        else:
            result = torchmin.minimize(
                self.gmm_objective,
                self._convert(start),
                method=method,
                tol=tol,
                max_iter=maxiter,
                disp=verbose,
            )
        p = result.x.detach().requires_grad_(True)
        gradient = torch.autograd.grad(self.gmm_objective(p), p)[0]
        if not torch.isfinite(result.fun) or (
            not result.success and float(gradient.abs().max()) > max(1e-6, 10 * tol)
        ):
            raise RuntimeError(f"GMM optimization failed: {result.message}")
        return result

    def to(self, device):
        self.device = torch.device(device)
        for name in ("z_", "y_", "x_", "W_"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    @staticmethod
    def iv_moment(z, y, x, beta):
        return z * (y - x @ beta).unsqueeze(-1)


_BACKENDS = {"scipy": GMMEstimatorScipy, "torch": GMMEstimatorTorch}
