"""
Generalized empirical likelihood (GEL) estimators.

Implements empirical likelihood (EL), exponential tilting (ET), and CUE-style
objectives with asymptotic covariance and overidentification testing.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Optional
import logging
from scipy.stats import chi2


# Tilt functions for GEL
def rho_exponential(v: np.ndarray) -> np.ndarray:
    """Exponential tilting (ET): rho(v) = 1 - exp(v)"""
    return 1 - np.exp(v)


def rho_cue(v: np.ndarray) -> np.ndarray:
    """Continuously Updated Estimator (CUE): rho(v) = -0.5*v^2 - v"""
    return -0.5 * v**2 - v


def rho_el(v: np.ndarray) -> np.ndarray:
    """Empirical Likelihood (EL): rho(v) = log(1-v)
    Note: requires v < 1 for all observations
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(v < 1, np.log1p(-v), -np.inf)


class GELEstimator:
    """
    Generalized empirical likelihood estimator for vector-valued moments.

    Parameters
    ----------
    m : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Moment function returning an `(n_obs, n_moments)` array for data and
        parameter vector.
    rho : Callable[[np.ndarray], np.ndarray], default=rho_exponential
        GEL tilt function defining the criterion (ET, EL, or CUE).
    min_method : str, default="L-BFGS-B"
        Optimization method used for both inner and outer problems.
    verbose : bool, default=False
        If True, enables optimizer display output.
    log : bool, default=False
        If True, sets logger level to INFO.
    """

    def __init__(
        self,
        m: Callable[[np.ndarray, np.ndarray], np.ndarray],
        rho: Callable[[np.ndarray], np.ndarray] = rho_exponential,
        min_method: str = "L-BFGS-B",
        verbose: bool = False,
        log: bool = False,
    ):
        self.m = m
        self.rho = rho
        self.rho_prime = self._get_rho_derivative(rho)
        self.rho_double_prime = self._get_rho_second_derivative(rho)
        self._min_method = min_method
        self._verbose = verbose
        self.est: Optional[np.ndarray] = None
        self.lam_hat: Optional[np.ndarray] = None
        self.Sigma: Optional[np.ndarray] = None
        self.se: Optional[np.ndarray] = None
        self.J_stat: Optional[float] = None
        self.J_pvalue: Optional[float] = None

        if log:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

    def fit(
        self,
        D: np.ndarray,
        startval: np.ndarray,
        startval2: Optional[np.ndarray] = None,
    ) -> None:
        """Fit GEL estimator with proper asymptotic standard errors"""
        if startval2 is None:
            startval2 = np.zeros(self.m(D, startval).shape[1])  # Start lambda at zero

        self.D_ = D
        self.n_ = D.shape[0]

        # Minimize the profiled (maximized over lambda) GEL criterion
        result = minimize(
            lambda theta: self._profile_value_gradient(theta, D, startval2),
            startval,
            jac=True,
            method=self._min_method,
            tol=1e-10,
            options={"maxiter": 2000},
        )

        self.result_ = result
        if not np.isfinite(result.fun) or not result.success:
            raise RuntimeError(f"GEL outer optimization failed: {result.message}")
        self.est = result.x

        # Get optimal lambda for final theta
        lam_result = self._solve_inner(self.est, D, startval2)
        self.lam_hat = lam_result.x

        # Compute proper asymptotic standard errors
        self._compute_asymptotic_covariance()

        # Compute J-test statistic
        self._compute_j_test()

    def summary(self, alpha: float = 0.05) -> dict:
        """Summary table with test statistics"""
        if self.est is None or self.se is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        from scipy.stats import norm

        t_stats = self.est / self.se
        p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))

        critical_val = norm.ppf(1 - alpha / 2)
        ci_lower = self.est - critical_val * self.se
        ci_upper = self.est + critical_val * self.se

        summary_dict = {
            "coefficients": self.est,
            "std_errors": self.se,
            "t_statistics": t_stats,
            "p_values": p_values,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "J_statistic": self.J_stat,
            "J_pvalue": self.J_pvalue,
            "n_obs": self.n_,
        }

        return summary_dict

    def _outer_maximisation(
        self, theta: np.ndarray, D: np.ndarray, startval2: np.ndarray
    ) -> float:
        result = self._solve_inner(theta, D, startval2)
        return -result.fun

    def _solve_inner(self, theta, D, start):
        from types import SimpleNamespace

        moments = self.m(D, theta)
        lam = np.asarray(start, dtype=float).copy()
        n = len(moments)
        for _ in range(200):
            tilts = moments @ lam
            objective = -self.rho(tilts).mean()
            gradient = -moments.T @ self.rho_prime(tilts) / n
            if np.linalg.norm(gradient, ord=np.inf) < 1e-9:
                return SimpleNamespace(x=lam, fun=objective * n, success=True)
            hessian = -(moments.T * self.rho_double_prime(tilts)) @ moments / n
            step = np.linalg.solve(hessian, gradient)
            rate = 1.0
            for _ in range(60):
                candidate = lam - rate * step
                v = moments @ candidate
                if self.rho is rho_el and np.any(v >= 1):
                    rate *= 0.5
                    continue
                trial = -self.rho(v).mean()
                if (
                    np.isfinite(trial)
                    and trial <= objective - 1e-4 * rate * (gradient @ step) + 1e-15
                ):
                    lam = candidate
                    break
                rate *= 0.5
            else:
                raise RuntimeError("GEL inner line search did not converge")
        raise RuntimeError("GEL inner optimization did not converge")

    def _profile_value_gradient(self, theta, D, start):
        from .gmm import numerical_jacobian

        result = self._solve_inner(theta, D, start)
        lam = result.x
        g = self.m(D, theta)
        # Envelope theorem: differentiate moments, holding optimal lambda fixed.
        derivative = numerical_jacobian(lambda b: self.m(D, b) @ lam, theta)
        gradient = (self.rho_prime(g @ lam)[:, None] * derivative).mean(axis=0)
        return -result.fun / len(D), gradient

    def _inner_minimisation(
        self, lam: np.ndarray, theta: np.ndarray, D: np.ndarray
    ) -> float:
        moments = self.m(D, theta)  # Moment conditions (n x k)
        tilts = np.dot(moments, lam)  # (n,)
        obj_value = -np.sum(self.rho(tilts))
        logging.info(f"Inner minimisation: lam={lam}, Objective value: {obj_value}")
        return obj_value

    def _get_rho_derivative(self, rho_func):
        """Return derivative of rho function"""
        if rho_func == rho_exponential:
            return lambda v: -np.exp(v)
        elif rho_func == rho_cue:
            return lambda v: -v - 1
        elif rho_func == rho_el:
            return lambda v: -1 / (1 - v)
        else:
            raise ValueError("Unknown rho function")

    def _get_rho_second_derivative(self, rho_func):
        """Return second derivative of rho function"""
        if rho_func == rho_exponential:
            return lambda v: -np.exp(v)
        elif rho_func == rho_cue:
            return lambda v: -np.ones_like(v)
        elif rho_func == rho_el:
            return lambda v: -1 / (1 - v) ** 2
        else:
            raise ValueError("Unknown rho function")

    def _compute_asymptotic_covariance(self):
        """Compute asymptotic covariance matrix using GEL theory"""
        moments = self.m(self.D_, self.est)  # n x q
        n, q = moments.shape
        p = len(self.est)  # number of parameters

        # Under correctly specified moments, EL/ET/CUE share the efficient
        # first-order covariance (G' Omega^-1 G)^-1 / n. G is q by p.
        # Do not replace a failed p-by-p covariance with a q-by-q moment matrix.
        from .gmm import numerical_jacobian, moment_covariance

        G = numerical_jacobian(lambda b: self.m(self.D_, b).mean(axis=0), self.est)
        omega = moment_covariance(moments)
        information = G.T @ np.linalg.solve(omega, G)
        if np.linalg.matrix_rank(information) < p:
            raise ValueError("GEL parameters are not locally identified")
        self.Sigma = np.linalg.inv(information) / n
        self.se = np.sqrt(np.diag(self.Sigma))

    def _compute_j_test(self):
        """Compute J-test for overidentifying restrictions"""
        moments = self.m(self.D_, self.est)
        n, q = moments.shape
        p = len(self.est)

        if q <= p:
            # Just identified or under-identified
            self.J_stat = None
            self.J_pvalue = None
            return

        # J-statistic: n * objective function value at optimum
        moment_avg = moments.mean(axis=0)
        tilts = np.dot(moments, self.lam_hat)

        # GEL J-statistic
        self.J_stat = 2 * n * np.sum(self.rho(tilts)) / n

        # Under null, J ~ chi2(q-p)
        df = q - p
        self.J_pvalue = 1 - chi2.cdf(self.J_stat, df)
