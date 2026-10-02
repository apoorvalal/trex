"""Experimental scalar linear-IV CUE via polynomial root enumeration.

Implements the positive-definite, one-endogenous-regressor characterization in
Moreira, Newey and Sharifvaghefi (2026), arXiv:2609.36445v1, Sections 2–3.
This is a floating-point prototype, NOT an exact global-optimality certificate.
It does not replace GMMEstimator or GELEstimator, implement singular-covariance
extensions, multiple endogenous regressors, CLR, or weak-IV inference.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.polynomial import chebyshev as ch
from scipy.fft import dct
from scipy.linalg import block_diag, solve_triangular

from .gmm import moment_covariance


@dataclass
class PolynomialCUEResult:
    """Candidate enumeration result; ``beta=inf`` means no selected finite fit.

    ``constant`` denotes a numerically flat criterion, not identification.
    ``diagnostics`` records interpolation and stationarity residuals. Passing
    these checks is not proof that floating-point roots are complete.
    """

    beta: float
    fun: float
    candidates: np.ndarray
    values: np.ndarray
    at_infinity: bool
    constant: bool
    diagnostics: dict
    elapsed_seconds: float
    certified: bool = False


class ScalarIVCUE:
    """Scalar CUE objective ``(a-beta*b)' Omega(beta)^-1 (a-beta*b)``.

    Parameters
    ----------
    a, b : array_like, shape (k,)
        Sample moments ``Z'y/sqrt(n)`` and ``Z'x/sqrt(n)``.
    sigma : array_like, shape (2*k, 2*k)
        Fixed symmetric positive-definite joint moment covariance, in block
        order (outcome, endogenous regressor). No ridge is added automatically.

    Notes
    -----
    ``Omega(beta) = S11 - beta*(S12+S21) + beta**2*S22``. A covariance-based
    affine change of parameter and an invertible instrument transformation
    improve conditioning without changing the objective. Only the polynomial
    solver is capped at 12 instruments; objective evaluation has no such cap.
    No parameter standard errors or hypothesis-test calibration are provided.
    """

    def __init__(self, a, b, sigma):
        a, b, sigma = [np.asarray(v, dtype=float).copy() for v in (a, b, sigma)]
        if (
            a.ndim != 1
            or not len(a)
            or b.shape != a.shape
            or sigma.shape != (2 * len(a), 2 * len(a))
            or not all(np.isfinite(v).all() for v in (a, b, sigma))
        ):
            raise ValueError("Need finite a,b (k,) and sigma (2k,2k)")
        if not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-14):
            raise ValueError("sigma must be symmetric positive definite")
        sigma = (sigma + sigma.T) / 2
        try:
            np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "sigma must be positive definite; no automatic ridge"
            ) from exc
        self.a, self.b, self.sigma = a, b, sigma
        self.k = len(a)
        k = self.k
        eye = np.eye(k)
        self.center = np.trace(sigma[:k, k:]) / np.trace(sigma[k:, k:])
        transform = np.block([[eye, -self.center * eye], [np.zeros((k, k)), eye]])
        centered = transform @ sigma @ transform.T
        self.scale = np.sqrt(np.trace(centered[:k, :k]) / np.trace(centered[k:, k:]))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(
                "Parameter rescaling failed; covariance is ill-conditioned"
            )
        transform[k:, k:] *= self.scale
        normalized = transform @ sigma @ transform.T
        root = np.linalg.cholesky(normalized[:k, :k] + normalized[k:, k:])
        whitener = solve_triangular(root, eye, lower=True)
        transform = block_diag(whitener, whitener) @ transform
        normalized = transform @ sigma @ transform.T
        normalized = (normalized + normalized.T) / 2
        means = transform @ np.r_[a, b]
        self.g0, self.g1 = means[:k], means[k:]
        self.A = normalized[:k, :k]
        self.B = normalized[:k, k:] + normalized[k:, :k]
        self.C = normalized[k:, k:]
        self.normalized_condition = float(np.linalg.cond(normalized))

    @classmethod
    def from_data(
        cls,
        z,
        y,
        x,
        *,
        covariance="hc0",
        weighting="reduced_form",
        demean=True,
        max_lags=0,
        clusters=None,
    ):
        """Build the fixed joint covariance and scalar-IV sample moments.

        ``weighting='reduced_form'`` uses residuals from OLS of each column of
        (y,x) on Z, as in the paper's setup. ``weighting='raw'`` uses (y,x)
        itself: with HC0 this is exactly the structural-moment CUE criterion
        profiled by Trex's quadratic GEL, up to its factor 1/(2*n).

        ``covariance`` is 'hc0', Bartlett 'hac', or one-way 'cluster' (CR0,
        sum of cluster-score outer products / n; no small-sample correction).
        ``demean=True`` removes an intercept, not arbitrary included controls.
        Pass one scalar endogenous regressor; do not pass an intercept in Z
        when demeaning. Joint covariance singularities are rejected.
        """
        z, y, x = [np.asarray(v, dtype=float) for v in (z, y, x)]
        if (
            z.ndim != 2
            or not z.shape[1]
            or y.ndim != 1
            or x.shape != y.shape
            or len(z) != len(y)
            or len(y) < 3
            or not all(np.isfinite(v).all() for v in (z, y, x))
        ):
            raise ValueError("Expected finite z (n,k), y (n,), x (n,), n>=3")
        if weighting not in {"reduced_form", "raw"}:
            raise ValueError("weighting must be 'reduced_form' or 'raw'")
        if covariance not in {"hc0", "hac", "cluster"}:
            raise ValueError("covariance must be 'hc0', 'hac', or 'cluster'")
        if covariance != "hac" and max_lags != 0:
            raise ValueError("max_lags is only used with covariance='hac'")
        if covariance != "cluster" and clusters is not None:
            raise ValueError("clusters is only used with covariance='cluster'")
        Y = np.column_stack((y, x))
        if demean:
            z, Y = z - z.mean(axis=0), Y - Y.mean(axis=0)
        if np.linalg.matrix_rank(z) != z.shape[1]:
            raise ValueError("Instruments must have full column rank after demeaning")
        residual = (
            Y - z @ np.linalg.lstsq(z, Y, rcond=None)[0]
            if weighting == "reduced_form"
            else Y
        )
        scores = np.concatenate([z * residual[:, j, None] for j in range(2)], axis=1)
        if covariance == "cluster":
            clusters = np.asarray(clusters)
            if clusters.shape != (len(z),) or any(
                v is None or v != v for v in clusters
            ):
                raise ValueError("Need one nonmissing cluster label per observation")
            _, inverse = np.unique(clusters, return_inverse=True)
            sums = np.zeros((inverse.max() + 1, scores.shape[1]))
            np.add.at(sums, inverse, scores)
            sigma = sums.T @ sums / len(z)
        else:
            sigma = moment_covariance(scores, max_lags if covariance == "hac" else 0)
        means = z.T @ Y / np.sqrt(len(z))
        obj = cls(means[:, 0], means[:, 1], sigma)
        obj.n = len(z)
        obj.weighting = weighting
        obj.covariance_type = covariance
        return obj

    def _chart(self, t, reciprocal=False):
        """Value, first and second derivative in a bounded projective chart."""
        if reciprocal:
            g, dg = t * self.g0 - self.g1, self.g0
            omega = t * t * self.A - t * self.B + self.C
            d_omega, dd_omega = 2 * t * self.A - self.B, 2 * self.A
        else:
            g, dg = self.g0 - t * self.g1, -self.g1
            omega = self.A - t * self.B + t * t * self.C
            d_omega, dd_omega = -self.B + 2 * t * self.C, 2 * self.C
        v = np.linalg.solve(omega, g)
        residual = dg - d_omega @ v
        return (
            float(g @ v),
            float(2 * dg @ v - v @ d_omega @ v),
            float(2 * residual @ np.linalg.solve(omega, residual) - v @ dd_omega @ v),
        )

    def value_derivatives(self, beta):
        """Return Q, dQ/dbeta, d²Q/dbeta²; infinity has no beta derivatives."""
        beta = float(beta)
        if np.isnan(beta):
            raise ValueError("beta cannot be NaN")
        if np.isinf(beta):
            return self._chart(0, True)[0], np.nan, np.nan
        t = (beta - self.center) / self.scale
        if abs(t) <= 1:
            q, d, dd = self._chart(t)
            return q, d / self.scale, dd / self.scale**2
        r = 1 / t
        q, d, dd = self._chart(r, True)
        return q, -d * r * r / self.scale, (dd * r**4 + 2 * d * r**3) / self.scale**2

    def value(self, beta):
        """Evaluate the original matrix criterion, including the common ±∞ limit."""
        return self.value_derivatives(beta)[0]

    def _polynomials(self, reciprocal, tolerance):
        degree = 2 * self.k
        count = degree + 1
        nodes = np.cos(np.pi * (np.arange(count) + 0.5) / count)

        def samples(ts):
            if reciprocal:
                g = ts[:, None] * self.g0 - self.g1
                omega = (
                    ts[:, None, None] ** 2 * self.A
                    - ts[:, None, None] * self.B
                    + self.C
                )
            else:
                g = self.g0 - ts[:, None] * self.g1
                omega = (
                    self.A
                    - ts[:, None, None] * self.B
                    + ts[:, None, None] ** 2 * self.C
                )
            signs, logs = np.linalg.slogdet(omega)
            if np.any(signs <= 0):
                raise FloatingPointError("Non-positive determinant after conditioning")
            values = np.einsum(
                "ij,ij->i", g, np.linalg.solve(omega, g[..., None])[..., 0]
            )
            return logs, values

        logs, values = samples(nodes)
        shift = float(logs.max())
        q_values = np.exp(logs - shift)
        p = dct(q_values * values, type=2) / count
        q = dct(q_values, type=2) / count
        p[0] /= 2
        q[0] /= 2
        # Validate off the interpolation nodes. Tests also use independent oracles.
        probes = np.cos(np.pi * np.arange(2 * count + 1) / (2 * count))
        check_logs, check_values = samples(probes)
        q_check = ch.chebval(probes, q)
        if np.any(q_check <= 0):
            raise FloatingPointError("Interpolated denominator lost positivity")
        ratio_error = np.max(
            np.abs(ch.chebval(probes, p) / q_check - check_values)
            / (1 + np.abs(check_values))
        )
        denominator_error = np.max(np.abs(q_check / np.exp(check_logs - shift) - 1))
        if max(ratio_error, denominator_error) > tolerance:
            raise FloatingPointError(
                "Polynomial interpolation failed validation; reduce dimension or use higher precision"
            )
        left, right = ch.chebmul(ch.chebder(p), q), ch.chebmul(p, ch.chebder(q))
        h = ch.chebsub(left, right)
        # The coefficient of degree 4k-1 cancels analytically (paper, Section 3.2).
        h = h[: 4 * self.k - 1]
        cancellation_scale = max(
            np.max(np.abs(left)), np.max(np.abs(right)), np.finfo(float).tiny
        )
        flat = np.max(np.abs(h)) <= 128 * np.finfo(float).eps * cancellation_scale
        return h, flat, float(ratio_error), float(denominator_error)

    def minimize_polynomial(self, *, validation_tol=1e-7, root_tol=1e-7):
        """Interpolate p/q, enumerate real companion roots, compare with infinity.

        Two overlapping projective charts, t=(beta-center)/scale and 1/t,
        cover the entire real line with |chart coordinate|<=1. This avoids
        discarding remote finite roots or imposing parameter search bounds.
        NumPy's Chebyshev companion eigenproblem is the basis-conditioned
        counterpart of the paper's monomial companion matrix. Each candidate
        is polished using derivatives of the original matrix objective.

        The finite-precision checks are diagnostics, not a certificate. A
        failed interpolation/stationarity check raises FloatingPointError.
        The singular and multivariate algorithms in the paper are not included.
        """
        if self.k > 12:
            raise ValueError(
                "Experimental float64 polynomial solver supports at most 12 instruments"
            )
        if not all(np.isfinite(t) and 0 < t < 0.01 for t in (validation_tol, root_tol)):
            raise ValueError("Tolerances must be finite and between 0 and .01")
        start = perf_counter()
        # Paper Section 2: just identification has an exact zero-moment solution.
        # Avoid interpolating a squared linear factor with extreme dynamic range.
        if self.k == 1:
            constant = bool(self.a[0] == 0 and self.b[0] == 0)
            beta = (
                float(self.a[0] / self.b[0])
                if self.b[0] != 0
                else (self.center if constant else np.inf)
            )
            candidates = np.unique([beta, np.inf])
            values = np.array([self.value(t) for t in candidates])
            return PolynomialCUEResult(
                beta,
                self.value(beta),
                candidates,
                values,
                bool(np.isinf(beta)),
                constant,
                dict(
                    closed_form="just-identified",
                    charts=[],
                    normalized_condition=self.normalized_condition,
                ),
                perf_counter() - start,
            )
        candidates, diagnostics, flats = [], [], []
        for reciprocal in (False, True):
            h, flat, ratio_error, denominator_error = self._polynomials(
                reciprocal, validation_tol
            )
            flats.append(flat)
            roots = np.array([]) if flat else ch.chebroots(h)
            residuals = []
            for root in roots:
                if (
                    abs(root.imag) > 1e-6 * (1 + abs(root.real))
                    or abs(root.real) > 1 + 1e-7
                ):
                    continue
                t = float(root.real)
                for _ in range(15):
                    value, derivative, curvature = self._chart(t, reciprocal)
                    if abs(derivative) <= root_tol * (1 + abs(value)):
                        break
                    if abs(curvature) < np.finfo(float).eps:
                        break
                    trial = t - derivative / curvature
                    if not np.isfinite(trial) or abs(trial) > 1 + 1e-5:
                        break
                    t = float(trial)
                value, derivative, _ = self._chart(t, reciprocal)
                residual = abs(derivative) / (1 + abs(value))
                if residual > root_tol:
                    raise FloatingPointError(
                        "A companion root failed matrix-derivative validation"
                    )
                residuals.append(residual)
                beta = (
                    (self.center + self.scale / t if t != 0 else np.inf)
                    if reciprocal
                    else self.center + self.scale * t
                )
                if np.isfinite(beta):
                    candidates.append(float(beta))
            diagnostics.append(
                dict(
                    reciprocal=reciprocal,
                    degree=len(h) - 1,
                    ratio_error=ratio_error,
                    denominator_error=denominator_error,
                    max_stationarity_residual=max(residuals, default=0.0),
                )
            )
        constant = all(flats)
        if any(flats) and not constant:
            raise FloatingPointError(
                "Charts disagree about a numerically constant criterion"
            )
        if constant:
            candidates = [self.center]
        candidates = np.unique(np.r_[candidates, np.inf])
        values = np.array([self.value(beta) for beta in candidates])
        index = int(np.argmin(values))
        if constant:
            index = 0
        return PolynomialCUEResult(
            float(candidates[index]),
            float(values[index]),
            candidates,
            values,
            bool(np.isinf(candidates[index])),
            constant,
            dict(charts=diagnostics, normalized_condition=self.normalized_condition),
            perf_counter() - start,
        )
