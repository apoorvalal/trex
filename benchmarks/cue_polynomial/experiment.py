"""Reproducible scalar CUE optimizer comparisons; see the executed notebook.

Run from the repository root:
    python -m benchmarks.cue_polynomial.experiment --replications 40
Outputs are JSON (no API calls or private data).
"""

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import minimize, minimize_scalar
import torch

from trex.gmm import ScalarIVCUE
from trex.gmm.gmm import numerical_jacobian

ROOT = Path(__file__).resolve().parent
CONFIGS = [(1, 1), (4, 1), (4, 4), (4, 16), (10, 4), (10, 64)]


def generate(seed, k, concentration, n=800):
    """Paper Section 5.1 DGP; mu²=64 is an additional stronger-IV check."""
    rng = np.random.default_rng(seed)
    z1, v, e1, e2 = rng.normal(size=(4, n))
    z = z1[:, None] if k == 1 else np.column_stack([z1**j for j in range(1, 5)])
    if k > 4:
        z = np.column_stack([z, rng.binomial(1, 0.5, size=(n, k - 4)) * z1[:, None]])
    x = np.sqrt(concentration / n) * z1 + v
    y = 5 * x + 0.6 * v + 0.8 * (np.sqrt(0.5) * z1 * e1 + np.sqrt(0.5) * e2)
    z, y, x = z - z.mean(0), y - y.mean(), x - x.mean()
    projection = z @ np.linalg.lstsq(z, x, rcond=None)[0]
    tsls = float(projection @ y / (projection @ x))
    return z, y, x, tsls


def independent_reference(problem, points=2049):
    """Direct angular matrix criterion + refinement of every grid local minimum.

    Deliberately independent of polynomial coefficients/roots. This is a dense
    numerical cross-check, not an exact certificate or an estimator benchmark.
    """
    a, b, S = problem.a, problem.b, problem.sigma
    k = problem.k

    def values(theta):
        theta = np.asarray(theta)
        c, s = np.cos(theta), np.sin(theta)
        g = c[..., None] * a - s[..., None] * b
        om = (
            c[..., None, None] ** 2 * S[:k, :k]
            - (c * s)[..., None, None] * (S[:k, k:] + S[k:, :k])
            + s[..., None, None] ** 2 * S[k:, k:]
        )
        return np.einsum("...i,...i->...", g, np.linalg.solve(om, g[..., None])[..., 0])

    angles = np.linspace(-np.pi / 2, np.pi / 2, points)
    grid = values(angles)
    candidates = [(float(grid[0]), np.inf)]
    local = np.flatnonzero((grid[1:-1] <= grid[:-2]) & (grid[1:-1] <= grid[2:])) + 1
    for j in local:
        r = minimize_scalar(
            values,
            bounds=(angles[j - 1], angles[j + 1]),
            method="bounded",
            options={"xatol": 1e-14},
        )
        candidates.append((float(r.fun), float(np.tan(r.x))))
    return min(candidates)


def bfgs(problem, start, *, analytic=False, trace=False, maxiter=2000):
    n = problem.n
    begin = perf_counter()
    history = []

    def record(beta):
        value, gradient, _ = problem.value_derivatives(float(beta[0]))
        history.append(
            dict(
                seconds=perf_counter() - begin,
                beta=float(beta[0]),
                objective=value,
                gradient=gradient,
            )
        )

    fun = lambda beta: problem.value(beta[0]) / n
    jac = (
        (lambda beta: np.array([problem.value_derivatives(beta[0])[1] / n]))
        if analytic
        else (
            lambda beta: numerical_jacobian(
                lambda v: np.atleast_1d(fun(v)), beta
            ).ravel()
        )
    )
    if trace:
        record([start])
    result = minimize(
        fun,
        [start],
        jac=jac,
        method="BFGS",
        tol=1e-9,
        options={"maxiter": maxiter},
        callback=record if trace else None,
    )
    elapsed = perf_counter() - begin
    value, gradient, _ = problem.value_derivatives(result.x[0])
    return dict(
        beta=float(result.x[0]),
        objective=value,
        success=bool(result.success),
        message=str(result.message),
        gradient=gradient,
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        seconds=elapsed,
        trace=history,
    )


def adam_batch(problems, starts, *, lr=0.05, maxiter=3000, trace=False):
    """Independent full-sample Adam fits batched for Monte Carlo throughput.

    Sum (not average) over fit objectives so each gradient/Adam epsilon matches
    an individual optimizer. A fit freezes when |d(Q/n)/dbeta|<=1e-9. Gradients
    cannot certify global convergence; the benchmark separately checks gaps.
    No mini-batching observations. Timings are BATCH time, not serial latency.
    """
    count = len(problems)
    dtype = torch.float64
    a = torch.tensor(np.array([p.g0 for p in problems]), dtype=dtype)
    b = torch.tensor(np.array([p.g1 for p in problems]), dtype=dtype)
    A = torch.tensor(np.array([p.A for p in problems]), dtype=dtype)
    B = torch.tensor(np.array([p.B for p in problems]), dtype=dtype)
    C = torch.tensor(np.array([p.C for p in problems]), dtype=dtype)
    center = torch.tensor([p.center for p in problems], dtype=dtype)
    scale = torch.tensor([p.scale for p in problems], dtype=dtype)
    n = torch.tensor([p.n for p in problems], dtype=dtype)
    beta = torch.tensor(starts, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=lr, foreach=False)
    iterations = np.full(count, maxiter)
    active = torch.ones(count, dtype=torch.bool)
    history = []
    begin = perf_counter()

    def criterion():
        t = (beta - center) / scale
        norm = torch.sqrt(1 + t * t)
        c, s = 1 / norm, t / norm
        g = c[:, None] * a - s[:, None] * b
        omega = (
            c[:, None, None] ** 2 * A
            - (c * s)[:, None, None] * B
            + s[:, None, None] ** 2 * C
        )
        v = torch.linalg.solve(omega, g[:, :, None])[:, :, 0]
        return (g * v).sum(1) / n

    for iteration in range(maxiter + 1):
        optimizer.zero_grad()
        losses = criterion()
        losses.sum().backward()
        done = active & (beta.grad.abs() <= 1e-9)
        iterations[done.numpy()] = iteration
        active = active & ~done
        if trace and (
            iteration < 20
            or iteration % 10 == 0
            or not active.any()
            or iteration == maxiter
        ):
            history.append(
                dict(
                    iteration=iteration,
                    seconds=perf_counter() - begin,
                    beta=beta.detach().tolist(),
                    objective=(losses.detach() * n).tolist(),
                )
            )
        if not active.any() or iteration == maxiter:
            break
        old = beta.detach().clone()
        beta.grad[~active] = 0
        optimizer.step()
        with torch.no_grad():
            beta[~active] = old[~active]
    elapsed = perf_counter() - begin
    fits = []
    for j, (p, v) in enumerate(zip(problems, beta.detach().numpy())):
        value, gradient, _ = p.value_derivatives(v)
        fits.append(
            dict(
                beta=float(v),
                objective=value,
                gradient=gradient,
                success=bool(not active[j]),
                iterations=int(iterations[j]),
                seconds_amortized=elapsed / count,
            )
        )
    return fits, history, elapsed


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(replications=40):
    torch.set_num_threads(1)
    rows, problems_metadata, batches = [], [], []
    start_labels = ["2SLS", "zero", "truth (infeasible)"]
    # Include a raw-moment bridge to the existing GEL criterion as a separate panel.
    configurations = [("reduced_form", k, mu) for k, mu in CONFIGS] + [("raw", 4, 4)]
    started = perf_counter()
    for design, (weighting, k, mu) in enumerate(configurations):
        batch_problems, batch_starts, batch_metadata = [], [], []
        for replication in range(replications):
            seed = 930000 + 10000 * design + replication
            z, y, x, tsls = generate(seed, k, mu)
            begin = perf_counter()
            p = ScalarIVCUE.from_data(z, y, x, demean=False, weighting=weighting)
            setup = perf_counter() - begin
            metadata = dict(
                weighting=weighting,
                k=k,
                concentration=mu,
                replication=replication,
                seed=seed,
            )
            reference, reference_beta = independent_reference(p)
            reference_metadata = dict(
                grid_reference=reference, grid_beta=reference_beta
            )
            try:
                poly = p.minimize_polynomial()
                gap = poly.fun - reference
                problems_metadata.append(
                    dict(
                        **metadata,
                        setup_seconds=setup,
                        polynomial=asdict(poly),
                        **reference_metadata,
                        polynomial_minus_grid=gap,
                    )
                )
                rows.append(
                    dict(
                        **metadata,
                        method="Polynomial",
                        start="none",
                        beta=poly.beta,
                        objective=poly.fun,
                        success=True,
                        iterations=None,
                        seconds=poly.elapsed_seconds,
                    )
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                problems_metadata.append(
                    dict(
                        **metadata,
                        setup_seconds=setup,
                        error=str(exc),
                        **reference_metadata,
                    )
                )
                rows.append(
                    dict(
                        **metadata,
                        method="Polynomial",
                        start="none",
                        beta=None,
                        objective=None,
                        success=False,
                        message=str(exc),
                    )
                )
            for label, start in zip(start_labels, [tsls, 0.0, 5.0]):
                for analytic in (False, True):
                    result = bfgs(p, start, analytic=analytic)
                    result.pop("trace")
                    rows.append(
                        dict(
                            **metadata,
                            method="BFGS analytic" if analytic else "BFGS Trex FD",
                            start=label,
                            **result,
                        )
                    )
                batch_problems.append(p)
                batch_starts.append(start)
                batch_metadata.append(dict(**metadata, start=label))
        for lr in [0.001, 0.05]:
            fits, _, elapsed = adam_batch(batch_problems, batch_starts, lr=lr)
            for meta, fit in zip(batch_metadata, fits):
                rows.append(dict(**meta, method=f"Adam lr={lr}", **fit))
            batches.append(
                dict(
                    weighting=weighting,
                    k=k,
                    concentration=mu,
                    lr=lr,
                    fits=len(fits),
                    seconds=elapsed,
                )
            )
        print(
            f"{weighting} k={k} mu2={mu}: {replications} paired datasets complete",
            flush=True,
        )
    environment = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        torch=torch.__version__,
        platform=platform.platform(),
        processor=platform.machine(),
        torch_threads=torch.get_num_threads(),
        dtype="float64",
        device="cpu",
    )
    return dict(
        environment=environment,
        replications=replications,
        n=800,
        beta_true=5.0,
        elapsed_seconds=perf_counter() - started,
        rows=rows,
        problems=problems_metadata,
        adam_batches=batches,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "results.json.gz")
    args = parser.parse_args()
    result = run(args.replications)
    payload = (
        json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n"
    ).encode()
    if args.output.suffix == ".gz":
        args.output.write_bytes(gzip.compress(payload, mtime=0))
    else:
        args.output.write_bytes(payload)
    print(args.output)
