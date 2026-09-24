"""Reproduce selected before/after errors without checking out the old branch.

Run from repo root with its test extra installed:
    python benchmarks/correctness_cleanup_audit.py --output tmp/cleanup-diagnostics.json
The baseline sources are read from Git and executed only in isolated modules.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.special import log_ndtr
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trex import LinearRegression, LogisticRegression, PoissonRegression
from trex.choice import BinaryProbit
from trex.metrics import sliced_wasserstein_distance

BASE = "e6286e088913799dee52e334c95ea113645208c7"


def baseline_module(path, name, package):
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True
    )
    module = types.ModuleType(name)
    module.__package__ = package
    sys.modules[name] = module
    exec(compile(source, path, "exec"), module.__dict__)
    return module


def run():
    d = pd.read_csv(ROOT / "tests/data/r_regression_input.csv")
    x = sm.add_constant(d[["x1", "x2"]]).to_numpy()
    y = d.y.to_numpy()
    w = d.w.to_numpy()
    old = baseline_module("trex/linear.py", "trex._baseline_linear", "trex")
    result = {}
    for use_fe in [False, True]:
        fe = [d.unit.to_numpy(), d.period.to_numpy()] if use_fe else None
        design = (
            x
            if not use_fe
            else np.column_stack(
                [
                    x,
                    pd.get_dummies(d.unit, drop_first=True),
                    pd.get_dummies(d.period, drop_first=True),
                ]
            )
        )
        ref = sm.WLS(y, design, weights=w).fit(cov_type="HC1")
        sl = slice(1, 3) if use_fe else slice(None)
        scores = {}
        for name, cls in [
            ("before", old.LinearRegression),
            ("after", LinearRegression),
        ]:
            m = cls(device="cpu").fit(
                torch.tensor(x),
                torch.tensor(y),
                weights=torch.tensor(w),
                fe=fe,
                se="HC1",
            )
            scores[name] = {
                "coef_max_abs_error": float(
                    np.max(
                        abs(m.params["coef"][sl].numpy() - np.asarray(ref.params)[sl])
                    )
                ),
                "se_max_abs_error": float(
                    np.max(abs(m.params["se"][sl].numpy() - np.asarray(ref.bse)[sl]))
                ),
            }
        result["weighted_fe" if use_fe else "weighted_ols"] = scores
    oldchoice = baseline_module(
        "trex/choice/static.py", "trex.choice._baseline_static", "trex.choice"
    )
    x1 = torch.ones((1, 1), dtype=torch.float64)
    y1 = torch.zeros(1, dtype=torch.float64)
    result["probit_rare_event"] = {"scipy_nll": float(-log_ndtr(-40.0))}
    for name, cls in [("before", oldchoice.BinaryProbit), ("after", BinaryProbit)]:
        b = torch.tensor([40.0], dtype=torch.float64, requires_grad=True)
        loss = cls(device="cpu")._negative_log_likelihood(b, x1, y1)
        loss.backward()
        result["probit_rare_event"][name] = {
            "nll": float(loss.detach()),
            "gradient": float(b.grad[0]),
        }
    a = np.array([[0.0], [10.0]])
    b = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0]])
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:trex/simdgp.py"], cwd=ROOT, text=True
    )
    namespace = {"np": np}
    exec(source[source.index("def sliced_wasserstein_distance(") :], namespace)
    result["unequal_sample_w1"] = {
        "before": namespace["sliced_wasserstein_distance"](a, b),
        "after": sliced_wasserstein_distance(a, b),
        "scipy": wasserstein_distance(a[:, 0], b[:, 0]),
    }
    r = json.loads((ROOT / "tests/data/r_regression_reference.json").read_text())
    for name, cls, outcome in [
        ("logistic", LogisticRegression, "binary"),
        ("poisson", PoissonRegression, "count"),
    ]:
        m = cls(device="cpu", tol=1e-10, maxiter=100).fit(
            torch.tensor(x), torch.tensor(d[outcome].to_numpy(dtype=float))
        )
        result[name] = {
            "r_coefficient_max_abs_error": float(
                np.max(abs(m.params["coef"].numpy() - r[name]["coef"]))
            ),
            "r_covariance_max_abs_error": float(
                np.max(abs(m.params["vcov"].numpy() - r[name]["covariance"]))
            ),
        }
    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "numpy",
            "scipy",
            "statsmodels",
            "linearmodels",
            "scikit-learn",
            "pytest",
        ]
    }
    result["baseline"] = BASE
    result["r_provenance"] = r["provenance"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
