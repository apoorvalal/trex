#!/usr/bin/env python3
"""Matched device benchmarks; see benchmarks/DEVICE_BENCHMARKS.md.

Generate data once, copy the same directory to each host, then run.
The parent isolates cases in subprocesses; imports and data I/O are not timed.
No estimator implementation is replaced or patched by this harness.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

THREADS = 8
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = str(THREADS)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTHONHASHSEED"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL,
                                       timeout=15).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def specification():
    cases = []
    shapes = {
        "wls": [(1000, 8), (20000, 32), (200000, 64)],
        "twfe": [(2000, 8), (20000, 16), (200000, 32)],
        "binary_logit": [(1000, 8), (20000, 32), (200000, 64)],
        "poisson": [(1000, 8), (20000, 32), (200000, 64)],
        "multinomial": [(1000, 6, 4), (20000, 12, 5), (200000, 24, 6)],
        "gmm": [(1000, 6, 10), (20000, 12, 18), (200000, 24, 32)],
        "mmr": [(256, 3), (1024, 3), (4096, 3)],
        "smd": [(1000, 6), (20000, 6), (200000, 6)],
        "neural_smd": [(1000, 6), (10000, 6), (50000, 6)],
        "matrix_completion": [(64, 40), (256, 120), (1024, 240)],
        "latent_gaussian": [(64, 40), (256, 120), (1024, 240)],
        "sdid": [(100, 50), (500, 120), (2000, 240)],
        "nfxp": [(2000, 16), (10000, 48), (50000, 128)],
        "hotz_miller": [(2000, 16), (10000, 48), (50000, 128)],
    }
    for family, sizes in shapes.items():
        for i, shape in enumerate(sizes):
            cases.append({"id": f"{family}_{i+1}", "family": family,
                          "size": ["small", "medium", "large"][i],
                          "shape": list(shape), "seed": 8270 + i,
                          "dtype": "float64" if family in
                          {"twfe", "gmm", "nfxp", "hotz_miller"} else "float32",
                          "budget": "fixed_steps" if family in
                          {"neural_smd", "latent_gaussian"} else "tolerance"})
    return {"schema": 1, "threads": THREADS, "warmups": 1, "repeats": 3,
            "tf32": False, "amp": False, "mps_fallback": False,
            "timing": "in-memory host input through construction, transfer, fit and CPU result copy",
            "cases": cases}


def generate(directory):
    import numpy as np
    from scipy.special import expit, logsumexp, softmax
    directory.mkdir(parents=True, exist_ok=True)
    spec = specification()
    for c in spec["cases"]:
        rng = np.random.default_rng(c["seed"])
        f, shape = c["family"], c["shape"]
        dtype = np.dtype(c["dtype"])
        a = {}
        if f in {"wls", "twfe", "binary_logit", "poisson", "multinomial", "gmm"}:
            n, p = shape[:2]
            x = rng.normal(size=(n, p))
            x[:, 0] = 1
            beta = rng.normal(size=p) / np.sqrt(p)
            eta = x @ beta
            a.update(X=x, beta=beta, init=np.zeros(p))
            if f in {"wls", "twfe"}:
                a.update(y=eta + rng.normal(size=n), weights=np.exp(0.2*x[:, 1]))
                if f == "twfe":
                    a["fe"] = np.c_[np.arange(n)//20, np.arange(n)%20].astype(np.int64)
                    a["y"] += rng.normal(size=n//20)[a["fe"][:, 0]]
                    a["y"] += rng.normal(size=20)[a["fe"][:, 1]]
            elif f == "binary_logit":
                a["y"] = rng.binomial(1, expit(eta))
            elif f == "poisson":
                a["beta"] *= 0.4
                a["y"] = rng.poisson(np.exp(x @ a["beta"]))
            elif f == "multinomial":
                j = shape[2]
                b = rng.normal(size=(p, j-1)) / np.sqrt(p)
                probs = softmax(np.c_[x @ b, np.zeros(n)], axis=1)
                chosen = (rng.random(n)[:, None] > probs.cumsum(axis=1)).sum(axis=1)
                a.update(y=np.eye(j)[chosen], beta=b, init=np.zeros((p, j-1)))
            else:
                z = rng.normal(size=(n, shape[2])); z[:, 0] = 1
                u = rng.normal(size=n)
                x = z[:, :p] + 0.3*rng.normal(size=(n, p)); x[:, 0] = 1
                x[:, 1] += u
                a.update(X=x, Z=z, y=x @ beta + 0.5*u + rng.normal(size=n))
        elif f in {"mmr", "smd", "neural_smd"}:
            n, p = shape
            z = rng.normal(size=(n, 3)); u = rng.normal(size=n)
            t = 0.8*z[:, 0] + 0.3*z[:, 1] + 0.5*u
            x = np.c_[np.ones(n), t, t*t, z[:, 1], z[:, 2], t*z[:, 1]][:, :p]
            beta = np.array([0.5, 0.4, 0.15, -0.2, 0.3, 0.1])[:p]
            a.update(X=x, Z=z, y=x@beta + 0.4*u + rng.normal(size=n), beta=beta)
            if f == "neural_smd":
                a["init_0"] = rng.normal(scale=1/np.sqrt(p), size=(64, p))
                a["init_1"] = np.zeros(64)
                a["init_2"] = rng.normal(scale=1/8, size=(1, 64))
                a["init_3"] = np.zeros(1)
        elif f in {"matrix_completion", "latent_gaussian", "sdid"}:
            n, t = shape
            factor = rng.normal(size=(n, 3)) @ rng.normal(size=(3, t)) / np.sqrt(3)
            signal = factor + rng.normal(size=(n, 1)) + rng.normal(size=(1, t))
            if f == "latent_gaussian":
                x = rng.normal(size=(n, t, 3)); beta = np.array([0.5, -0.3, 0.2])
                signal = factor + x @ beta
                a.update(X=x, beta=beta)
            a["Y"] = signal + 0.2*rng.normal(size=(n, t))
            a["signal"] = signal
            mask = rng.random((n, t)) < 0.8; mask[:, 0] = True; mask[0, :] = True
            a["mask"] = mask
            if f == "sdid":
                a["Y"][int(0.9*n):, int(0.75*t):] += 1.0
        else:
            n, s = shape
            trans = np.zeros((s, 2, s))
            for k in range(s):
                trans[k, 0, k] += 0.25
                trans[k, 0, min(k+1, s-1)] += 0.75
                trans[k, 1, :2] = [0.8, 0.2]
            beta = np.array([3.0/s, 2.0])
            utility = np.c_[-beta[0]*np.arange(s), np.full(s, -beta[1])]
            value = np.zeros(s)
            for _ in range(1000):
                new = logsumexp(utility + 0.9*np.einsum("sat,t->sa", trans, value), axis=1)
                if np.max(np.abs(new-value)) < 1e-12:
                    value = new; break
                value = new
            probs = softmax(utility + 0.9*np.einsum("sat,t->sa", trans, value), axis=1)
            states = rng.integers(s, size=n)
            actions = (rng.random(n) < probs[states, 1]).astype(np.int64)
            a.update(P=trans, states=states, actions=actions, beta=beta,
                     init=np.array([2.4/s, 1.5]))
        for name, value in a.items():
            if name not in {"fe", "mask", "states", "actions"}:
                a[name] = np.asarray(value, dtype=dtype)
        path = directory / (c["id"] + ".npz")
        np.savez(path, **a)
        c["input_sha256"] = sha(path)
        c["input_bytes"] = path.stat().st_size
        print("generated", c["id"], flush=True)
    (directory / "manifest.json").write_text(json.dumps(spec, indent=2)+"\n")


def setup_torch():
    import torch
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def metadata(torch, host):
    files = sorted((ROOT / "trex").rglob("*.py"))
    code = hashlib.sha256()
    for p in files:
        code.update(p.relative_to(ROOT).as_posix().encode()); code.update(p.read_bytes())
    return {"host": host, "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpus": os.cpu_count(), "threads": THREADS,
            "cpu": command("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin"
                   else command("lscpu"),
            "memory": command("sysctl", "-n", "hw.memsize") if sys.platform == "darwin"
                      else command("free", "-b"),
            "gpu": command("system_profiler", "SPDisplaysDataType", "-json") if sys.platform == "darwin"
                   else command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
            "python": platform.python_version(),
            "versions": {n: importlib.metadata.version(n) for n in
                         ["torch", "numpy", "scipy", "pandas", "pytorch-minimize"]},
            "torch_config": torch.__config__.show(), "cuda": torch.version.cuda,
            "mps_available": torch.backends.mps.is_available(),
            "source_commit": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
            "implementation_sha256": code.hexdigest(), "runner_sha256": sha(__file__),
            "mps_fallback": False, "tf32": False,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def synchronize(torch, device):
    if device == "cuda": torch.cuda.synchronize()
    elif device == "mps": torch.mps.synchronize()


def prepare(torch, c, arrays, device):
    from trex import (LinearRegression, PoissonRegression, MaximumMomentRestriction,
                      SieveMinimumDistance, NuclearNormMatrixCompletion, LatentFactorGLM,
                      SyntheticDID)
    from trex.choice import BinaryLogit, MultinomialLogit, RustNFP, HotzMillerCCP, ReplacementUtility
    from trex.gmm.gmm import GMMEstimator, GMMEstimatorTorch
    f = c["family"]
    needed = {"X", "y", "weights", "fe", "init", "Z", "Y", "mask", "P", "states", "actions"}
    if f == "gmm": needed = set()  # Public fit performs its own host-to-device conversion.
    a = {k: torch.as_tensor(v, device=device) for k, v in arrays.items() if k in needed}
    if f in {"wls", "twfe"}:
        model = LinearRegression(device=device)
        options = {"weights": a["weights"], "se": "HC1"}
        if f == "twfe":
            options.update(fe=a["fe"], df_absorbed=c["shape"][0]//20+19, tol=1e-9)
        fit = lambda: model.fit(a["X"], a["y"], **options)
    elif f in {"binary_logit", "poisson", "multinomial"}:
        cls = {"binary_logit": BinaryLogit, "poisson": PoissonRegression,
               "multinomial": MultinomialLogit}[f]
        model = cls(maxiter=60, tol=1e-7, device=device)
        fit = lambda: model.fit(a["X"], a["y"], init_params=a["init"])
    elif f == "gmm":
        model = GMMEstimator(GMMEstimatorTorch.iv_moment, backend="torch", device=device)
        # Public GMM.fit validates/coerces host arrays itself, including transfers.
        fit = lambda: model.fit(arrays["Z"], arrays["y"], arrays["X"], maxiter=200, tol=1e-8)
    elif f in {"mmr", "smd", "neural_smd"}:
        p = c["shape"][1]
        if f == "neural_smd":
            net = torch.nn.Sequential(torch.nn.Linear(p, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
            for j, param in enumerate(net.parameters()):
                param.data.copy_(torch.from_numpy(arrays[f"init_{j}"]))
            opts = {"optimizer": torch.optim.Adam, "maxiter": 60, "lr": 0.01}
        else:
            net = torch.nn.Linear(p, 1, bias=False)
            torch.nn.init.zeros_(net.weight)
            opts = {"maxiter": 100, "tolerance_grad": 1e-6, "tolerance_change": 1e-9}
        cls = MaximumMomentRestriction if f == "mmr" else SieveMinimumDistance
        extra = {"bandwidth": 1.0} if f == "mmr" else {"degree": 3, "ridge": 1e-6}
        model = cls(net, lambda prediction, y: prediction-y, device=device,
                    dtype=torch.float32, **opts, **extra)
        fit = lambda: model.fit(a["X"], a["y"], a["Z"])
    elif f == "matrix_completion":
        model = NuclearNormMatrixCompletion(lambda_fraction=0.1, maxiter=80, tol=1e-5, device=device)
        fit = lambda: model.fit(a["Y"], mask=a["mask"])
    elif f == "latent_gaussian":
        model = LatentFactorGLM(family="gaussian", rank=3, penalty=0.001,
                               optimizer_kwargs={"lr": 0.02}, maxiter=120, tol=0, device=device)
        fit = lambda: model.fit(a["X"], a["Y"], mask=a["mask"])
    elif f == "sdid":
        n, t = c["shape"]
        model = SyntheticDID(maxiter=1000, min_decrease=1e-4, device=device)
        fit = lambda: model.fit(a["Y"], int(0.9*n), int(0.75*t))
    else:
        cls = RustNFP if f == "nfxp" else HotzMillerCCP
        model = cls(c["shape"][1], 2, 0.9, maxiter=40, tol=1e-7, device=device)
        model.set_transition_probabilities(a["P"])
        model.set_flow_utility(ReplacementUtility(device=device))
        fit = lambda: model.fit({"states": a["states"], "actions": a["actions"]}, init_params=a["init"])
    return model, fit, a


def snapshot(torch, model, c, a):
    """Copy fitted quantities to CPU; included in end-to-end timing."""
    f = c["family"]
    out = {}
    if f == "gmm":
        out = {"coef": model.theta_.copy(), "se": model.std_errors_.copy()}
    elif f in {"mmr", "smd", "neural_smd"}:
        if f != "neural_smd": out["coef"] = model.model.weight.detach().cpu().numpy().ravel().copy()
        else:
            out["weights"] = torch.cat([p.detach().flatten() for p in model.model.parameters()]).cpu().numpy().copy()
    elif f == "matrix_completion":
        out["fitted"] = model.result_.completed.cpu().numpy().copy()
    elif f == "latent_gaussian":
        for key in ["coef", "unit_factors", "time_factors"]:
            out[key] = model.params[key].detach().cpu().numpy().copy()
    elif f == "sdid":
        for key in ["estimate", "omega", "lambda"]:
            out[key] = model.params[key].detach().cpu().numpy().copy()
    else:
        for key in ["coef", "se"]:
            if key in model.params: out[key] = model.params[key].detach().cpu().numpy().copy()
    return out


def diagnostics(torch, model, c, a, out, arrays):
    import numpy as np
    from scipy.special import expit, logsumexp, softmax
    f = c["family"]
    metrics = {}
    iterations = getattr(model, "iterations_run", None)
    if f in {"wls", "twfe"}:
        residual = model.residuals_.detach().cpu().numpy().astype(float)
        metrics["weighted_rmse"] = float(np.sqrt(np.average(residual**2, weights=arrays["weights"])))
        iterations = 1
    elif f in {"binary_logit", "poisson", "multinomial"}:
        x, y, b = arrays["X"].astype(float), arrays["y"].astype(float), out["coef"].astype(float)
        if f == "multinomial":
            eta = np.c_[x@b, np.zeros(len(x))]
            logp = eta-logsumexp(eta, axis=1)[:, None]
            metrics["loss"] = float(-(y*logp).sum()/len(x))
            gradient = x.T @ (softmax(eta, axis=1)-y)[:, :-1]/len(x)
        else:
            eta = x@b
            mu = expit(eta) if f == "binary_logit" else np.exp(eta)
            metrics["loss"] = float(np.mean(np.logaddexp(0, eta)-y*eta) if f == "binary_logit"
                                    else np.mean(mu-y*eta))
            gradient = x.T@(mu-y)/len(x)
        metrics["score_inf"] = float(np.max(np.abs(gradient)))
    elif f == "gmm":
        metrics["loss"] = float(model.result_.fun)
        p = model._convert(model.theta_).requires_grad_(True)
        gradient = torch.autograd.grad(model.gmm_objective(p), p)[0]
        metrics["score_inf"] = float(gradient.detach().abs().max().cpu())
        iterations = getattr(model.result_, "nit", None)
    elif f in {"mmr", "smd", "neural_smd"}:
        pred = model.predict(a["X"])
        r = pred-a["y"][:, None]
        if f == "mmr":
            from trex.cmr import mmr_loss
            criterion = mmr_loss(r, a["Z"], bandwidth=1.0)
        else:
            m = model.basis_.T@r/len(r)
            criterion = torch.trace(m.T@model.weighting_matrix_@m)
        grad = torch.autograd.grad(criterion, list(model.model.parameters()))
        metrics["score_inf"] = max(float(g.abs().max().cpu()) for g in grad)
        metrics["loss"] = float(criterion.detach().cpu())
        if f == "neural_smd":
            out["prediction"] = pred.detach().cpu().numpy()[::max(1,len(pred)//1024)].ravel().copy()
            iterations = 60
        else: iterations = None  # LBFGS's actual inner step count is not exposed.
    elif f in {"matrix_completion", "latent_gaussian"}:
        if f == "matrix_completion":
            fitted = out["fitted"]
            metrics["loss"] = float(model.result_.objective)
            iterations = int(model.result_.iterations)
        else:
            fitted = arrays["X"]@out["coef"] + out["unit_factors"]@out["time_factors"].T
            metrics["loss"] = model.history["loss"][-1]
        heldout = ~arrays["mask"]
        metrics["heldout_rmse"] = float(np.sqrt(np.mean((fitted[heldout]-arrays["Y"][heldout])**2)))
        out["prediction"] = fitted.ravel()[::max(1,fitted.size//1024)].copy()
        out = {k:v for k,v in out.items() if k in {"coef","prediction"}}
    elif f == "sdid":
        metrics["att"] = float(out["estimate"])
        metrics["omega_objective"] = float(model.result_.omega_values[-1])
        metrics["lambda_objective"] = float(model.result_.lambda_values[-1])
        iterations = {"omega": len(model.result_.omega_values), "lambda": len(model.result_.lambda_values)}
    else:
        coefficient = model.params["coef"].clone().requires_grad_(True)
        criterion = model._negative_log_likelihood(coefficient, {"states": a["states"], "actions": a["actions"]})/len(a["states"])
        grad = torch.autograd.grad(criterion, coefficient)[0]
        metrics["loss"] = float(criterion.detach().cpu())
        metrics["score_inf"] = float(grad.abs().max().cpu())
    output = {key: np.asarray(value).ravel().tolist() for key,value in out.items() if key != "weights"}
    finite = all(np.isfinite(v).all() for v in output.values()) and all(np.isfinite(v) for v in metrics.values())
    return {"metrics": metrics, "iterations": iterations, "finite": bool(finite), "solution": output}


def worker(args):
    import numpy as np
    torch = setup_torch()
    spec = json.loads((args.data/"manifest.json").read_text())
    c = next(c for c in spec["cases"] if c["id"] == args.case)
    path = args.data/(args.case+".npz")
    if sha(path) != c["input_sha256"]: raise RuntimeError("Input checksum mismatch")
    with np.load(path) as f: arrays = {k:f[k] for k in f.files}
    rows = []
    for rep in range(-args.warmups, args.repeats):
        gc.collect()
        torch.manual_seed(192)
        synchronize(torch, args.device)
        if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        model, fit, tensors = prepare(torch, c, arrays, args.device)
        synchronize(torch, args.device)
        prepared = time.perf_counter()
        fit()
        synchronize(torch, args.device)
        fitted = time.perf_counter()
        out = snapshot(torch, model, c, tensors)
        synchronize(torch, args.device)
        ended = time.perf_counter()
        peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        details = diagnostics(torch, model, c, tensors, out, arrays)
        rows.append({"repeat": rep, "warmup": rep < 0, "prepare_s": prepared-started,
                     "fit_s": fitted-prepared, "end_to_end_s": ended-started,
                     "cuda_peak_allocated_bytes": peak,
                     **details})
        del model, fit, tensors, out
    return {"status": "ok", "case": c, "device": args.device, "runs": rows}


def run(args):
    torch = setup_torch()
    spec = json.loads((args.data/"manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    meta = metadata(torch, args.host)
    (args.output/(args.host+"-metadata.json")).write_text(json.dumps(meta, indent=2)+"\n")
    for device in args.devices.split(","):
        for c in spec["cases"]:
            if args.only and c["id"] not in args.only.split(","): continue
            dest = args.output/f"{args.host}-{device}-{c['id']}.json"
            if dest.exists() and not args.overwrite: continue
            cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--data", str(args.data),
                   "--case", c["id"], "--device", device, "--repeats", str(args.repeats),
                   "--warmups", str(args.warmups)]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
                if result.returncode == 0:
                    record = json.loads(result.stdout.split("BENCH_RESULT=")[-1])
                else:
                    record = {"status": "error", "case": c, "device": device,
                              "error": result.stderr[-10000:], "stdout": result.stdout[-2000:]}
            except subprocess.TimeoutExpired:
                record = {"status": "timeout", "case": c, "device": device,
                          "timeout_s": args.timeout}
            record.update(host=args.host, runner_sha256=meta["runner_sha256"],
                          implementation_sha256=meta["implementation_sha256"],
                          timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            dest.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
            times = [r["end_to_end_s"] for r in record.get("runs",[]) if not r["warmup"]]
            print(args.host, device, c["id"], record["status"], times, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    g = sub.add_parser("generate"); g.add_argument("--data", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", type=Path, required=True); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--host", required=True); r.add_argument("--devices", required=True)
    r.add_argument("--only"); r.add_argument("--overwrite", action="store_true")
    r.add_argument("--timeout", type=int, default=240)
    r.add_argument("--repeats", type=int, default=3); r.add_argument("--warmups", type=int, default=1)
    w = sub.add_parser("worker"); w.add_argument("--data", type=Path, required=True)
    w.add_argument("--case", required=True); w.add_argument("--device", required=True)
    w.add_argument("--repeats", type=int, default=3); w.add_argument("--warmups", type=int, default=1)
    args = parser.parse_args()
    if args.action == "generate": generate(args.data)
    elif args.action == "run": run(args)
    else: print("BENCH_RESULT="+json.dumps(worker(args), allow_nan=False))


if __name__ == "__main__":
    main()
