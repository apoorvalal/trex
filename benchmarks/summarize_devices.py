#!/usr/bin/env python3
"""Validate, archive and summarize saved matched-device measurements.

This program does not rerun fits. The Quarto page calls the same analysis.
Agreement thresholds were declared in DEVICE_BENCHMARKS.md before final runs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["mac-cpu", "mac-mps", "fwk-cpu", "fwk-cuda"]
LABELS = {
    "wls": "Weighted least squares", "twfe": "Two-way fixed effects",
    "binary_logit": "Binary logit", "poisson": "Poisson",
    "multinomial": "Multinomial logit", "gmm": "Two-step IV GMM",
    "mmr": "Kernel MMR", "smd": "Linear sieve MD",
    "neural_smd": "Neural sieve MD", "matrix_completion": "Matrix completion",
    "latent_gaussian": "Gaussian latent factors", "sdid": "Synthetic DID",
    "nfxp": "NFXP", "hotz_miller": "Hotz–Miller",
}


def measured(record):
    return [r for r in record.get("runs", []) if not r["warmup"]]


def scaled_difference(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a-b)) / (1 + np.max(np.abs(b))))


def agreement(record, reference):
    """Worst comparison over every pair of measured repetitions."""
    if record["status"] != "ok" or reference["status"] != "ok":
        return None, None, None
    family = record["case"]["family"]
    keys = ({"estimate"} if family == "sdid" else {"prediction"} if family in
            {"matrix_completion", "latent_gaussian", "neural_smd"} else {"coef", "se"})
    solution_error = criterion_error = 0.0
    for r in measured(record):
        for ref in measured(reference):
            active = keys & r["solution"].keys() & ref["solution"].keys()
            assert active, f"No comparison target for {family}"
            solution_error = max(solution_error, *(scaled_difference(r["solution"][k], ref["solution"][k])
                                                   for k in active))
            metric_keys = {"loss", "weighted_rmse", "omega_objective", "lambda_objective"} & r["metrics"].keys()
            assert metric_keys
            criterion_error = max(criterion_error, *(scaled_difference(r["metrics"][k], ref["metrics"][k])
                                                     for k in metric_keys))
    tol = 1e-5 if record["case"]["dtype"] == "float64" else 1e-3
    return solution_error <= tol and criterion_error <= 1e-4, solution_error, criterion_error


def failure_reason(record):
    if record["status"] == "timeout": return "240 s whole-case timeout"
    error = record.get("error", "")
    if "doesn't support float64" in error: return "MPS has no float64"
    if "linalg_lstsq" in error: return "MPS least-squares operation unavailable"
    if "linalg_svd" in error: return "MPS SVD operation unavailable"
    return error.strip().splitlines()[-1] if error.strip() else record["status"]


def cap_hit(record):
    f = record["case"]["family"]
    cap = {"binary_logit": 60, "poisson": 60, "multinomial": 60,
           "matrix_completion": 80, "nfxp": 40, "hotz_miller": 40}.get(f)
    for r in measured(record):
        it = r["iterations"]
        if f == "sdid" and max(it.values()) >= 1000: return True
        if cap is not None and it is not None and it >= cap: return True
    return False


def summarize(bundle):
    records = bundle["records"]
    rows = []
    for c in bundle["manifest"]["cases"]:
        reference = records[f"mac-cpu-{c['id']}"]
        for backend in BACKENDS:
            record = records[f"{backend}-{c['id']}"]
            runs = measured(record)
            matched, solution_error, criterion_error = agreement(record, reference)
            row = {"case": c["id"], "family": c["family"], "size": c["size"],
                   "shape": " × ".join(map(str, c["shape"])), "dtype": c["dtype"],
                   "budget": c["budget"], "backend": backend, "status": record["status"],
                   "agreement_pass": matched, "solution_scaled_error": solution_error,
                   "criterion_scaled_error": criterion_error}
            if record["status"] == "ok":
                assert len(runs) == 3 and all(r["finite"] for r in runs)
                t = [r["end_to_end_s"] for r in runs]
                scores = [r["metrics"]["score_inf"] for r in runs if "score_inf" in r["metrics"]]
                row.update(median_s=statistics.median(t), min_s=min(t), max_s=max(t),
                           fit_median_s=statistics.median(r["fit_s"] for r in runs),
                           prepare_median_s=statistics.median(r["prepare_s"] for r in runs),
                           max_score=max(scores) if scores else None,
                           score_flag=bool(scores and max(scores) > 1e-4),
                           cap_hit=cap_hit(record),
                           iterations=json.dumps([r["iterations"] for r in runs], separators=(",", ":")),
                           reason="")
            else:
                row.update(median_s=None, min_s=None, max_s=None, fit_median_s=None,
                           prepare_median_s=None, max_score=None, score_flag=False,
                           cap_hit=False, iterations="", reason=failure_reason(record))
            rows.append(row)
    index = {(r["case"], r["backend"]): r for r in rows}
    for r in rows:
        host = r["backend"].split("-")[0]
        cpu = index[r["case"], host+"-cpu"]
        mac = index[r["case"], "mac-cpu"]
        ok = r["status"] == "ok" and r["agreement_pass"] is True
        r["same_host_speedup"] = (cpu["median_s"] / r["median_s"]
            if ok and cpu["agreement_pass"] is True else None)
        r["mac_cpu_runtime_ratio"] = (mac["median_s"] / r["median_s"]
            if ok and mac["agreement_pass"] is True else None)
    return rows


def validate(bundle):
    manifest, records, meta = bundle["manifest"], bundle["records"], bundle["metadata"]
    expected = {f"{b}-{c['id']}" for c in manifest["cases"] for b in BACKENDS}
    assert set(records) == expected, ("Missing", expected-set(records), "Extra", set(records)-expected)
    assert len(manifest["cases"]) == 42
    for key in ("source_commit", "implementation_sha256", "runner_sha256", "threads", "mps_fallback", "tf32"):
        assert meta["mac"][key] == meta["fwk"][key], key
    for name in ("torch", "numpy", "scipy", "pandas", "pytorch-minimize"):
        assert meta["mac"]["versions"][name].split("+")[0] == meta["fwk"]["versions"][name].split("+")[0], name
    for c in manifest["cases"]:
        for b in BACKENDS:
            record = records[f"{b}-{c['id']}"]
            assert record["case"] == c
            for key in ("runner_sha256", "implementation_sha256"):
                assert record[key] == meta[b.split("-")[0]][key]
            if record["status"] == "ok":
                assert len(record["runs"]) == 4 and len(measured(record)) == 3
                assert all(r["finite"] for r in record["runs"])
                for r in record["runs"]:
                    assert 0 < r["fit_s"] <= r["end_to_end_s"]
    return True


def load_bundle(path):
    with gzip.open(path, "rt") as f: bundle = json.load(f)
    validate(bundle)
    return bundle


def collect(args):
    manifest = json.loads((args.data/"manifest.json").read_text())
    records = {p.stem: json.loads(p.read_text()) for directory in args.results
               for p in directory.glob("*.json") if not p.name.endswith("-metadata.json")}
    meta = {host: json.loads(next(p for directory in args.results
                for p in directory.glob(f"{host}-metadata.json")).read_text()) for host in ("mac", "fwk")}
    if meta["mac"]["gpu"]:
        gpu = json.loads(meta["mac"]["gpu"])
        for item in gpu.get("SPDisplaysDataType", []):
            item.pop("spdisplays_ndrvs", None)  # Connected displays are not benchmark hardware.
        meta["mac"]["gpu"] = gpu
    bundle = {"schema": 1, "manifest": manifest, "metadata": meta, "records": records}
    # Preserve diagnostics without publishing workstation-specific absolute paths.
    text = json.dumps(bundle, ensure_ascii=False)
    for prefix in ("/Users/alal/Desktop/code/trex", "/home/alal/Desktop/code/econometrics/trex"):
        text = text.replace(prefix, "<trex>")
    for prefix in ("/Users/alal", "/home/alal", "/Volumes/CodexProjects"):
        text = text.replace(prefix, "<environment>")
    bundle = json.loads(text)
    validate(bundle)
    args.out.mkdir(parents=True, exist_ok=True)
    content = json.dumps(bundle, ensure_ascii=False, indent=2).encode()
    (args.out/"measurements.json.gz").write_bytes(gzip.compress(content, mtime=0))
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.out/"metadata.json").write_text(json.dumps(bundle["metadata"], indent=2)+"\n")
    rows = summarize(bundle)
    with (args.out/"summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [args.out/n for n in ("measurements.json.gz", "manifest.json", "metadata.json", "summary.csv")]}
    (args.out/"SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name, h in hashes.items()))
    print(json.dumps({"cases": len(records), "ok": sum(r["status"] == "ok" for r in rows),
                     "matched": sum(r["agreement_pass"] is True for r in rows),
                     "score_flags": sum(r["score_flag"] for r in rows),
                     "cap_hits": sum(r["cap_hit"] for r in rows)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    collect(p.parse_args())
