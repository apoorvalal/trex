# Matched device benchmarks

`devices.py` measures public Trex fits on **identical serialized synthetic data**.
The suite contains 14 workloads spanning 12 estimator classes, each at three
sizes. The manifest records dimensions, precision, seeds and input checksums.
Generate the data once; copy those exact files to every comparison host.

```bash
python benchmarks/devices.py generate --data tmp/device-bench/data
python benchmarks/devices.py run --host mac --devices cpu,mps \
  --data tmp/device-bench/data --output tmp/device-bench/results
# On the CUDA host, after copying data:
python benchmarks/devices.py run --host fwk --devices cpu,cuda \
  --data tmp/device-bench/data --output tmp/device-bench/results
```

Use matching PyTorch, NumPy, SciPy, pandas and pytorch-minimize versions. Host
metadata records the actual versions, Torch build, CPU/GPU, thread count,
implementation hash and runner hash. Hardware and OS still differ between hosts.
The benchmark does not update, replace or monkeypatch an estimator.

## Measurement contract

- Every case starts in a separate process. Imports and file reads are outside
  timing. One whole fit warms up that case, then three fresh fits are measured.
  All raw repetitions are retained; warm-up results are labeled.
- Inputs start in host RAM. Primary end-to-end time includes model construction,
  device transfers, public `fit`, and copying fitted results back to CPU.
  Fit-only and preparation times are also reported. Evaluation diagnostics,
  reference comparisons, serialization, and machine-to-machine file transfer
  are excluded. CUDA/MPS are synchronized at timing boundaries.
- Eight intra-op/BLAS threads and one Torch inter-op thread are requested on
  both hosts. No CPU affinity, clock lock or backend-specific tuning is applied.
  TF32, automatic mixed precision, `torch.compile`, and MPS CPU fallback are off.
- Most workloads use float32 on every device. TWFE, GMM and dynamic-choice
  implementations require float64; those use float64 on both CPU and CUDA.
  A Metal failure is recorded, not silently reinterpreted as a GPU timing.
- Same data, parameter initialization, objective, solver, tolerances and budget
  across backends. Neural SMD weights come from the copied arrays, not a
  backend-dependent random seed. Latent-factor initialization uses the
  estimator's SVD; compare fitted predictions, not unidentified factor signs.
- Neural SMD (60 Adam steps) and Gaussian latent factors (120 AdamW steps) are
  **fixed-work training comparisons**, not claims of convergence. Other
  iterative fits use the same stopping rule and cap; diagnostics/iteration
  counts accompany their times. A relative-loss stop is not an optimality proof.
- Every successful fit returns numerical diagnostics and fitted coefficients,
  standard errors or a deterministic grid of fitted predictions. CPU/backend
  agreement and score residuals are checked separately from speed. Timing a
  failed, nonfinite or materially different solution is not a valid speedup.
- A 240-second case timeout includes imports, the warm-up, repetitions and
  diagnostics, not just a single fit. Timeout/error cases have no speed estimate.

The suite is a workload study on two machines, not a hardware leaderboard or a
Monte Carlo estimator comparison. The three repetitions reuse one fixed DGP
draw per size. They measure runtime variability, not statistical uncertainty.

## Smoke runs and results

Use `--only wls_1,mmr_1` to restrict cases, and `--warmups 0 --repeats 1`
for a preflight. Preflight outputs must be kept separate from final results.
Existing result files are not overwritten unless `--overwrite` is supplied.

Raw data and process logs belong in ignored `tmp/`; compact final results,
manifest and analysis belong under `benchmarks/results/` and are linked from
the Quarto performance page. CI renders saved measurements; it does not invent
GPU measurements on a hosted CPU runner.
