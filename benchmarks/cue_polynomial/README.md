# Polynomial-ratio CUE: scalar linear-IV prototype

Independent implementation of the positive-definite scalar characterization in
[Moreira, Newey and Sharifvaghefi (2026), arXiv:2609.36445v1](https://arxiv.org/abs/2609.36445v1).
See the [executed notebook](../../nb/cue_polynomial_comparison.ipynb),
[source provenance](source/README.md), and [solver](../../trex/gmm/cue.py).

## Scope and integration

`ScalarIVCUE` is an opt-in CPU/float64 objective and candidate-enumeration API.
It does not change `GMMEstimator` (two-step GMM) or `GELEstimator` defaults.

```python
from trex.gmm import ScalarIVCUE

problem = ScalarIVCUE.from_data(z, y, x, weighting="reduced_form")
result = problem.minimize_polynomial()
print(result.beta, result.fun, result.at_infinity, result.certified)
```

`z` is n-by-k, `y` and the single endogenous regressor `x` are vectors.
Demeaning removes an intercept by default. More general included controls are
not handled by this data helper. The low-level constructor also accepts
`a=Z'y/sqrt(n)`, `b=Z'x/sqrt(n)`, and a fixed positive-definite 2k-by-2k joint
covariance in outcome/regressor block order.

Supported covariance builders: HC0, Bartlett HAC (`covariance="hac"`, explicit
`max_lags`), and one-way CR0 clustering (`covariance="cluster"`, `clusters`).
The dependence builders are unit-tested, not Monte Carlo-validated here.
No automatic ridge, parameter standard errors, or weak-IV inference is added.

**Weight definitions matter.** The default uses reduced-form OLS residuals to
estimate the fixed joint covariance. `weighting="raw"` instead reproduces the
structural-residual moment covariance. With HC0 its objective equals
`2*n` times the existing quadratic GEL profile; this is tested pointwise and
through actual GEL fits in the notebook. Different weights are different
finite-sample objectives, not just different optimizers.

## Algorithm sketch

1. Form affine moments g(beta)=a−beta*b and quadratic covariance Omega(beta).
2. Use Q=p/q with p=g'adj(Omega)g and q=det(Omega), both degree at most 2k.
3. Recover their coefficients by Chebyshev interpolation using matrix solves
   and determinants. Construct h=p'q−pq', whose degree is at most 4k−2.
4. Enumerate real companion eigenvalues, polish them against analytic matrix
   derivatives, and compare original criterion values, not interpolated ratios.
5. Compare explicitly with Q(infinity)=b'S22^-1*b. Handle a constant criterion
   and the just-identified IV ratio separately.

Covariance-based parameter centering/scaling, instrument whitening, and direct
and reciprocal charts improve numerical behavior without a bounded coefficient
search. The two charts cover the whole extended real line. This basis choice
implements the paper's scalar root recipe; it is not the author's unpublished
interpolation code. No symbolic algebra dependency is required.

**Not a certificate:** `certified` is always false. Off-node interpolation and
matrix-derivative checks reject failures, but are not rigorous root isolation.
The current polynomial solver caps k at 12. Four of the 700 saved attempts
failed numerical checks at k=10; the cap is not a guarantee of success below it.
The paper's k=30/60 experiments, singular/rank-changing covariance case and
multivariate elimination/boundary systems remain outside the implementation.

## Executed experiment

100 draws per design, n=800, true beta=5, endogeneity rho=.6,
heteroskedasticity w=sqrt(.5), with the paper's power/interaction instruments.
Six reduced-form weighting designs plus a separately labeled raw-weighting
bridge: **700 datasets, 9,100 optimizer/start outcomes**. This is not a
reproduction of the paper's 10,000-replication tables.

All solvers within each design use identical data and covariance. Iterative
methods start at 2SLS, zero, and the true coefficient (an infeasible diagnostic).
The BFGS baseline uses Trex's central finite-difference helper; a second BFGS
uses analytic gradients. Full-sample PyTorch Adam uses lr=.001 (default) and
.05, each with at most 3,000 updates. Rates were specified before the full run.

Primary six-design results, starting iterative methods at **2SLS**:

| Method | Numerical minimum recovered / 600 | Stopping/validation rule passed / 600 |
|---|---:|---:|
| Polynomial | 596 | 596 |
| BFGS, Trex finite differences | 571 | 600 |
| BFGS, analytic gradient | 571 | 600 |
| Adam, lr=.001 | 469 | 454 |
| Adam, lr=.05 | 555 | 569 |

Recovery is relative to an independent 2,049-point angular grid plus refinement
of each detected local minimum, with objective tolerance 1e-6*(1+abs(reference)).
The grid reference itself is numerical, not a global certificate. BFGS recovers
455/600 objectives from zero and 548/600 from the true coefficient despite
reporting success on every fit. Small gradient and objective agreement are
different diagnostics; hence neither column necessarily dominates the other.

Across both weighting panels, all **696 accepted polynomial results** agree
with the independent reference within **5.1e-12**; four attempts explicitly
refused to return a solution. Seeds and rejection messages are retained.

**Statistical error does not uniformly improve.** At k=4, mu²=1, median absolute
error is about **1.25 for polynomial CUE versus 1.11 for BFGS-from-2SLS**.
At k=4, mu²=4 it is **.65 versus .67**. The notebook reports median bias,
median absolute error, RMSE, 90% interquantile ranges, paired bootstrap
intervals, missing/finite counts and initial-value sensitivity. Failed
iterations are not silently removed from error summaries.

Adam is batched across independent fits for Monte Carlo throughput; batch
amortized times are not compared with serial solver latency. The notebook
separately runs warm-up plus five serial timing repeats and checks batched
versus separate Adam estimates and stopping iterations.

## Reproduce

Use this checkout's shared `torch` conda environment, as specified in AGENTS.md:

```bash
conda run -n torch python -m pip install -e '.[test,docs]'
conda run -n torch python benchmarks/cue_polynomial/source/fetch.py
conda run -n torch python -m pytest tests/test_cue_polynomial.py
conda run -n torch python -m benchmarks.cue_polynomial.experiment --replications 100
```

Exact numerical/notebook versions used by the saved run are recorded in
`requirements.txt` (Python 3.12). In particular, the notebook uses pandas >=2.2
for `include_groups=False`; these experiment pins do not constrain Trex's
runtime dependencies. Install them in a dedicated replication environment if
the existing shared environment differs, rather than downgrading shared work.

`results.json.gz` contains the full float64 run, environment versions, every
seed/estimate/objective/status, root candidates and diagnostics, reference
values, setup timings and Adam batch timings. No data download is needed.
Open `nb/cue_polynomial_comparison.ipynb` with the same Python environment and
run all cells. It reads saved results by default; set `RECOMPUTE=True` to rerun
the Monte Carlo in the notebook.

## Further work

- Adaptive precision and interval/root-isolation diagnostics before expanding
  the instrument cap; recover the four refused cases without hiding them.
- Multiple endogenous regressors require stationary polynomial **systems** and
  stationary boundary directions, with real feasibility checks. Coordinate
  descent is not a replacement for the paper's global characterization.
- Singular covariance requires rank-drop candidates and interval limits;
  adding a ridge defines a different criterion.
- Add estimator-level inference only with its identification assumptions;
  obtaining a numerical minimum does not validate weak-IV Wald inference.
