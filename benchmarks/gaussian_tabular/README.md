# Gaussian blobs as tabular generators

This experiment transfers the **representation** and **split/refine idea** of
Gaussian splatting to density estimation. It does not transfer the image
renderer. There are no cameras, colours, opacity, or depth ordering here.

A component has a probability mass, a centre in feature space, and a positive
covariance matrix. The density is a normalized sum of Gaussian densities:

$$p(x) = \sum_{k=1}^K \pi_k\,\mathcal{N}(x;\mu_k,\Sigma_k),\qquad
\pi_k\geq 0,\quad \sum_k \pi_k=1.$$

This is a **Gaussian mixture model (GMM)**, not a novel probability family.
Splat-inspired adaptation is an optimization/capacity-management idea: begin
with one broad component, split along principal axes, and refine component
locations, shapes, and masses by gradient descent. Ordinary EM fitting is a
necessary baseline; an improvement over a GAN alone would not establish any
benefit from the splatting analogy.

## Prototype

`model.py` implements a differentiable full-covariance mixture in PyTorch:

- Softmax masses, learned means, and Cholesky covariance factors.
- A fixed eigenvalue floor, `Sigma = L L' + 0.0025 I` in transformed units,
  prevents the unbounded-likelihood covariance-collapse pathology.
- Mini-batch negative-log-likelihood training with Adam.
- Symmetric principal-axis splits preserve component mass, mean, and covariance
  before further fitting. Every component splits at each doubling: this is a
  **uniform growth schedule**, not residual-directed densification or pruning.
- Sampling chooses a component by its mass, then draws from its Gaussian.
- K controls capacity, not matrix rank. A full-covariance K-component density
  in d dimensions has `K * (d + d*(d+1)/2) + K - 1` free parameters.

The prototype deliberately lives outside Trex's public API. It is not a
`gsplat` backend: its 2D/3D rasterization kernels do not implement tabular
likelihoods in arbitrary dimension. PyTorch is sufficient for this pilot.

## Evaluation design

`run.py` compares:

- Split/refine mixtures with K = 1, 2, 4, 8, 16, 32; 300 Adam updates per level.
- Full-covariance sklearn EM mixtures at the same K values, three initializations
  each and up to 300 EM iterations, with covariance regularization 0.0025.
- Trex's already merged WGAN-GP, PTGAN, and DDPM, each with 64x64 hidden layers
  and 1,500 updates. GANs use one critic update per generator update; DDPM uses
  1,000 diffusion time steps and learning rate 0.001. These are **pilot budgets,
  not convergence-certified or equally tuned comparisons**.
- Resampling the training rows, plus independent target draws for synthetic
  examples. These give finite-sample reference discrepancies, not zero-error
  targets or universal performance bounds.

Targets are the existing eight-Gaussian ring benchmark, a curved non-mixture
"banana" construction, and the 445-row Lalonde experimental sample from
`evanmunro/dswgan-paper`. The banana still admits Gaussian-mixture approximation;
its role is to avoid evaluating solely on a finite Gaussian mixture.

Three data/split seeds are used. Synthetic train/validation/test sizes are
2,048/1,024/1,024; Lalonde uses a treatment-stratified 60/20/20 split. All
transformations are estimated on training rows only. Continuous-example model
capacity and Adam checkpoints are selected by validation likelihood. EM chooses
its best restart by training likelihood. On Lalonde, cross-K selection uses
validation sliced Wasserstein; Adam checkpoints within K use training proxy
likelihood. Test rows are used only for reporting.

Every reported sample discrepancy uses equal-sized synthetic and test samples:
Trex's current sliced-Wasserstein helper is not a correct unequal-sample
quantile comparison, so unequal sizes are explicitly avoided here. Standardized
metrics are primary, following the existing benchmark convention. Reported
fit times include the entire K sweep for mixture methods, not just the chosen
K; GPU timing is synchronized. EM runs on CPU, PyTorch models on CUDA. No
claim of equal compute budgets is made.

### Mixed-data limits

Lalonde binaries are left at 0/1 during fitting and rounded on inverse transform;
nonnegative columns are clipped. This reuses Trex's current preprocessing but
**does not give an appropriate mixed discrete/continuous likelihood**. No
Lalonde held-out density likelihood is reported. Thresholding/clipping can
produce wrong category probabilities, spurious zero masses, invalid category
combinations, and fractional age/education. The pilot is not production-ready
mixed-data synthesis.

Earnings zero rates and an adjusted treatment/outcome **association** are
checked alongside distribution metrics. The latter is a descriptive synthetic
regression diagnostic, not a known causal effect or recovered counterfactual.
Distribution fit does not establish causal identification or estimator validity.
No privacy guarantee is provided, and train/test discrepancy is not a privacy
audit. Row bootstrap deliberately repeats training records.

## Run

From the repository root, in the configured `torch` environment:

```bash
python -m pytest -q -o addopts= benchmarks/gaussian_tabular/test_model.py
python benchmarks/gaussian_tabular/run.py \
  --paper-repo ../dswgan-paper \
  --out tmp/gaussian-tabular-2026-09-23
python benchmarks/gaussian_tabular/summarize.py \
  --results tmp/gaussian-tabular-2026-09-23
```

`--trex-source` optionally points to a read-only snapshot of `base.py` and
`simdgp.py`, useful when a remote canonical checkout has independent local
work. Source hashes are recorded in `config.json`. The initial experiment uses
these files from merged main `e6286e0`, without changing the remote checkout's
branch, installed package, or environment.

`metrics.json` contains test comparisons; `capacity.json` records each mixture
capacity's selection score and continuous test NLL; NPZ files contain splits,
samples, and learned mixture parameters. The review notebook has embedded
figures and results; experiment outputs are retained in the ignored `tmp/`
directory, while the scripts and review artifacts are committed.

## What would make this worth developing?

The next relevant comparison is **fixed-K EM versus adaptive EM versus adaptive
Adam**, with time-to-held-out-accuracy curves, not merely equal iteration counts.
A useful advance would be residual-directed splitting, pruning negligible
components, and scalable low-rank-plus-diagonal covariances, or estimand-targeted
losses that demonstrably improve downstream simulation.

For actual mixed tables, use categorical/Bernoulli component factors and hurdle
models for exact zeros plus positive earnings, with explicit joint constraints.
For continuous conditional generation, a joint Gaussian mixture already admits
analytic conditioning: update component masses using the observed coordinates,
then use each Gaussian's conditional mean and Schur-complement covariance.
That is attractive for imputation and controlled simulation, but it is **not yet
implemented in this prototype**, and conditioning is not a causal intervention.

## Reviewing the pilot

The executed notebook is `nb/gaussian_tabular_pilot.ipynb`. Small machine-readable
results are committed under `benchmarks/gaussian_tabular/results/`; full draws,
parameters and figures remain in `tmp/gaussian-tabular-2026-09-23`. There are
60 test-comparison rows and 108 per-capacity records across the three seeds and
three datasets. All EM fits report convergence. Six model tests passed; the
full Trex suite was not run, since its public implementation is unchanged.

The pilot finds split/refine and ordinary EM similarly effective on the curved
continuous target (mean sliced Wasserstein 0.0482 versus 0.0512); EM's complete
capacity search is much faster (0.220 versus 3.261 seconds on this hardware).
On the ring, EM's sample discrepancy is lower (0.0427 versus 0.0549). On
Lalonde, split/refine is lower in this pilot (0.1550 versus 0.1905), but the tiny
validation/test samples, different fitting budgets, and approximate mixed-data
support preclude a general performance claim. Short neural runs are not a
substitute for a tuned/converged comparison.

EM is fitted in float64: the first float32 exploratory run triggered sampling
covariance-precision warnings. The recorded run uses float64 EM and reports no
such warnings. Gaussian-gradient and neural training use float32 on CUDA.
The KS routine sometimes falls back from exact to asymptotic p-values on tied
mixed-data samples; only the KS statistic is used here.

To regenerate the executed notebook, build it with `build_review.py` and execute
with `nbclient` in the configured environment. A dependency-free static HTML
render can be made using Pandoc (also bundled with Quarto):

```bash
pandoc nb/gaussian_tabular_pilot.ipynb --from=ipynb --to=html5 \
  --standalone --mathml --embed-resources \
  --metadata pagetitle='Gaussian mixtures with splat-inspired refinement' \
  --css=benchmarks/gaussian_tabular/review.css \
  --output=tmp/gaussian-tabular-2026-09-23/index.html
```

The HTML uses browser-native MathML, embedded plots and no remote script/CDN.
Serve it alongside a copy of the executed notebook and `source.tar.gz` to make
its download links work.
