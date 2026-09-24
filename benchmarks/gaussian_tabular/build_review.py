"""Build an executable review notebook from the pilot artifacts."""
import argparse
from pathlib import Path
import nbformat as nbf

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=Path('nb/gaussian_tabular_pilot.ipynb'))
a=p.parse_args()
nb=nbf.v4.new_notebook()
nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb.cells=[md(r'''# Gaussian mixtures with splat-inspired refinement

A feasibility experiment for tabular simulation in Trex. Gaussian blobs can
approximate a distribution in feature space just as they can approximate image
content, but the probability model is different from an image renderer.

**Main conclusion:** normalized Gaussian components are a useful lightweight
baseline. This pilot does not establish an advantage of gradient-based
split/refine training over ordinary EM, nor a general ranking of mixture,
adversarial, and diffusion generators.

## Probability model

For a $d$-column continuous row $x$, the model is

$$p_\theta(x)=\sum_{k=1}^{K}\pi_k\,
\frac{\exp\{-\tfrac12(x-\mu_k)'\Sigma_k^{-1}(x-\mu_k)\}}
{(2\pi)^{d/2}|\Sigma_k|^{1/2}},\qquad
\pi_k=\frac{\exp(a_k)}{\sum_j\exp(a_j)}.$$

Unlike image splatting, there is no colour, opacity compositing, or front-to-back
ordering. Each Gaussian integrates to one and its mass is $\pi_k$. This is an
ordinary **Gaussian mixture family**. The experimental part is growing and
refining the representation, not a new family of probability distributions.

Covariances are parameterized as $\Sigma_k=L_kL_k'+\epsilon I$, where $L_k$ is
lower triangular with positive diagonal and $\epsilon=0.0025$ in transformed
units. The floor prevents components shrinking to point masses to make the
training likelihood unbounded. Fitting minimizes

$$\mathcal L(\theta)=-\frac1n\sum_{i=1}^n\log p_\theta(x_i).$$

The code evaluates the sum in log space and differentiates through the
covariance factors. To sample a row, draw $k\sim\mathrm{Categorical}(\pi)$,
then $x=\mu_k+C_k z$, with $C_kC_k'=\Sigma_k$ and $z\sim N(0,I)$.

## Capacity growth

The experiment starts with one component and doubles capacity through
$K\in\{1,2,4,8,16,32\}$. If $\lambda,v$ are a component's leading covariance
eigenpair, set $\delta=\tfrac12\sqrt{\lambda-\epsilon}\,v$. Replace it by
children with means $\mu\pm\delta$, covariance
$\Sigma-\delta\delta'$, and half its mass. This preserves the parent's first
two moments, while subsequent fitting lets the children diverge.

Every component splits at each stage. **This pilot does not implement
error-targeted splitting or pruning.** At dimension $d$, the normalized mixture
has $K[d+d(d+1)/2]+K-1$ free parameters. Component count is capacity, not matrix
rank. Full covariance costs grow quadratically with column count.

## Experimental design

Three seeds each cover an eight-mode Gaussian ring, a curved banana-shaped
continuous distribution, and the 445-row Lalonde experimental sample. Synthetic
train/validation/test sizes are 2,048/1,024/1,024. Lalonde uses treatment-stratified
60/20/20 splits. Scaling is fit on training data only.

Mixture methods consider the same six capacities. Adam runs 300 updates per
capacity; EM runs three starts with up to 300 iterations. Continuous examples
select capacity using validation likelihood; Adam also selects its checkpoint
using validation likelihood. On Lalonde the continuous-mixture likelihood is
only a fitting proxy: cross-capacity selection uses validation sliced
Wasserstein, and no test density likelihood is claimed.

Merged Trex WGAN-GP, PTGAN and DDPM use 64-by-64 hidden layers and 1,500 training
updates. These are short, prespecified pilot runs—not converged, hyperparameter-
tuned benchmarks. Computational budgets are not equal across methods. Mixture
fit time includes the entire capacity search; EM is CPU-based and the neural
and split/refine models use the RTX 5070. GPU timings are synchronized.

All discrepancy comparisons use equally sized samples. Metrics are computed
on training-standardized columns (with binary columns left at 0/1, as in Trex's
existing transformer). Three-seed standard deviations below describe run/split
variation, not confidence intervals.'''),code('''from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
if ROOT.name == 'nb': ROOT = ROOT.parent
RESULTS = ROOT / 'tmp/gaussian-tabular-2026-09-23'
rows = pd.DataFrame(json.loads((RESULTS / 'metrics.json').read_text()))
summary = pd.DataFrame(json.loads((RESULTS / 'summary.json').read_text()))
config = json.loads((RESULTS / 'config.json').read_text())
display(Markdown('**Hardware:** ' + config['device_name'] + '; PyTorch ' + config['torch_version']))
table = summary.pivot(index='method', columns='dataset', values='sw_mean').round(4)
display(Markdown('### Held-out sliced Wasserstein — lower is better'))
display(table)
display(Markdown('### Fit time in seconds, including mixture capacity search'))
display(summary.pivot(index='method', columns='dataset', values='fit_seconds').round(3))
'''),md('''## Reading the comparison

The Gaussian ring is favorable to Gaussian mixtures by construction. The
banana example checks a curved distribution that is not a finite Gaussian
mixture. On both, inspect mode coverage, tails and curve thickness alongside
a scalar discrepancy. Bootstrap and fresh independent target draws show that
two finite samples from the same population need not have discrepancy zero.

An empirical bootstrap is a strong baseline for a small table. Its limitation
is that it cannot generate new row values or extrapolate; it is not a privacy
mechanism. Beating an undertrained neural baseline is not sufficient evidence
to replace Trex's generators.'''),code('''for dataset in ['ring', 'banana']:
    display(Image(filename=str(RESULTS / f'{dataset}-comparison.png')))
    display(Image(filename=str(RESULTS / f'{dataset}-capacity.png')))
'''),md('''The ellipses are component footprints at two standard deviations, not a
jointly calibrated confidence set or a rendered surface. Their opacity is
only a plotting aid; it is not part of the probability model. All sample
panels use the same axes within an example; extreme tails can fall outside
those axes. Numeric diagnostics use all rows.'''),code('''display(Image(filename=str(RESULTS / 'capacity-nll.png')))
display(Markdown('### Selected component counts and test density scores'))
display(rows[rows.method.isin(['split_adam','em_gmm'])][['dataset','seed','method','k','test_nll']])
display(Markdown('### Ring coverage and off-mode mass'))
display(rows[rows.dataset.eq('ring')].groupby('method')[['mode_coverage','off_mode_fraction']].mean().round(4))
'''),md('''## Lalonde: support and inferential targets

The first pilot reuses Trex's transformer: binary columns are rounded and
nonnegative columns are clipped after generation. That is convenient for a
common comparison, but continuous Gaussian likelihood is **not** a valid
mixed-data likelihood for a table with category indicators and exact zeros.
Rounding does not enforce mutually exclusive category combinations; clipping
can create spurious zero masses; age and education need not be integers.

The table below therefore measures zero-mass errors as well as mean earnings.
The adjusted treatment coefficient is the coefficient of `t` in OLS for
`re78` controlling age, education and the four demographic indicators. It is
a **descriptive association** in a generated table, compared with the held-out
sample—not a known causal target, treatment-effect recovery, or an inference
coverage result. With only 89 test rows, this diagnostic is especially noisy.
A simulator requires separate causal structure if it is to generate
counterfactuals or test identification and estimator coverage.'''),code('''cols = ['sliced_wasserstein','earnings_zero_rate_mae','re78_mean_abs_error','adjusted_association_abs_error']
display(rows[rows.dataset.eq('lalonde')].groupby('method')[cols].agg(['mean','std']).round(3))
'''),md('''## What to build next

The promising target is a compact, inspectable simulator with cheap sampling
and conditional generation. Gaussian mixtures already have analytic
continuous conditional distributions: observing some coordinates updates
component probabilities and yields conditional Gaussian means and Schur-
complement covariances. This is useful for imputation, but conditioning is
not intervention, and conditional sampling is not implemented in this pilot.

Before a public Trex API, compare fixed-K EM, adaptive EM and adaptive Adam
on time to held-out accuracy. Add residual-directed splitting, pruning, and
low-rank-plus-diagonal covariances only when they improve that trade-off.
Mixed tables need Bernoulli/categorical component factors and hurdle models
for exact zero versus positive earnings, together with explicit support
constraints. These are more substantive extensions than renaming a GMM.

The prototype passed six tests: independent SciPy log-density agreement,
mixture moment preservation under splitting, sampling/reproducibility,
finite-difference covariance gradients, bimodal fitting, and input validation.
This is not a full Trex test-suite result. Source lives in
`benchmarks/gaussian_tabular/`; rerun commands and limitations are in its README.
The existing package API and policy-CATE PR are unchanged.

### References

- [Trex tabular generators](https://github.com/apoorvalal/trex/blob/main/trex/simdgp.py)
- [gsplat](https://jmlr.org/papers/v26/24-1476.html): differentiable Gaussian image rendering, the representation inspiration rather than a tabular likelihood implementation.
- [Lalonde replication inputs](https://github.com/evanmunro/dswgan-paper).
- [scikit-learn Gaussian mixture reference](https://scikit-learn.org/stable/modules/mixture.html).
''')]
nb.cells[-1].source += '\n\n## Source and results\n\n[Experiment source and result summaries](https://github.com/apoorvalal/trex/tree/experiment/gaussian-tabular-2026-09-23/benchmarks/gaussian_tabular) · [Executed notebook](https://github.com/apoorvalal/trex/tree/experiment/gaussian-tabular-2026-09-23/nb/gaussian_tabular_pilot.ipynb)\n\nThe experiment lives on a separate Trex branch and leaves the public package\nAPI unchanged.\n'
a.output.parent.mkdir(parents=True,exist_ok=True)
nbf.write(nb,a.output)
print(a.output)
