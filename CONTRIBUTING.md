# Contributing

Use a branch from `main` and keep numerical fixes separate from cosmetic changes.
For estimator changes, add an independent reference or an analytically solved
case, not only an assertion that optimization returned finite numbers.

## Development and tests

```bash
python -m pip install -e '.[test,dev]'
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest
```

The `test` extra includes Statsmodels, linearmodels and scikit-learn. SciPy and
pandas are runtime dependencies. All committed Python tests run offline after
installation. CUDA tests skip explicitly when CUDA is unavailable.

When working in the maintained checkout, use its shared conda `torch`
environment as described in `AGENTS.md`; a repository-local virtual environment
is unnecessary there.

## R reference fixtures

Install R packages `jsonlite`, `sandwich`, `fixest`, and `gmm`, then run:

```bash
Rscript tests/reference/generate_r_fixtures.R
Rscript tests/reference/generate_r_gel.R
python -m pytest tests/test_linear_reference.py tests/test_mle_reference.py tests/test_gel_reference.py
```

Generators contain seeds, specifications and explicit covariance conventions.
The JSON fixtures record package versions. If they change, explain whether the
cause is a specification, a reference version, a tolerance or an implementation
correction. Do not weaken agreement tolerances to conceal a discrepancy.

CI runs Python 3.10/3.12 and a separate R regeneration/comparison job; API docs
are built in their own workflow. The numerical suite does not download external
datasets or require the author's sibling checkouts.

See the [correctness audit](docs/audits/2026-09-23-cleanup.md) for coverage,
known limitations and API conventions, and [docs/README.md](docs/README.md) for
the API documentation build.
