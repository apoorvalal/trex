# API Documentation

[Correctness and refactoring audit, September 2026](audits/2026-09-23-cleanup.md)
documents independent Python/R checks, numerical fixes and remaining limits.

This project uses [`pdoc`](https://pdoc.dev) to generate API documentation from
Python docstrings.

## Build

From the repository root:

```bash
uv sync --extra docs
bash docs/build_api_docs.sh
```

By default, docs are generated into `docs/api/`.
The build script enumerates the full `trex` module tree so subpackages
like `trex.choice` and `trex.choice.dynamic` are included.
When `torch` is not installed (for example in CI docs builds), a lightweight stub
is used so API pages can still be generated without installing full PyTorch.

To choose a different output directory:

```bash
bash docs/build_api_docs.sh tmp/pdoc-site
```
