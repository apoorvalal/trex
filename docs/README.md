# Quarto documentation

Public site: <https://apoorvalal.github.io/trex/>.

Install Quarto **1.8.27** separately, then use the Python environment containing
Trex and PyTorch (on the maintainer's Framework, the shared Conda `torch` env):

```bash
python -m pip install -e '.[docs]'
python docs/build.py
```

This generates the API reference with Quartodoc, verifies public-export coverage,
executes lightweight examples, renders `docs/_site/`, and checks local links,
anchors, images, search entries and math assets. No model-weight downloads or
LLM calls occur. Source-generated reference pages, caches and rendered output
are ignored by git. To generate API sources before a writing preview:

```bash
python docs/build.py --api-only
quarto preview docs
```

`build_api_docs.sh` and `generate_api_docs.py` are compatibility wrappers for API
source generation; they no longer run pdoc, stub Torch or accept an output path.

## Publishing

- Edit `.qmd` files, `_quarto.yml`, assets or Python docstrings on a source branch.
- The `docs` workflow builds and checks PRs; it does not publish PR content.
- Browser checks exercise all pages at desktop/mobile widths, KaTeX, search,
  theme switching and mobile navigation. CI installs Playwright 1.58.2; an
  existing local installation can be selected with `TREX_PLAYWRIGHT_MODULE`.
- On `main`, it commits the verified `_site` tree to `gh-pages` and deploys the
  same Pages artifact. The explicit deployment is intentional: pushes made by
  `GITHUB_TOKEN` do not reliably trigger another Pages build workflow.
- GitHub Pages remains in **GitHub Actions** mode. `gh-pages` holds the published
  files and history, not Quarto source. Do not switch Pages to Jekyll/branch build.
- A manual `docs` workflow run on `main` rebuilds/redeploys without a source edit.
- `build-info.json` and the branch commit message identify the source revision.

`docs/publish.py` is the workflow's branch publisher. It only replaces the
generated tree after the site checks pass, preserves branch history, and never
force-pushes. It is also usable by an authenticated maintainer:

```bash
python docs/publish.py --remote origin
```

Publication is not part of the default local build. CI's separate deployment
job uploads/deploys the same site to Pages after the branch push succeeds.

## Content map

- `start/`: installation, complete first fit, estimator selection, numerics.
- `guides/`: mathematical rationale, executable vignettes and explicit limits.
- `reference/`: generated signatures, methods and docstrings (not edited by hand).
- `references.qmd`: methodology and numerical-reference reading list.
- `contributing.qmd`: contributor and documentation workflow.
- `audits/2026-09-23-cleanup.md`: historical correctness/refactoring audit.

The site is multi-page with shared assets and local KaTeX rather than a
self-contained HTML report. This avoids embedding identical scripts in every
API page while keeping the runtime independent of external math/font CDNs.
