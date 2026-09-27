#!/usr/bin/env python3
"""Generate, execute and verify the Trex documentation from this checkout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT))


def run(*args, cwd=ROOT, env=None):
    subprocess.run(args, cwd=cwd, env=env, check=True)


def output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def resolve(name):
    """Resolve exports, including ones in a submodule not re-exported."""
    bits = ("trex." + name).split(".")
    for split in range(len(bits), 0, -1):
        try:
            obj = importlib.import_module(".".join(bits[:split]))
        except ModuleNotFoundError as exc:
            if exc.name != ".".join(bits[:split]):
                raise
            continue
        for bit in bits[split:]:
            obj = getattr(obj, bit)
        return obj
    raise ImportError(name)


def generate_api(env):
    import yaml
    import trex

    assert Path(trex.__file__).resolve().parent == ROOT / "trex"
    config = yaml.safe_load((DOCS / "_quarto.yml").read_text())
    sections = config["quartodoc"]["sections"]
    names = [name for section in sections for name in section["contents"]]
    covered = {id(resolve(name)) for name in names}
    missing = []
    for module_name in ["trex", "trex.choice", "trex.gmm", "trex.cmr"]:
        module = importlib.import_module(module_name)
        missing += [f"{module_name}.{name}" for name in module.__all__
                    if id(getattr(module, name)) not in covered]
    if missing:
        raise RuntimeError(f"Public exports missing from API inventory: {missing}")

    reference = DOCS / "reference"
    # This directory is generated-only and ignored by git.
    if reference.exists():
        shutil.rmtree(reference)
    run(sys.executable, "-m", "quartodoc", "build", cwd=DOCS, env=env)
    revision = output("git", "rev-parse", "HEAD")
    guide_for_section = ["linear-regression", "gmm", "conditional-moments",
                         "static-choice", "synthetic-did", "generative-models"]
    special = {
        "LogisticRegression": "likelihood-glms", "PoissonRegression": "likelihood-glms",
        "MaximumLikelihoodEstimator": "likelihood-glms",
        "LatentFactorGLM": "latent-factors", "KNNGroupedFixedEffects": "grouped-effects",
        "NuclearNormMatrixCompletion": "matrix-completion",
    }
    for section, guide in zip(sections, guide_for_section):
        for name in section["contents"]:
            page = reference / f"{name}.qmd"
            if not page.exists():
                raise RuntimeError(f"Quartodoc did not generate {page}")
            obj = resolve(name)
            source = inspect.getsourcefile(obj)
            source_link = ""
            if source and Path(source).resolve().is_relative_to(ROOT):
                line = inspect.getsourcelines(obj)[1]
                path = Path(source).resolve().relative_to(ROOT).as_posix()
                source_link = f" · [Source](https://github.com/apoorvalal/trex/blob/{revision}/{path}#L{line})"
            target = special.get(name, guide)
            if any(key in name for key in ["RustNFP", "HotzMiller", "DynamicChoice", "FlowUtility", "ReplacementUtility", "transition", "ccp", "DeepValue"]):
                target = "dynamic-choice"
            elif any(key in name for key in ["Antitonic", "antitonic"]):
                target = "score-matching"
            elif any(key in name for key in ["GEL", "rho_"]):
                target = "gel"
            elif "grouped_fe" in name or "panel_embeddings" in name:
                target = "grouped-effects"
            elif "matrix_completion" in name or "MatrixCompletionResult" in name:
                target = "matrix-completion"
            elif "Safetensors" in name:
                target = "llm-generators"
            text = page.read_text()
            # Quartodoc lists undocumented attributes but omits their detail
            # sections. Give those rows real targets instead of dangling links.
            anchors = set(re.findall(r"\{\s*#([^\s}]+)", text))
            def attribute_anchor(match):
                label, anchor = match.groups()
                return match[0] if anchor in anchors else f"[`{label}`]{{#{anchor}}}"
            text = re.sub(r"\[([^\]\n]+)\]\(#(trex\.[^)]+)\)", attribute_anchor, text)
            first, rest = text.split("\n", 1)
            heading = re.search(r"\{\s*#([^\s}]+)", first)
            first = f"[]{{#{heading[1]}}}" if heading else first
            page.write_text(f"---\ntitle: {json.dumps(name)}\nrepo-actions: false\nexecute:\n  eval: false\n---\n\n" + first +
                            f"\n\n[Guide & worked example](../guides/{target}.qmd){source_link}\n" + rest)
    index = reference / "index.qmd"
    content = index.read_text()
    first, rest = content.split("\n", 1)
    index.write_text("---\ntitle: API reference\nrepo-actions: false\nexecute:\n  eval: false\n---\n" +
        "\n\nSignatures and public methods are generated from this checkout. "
        "Start with the [method selection guide](../start/choosing.qmd) for "
        "assumptions, data shapes and inference limits. Inherited methods "
        "may be unimplemented for a particular estimator; a signature alone "
        "is not a claim of support. Edit Python docstrings or the "
        "[API inventory](https://github.com/apoorvalal/trex/blob/main/docs/_quarto.yml), "
        "not these generated pages.\n" + rest)
    print(f"API: {len(names)} entries; all declared public exports covered.", flush=True)
    return names


def local_math(site):
    """Bundle a hash-pinned KaTeX release rather than Pandoc's latest CDN URLs."""
    version = "0.16.22"
    url = f"https://registry.npmjs.org/katex/-/katex-{version}.tgz"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    expected = "e9e0d167db3175481cbadaff38e8d90b130f6a3ddb451a47e43c577fd511f365"
    if hashlib.sha256(archive).hexdigest() != expected:
        raise RuntimeError("KaTeX archive hash mismatch")
    dest = site / "assets" / "katex"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as package:
        for member in package.getmembers():
            if not member.isfile():
                continue
            if member.name.startswith("package/dist/"):
                relative = Path(member.name.removeprefix("package/dist/"))
            elif member.name == "package/LICENSE":
                relative = Path("LICENSE")
            else:
                continue
            path = (dest / relative).resolve()
            if not path.is_relative_to(dest.resolve()):
                raise RuntimeError("Invalid archive member")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(package.extractfile(member).read())
    # The Jupyter templates used by Quarto 1.5 and 1.8 use different CDNs.
    unused_loaders = [
        "https://cdnjs.cloudflare.com/ajax/libs/jquery/3.5.1/jquery.min.js",
        "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js",
        "https://cdn.jsdelivr.net/npm/jquery@3.5.1/dist/jquery.min.js",
        "https://cdn.jsdelivr.net/npm/requirejs@2.3.6/require.min.js",
    ]
    loader_pattern = "|".join(re.escape(url) for url in unused_loaders)
    for page in site.rglob("*.html"):
        prefix = Path(os.path.relpath(dest, page.parent)).as_posix() + "/"
        text = page.read_text()
        text = re.sub(r"https://cdn\.jsdelivr\.net/npm/katex@[^/]+/dist/", prefix, text)
        # These vignettes produce static HTML tables/figures, not widgets.
        # Quarto injects unused Jupyter loaders, whose AMD define conflicts
        # with site scripts. Remove those specific CDN shims, not user scripts.
        text = re.sub(rf'<script[^>]*src="(?:{loader_pattern})"[^>]*>\s*</script>', '', text)
        text = re.sub(r"<script[^>]*>\s*define\(['\"]jquery['\"],\s*\[\],\s*function\(\)\s*\{\s*return window\.jQuery;\s*\}\)\s*;?\s*</script>", '', text)
        page.write_text(text)


def provenance(names):
    versions = {name: importlib.metadata.version(name) for name in
                ["torch", "numpy", "scipy", "pandas", "quartodoc", "griffe", "matplotlib"]}
    return {
        "source_repository": "https://github.com/apoorvalal/trex",
        "source_commit": output("git", "rev-parse", "HEAD"),
        "tracked_source_dirty": bool(output("git", "status", "--porcelain", "--untracked-files=no")),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "quarto": output("quarto", "--version"),
        "dependencies": versions, "api_entries": len(names),
        "katex": "0.16.22 (locally bundled, SHA-256 verified)",
        "executable_guides": [p.stem for p in sorted((DOCS / "guides").glob("*.qmd"))
                              if p.stem != "llm-generators"],
        "nonexecuted_guides": ["llm-generators"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-only", action="store_true")
    args = parser.parse_args(argv)
    env = os.environ.copy()
    env["QUARTO_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    names = generate_api(env)
    (DOCS / "build-info.json").write_text(json.dumps(provenance(names), indent=2) + "\n")
    if args.api_only:
        return
    site = DOCS / "_site"
    if site.exists():
        shutil.rmtree(site)
    run("quarto", "render", str(DOCS), "--execute", "--cache-refresh", cwd=ROOT, env=env)
    local_math(site)
    (site / ".nojekyll").touch()
    from check_site import check_site
    check_site(site)


if __name__ == "__main__":
    main()
