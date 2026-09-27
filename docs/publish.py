#!/usr/bin/env python3
"""Commit a verified site to gh-pages without touching the source worktree."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from check_site import check_site

ROOT = Path(__file__).resolve().parents[1]


def git(*args, env=None, cwd=ROOT, input=None):
    return subprocess.check_output(["git", *args], cwd=cwd, env=env,
                                   input=input, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--site", type=Path, default=ROOT / "docs" / "_site")
    args = parser.parse_args()
    site = args.site.resolve()
    check_site(site)
    info = json.loads((site / "build-info.json").read_text())
    revision = git("rev-parse", "HEAD")
    if info["source_commit"] != revision or info["tracked_source_dirty"]:
        raise RuntimeError("Publish only a clean build of the checked-out source commit")
    if not (site / ".nojekyll").exists():
        raise RuntimeError("Expected a complete build with .nojekyll")
    branch = "refs/heads/gh-pages"
    existing = git("ls-remote", "--heads", args.remote, branch)
    parent = None
    if existing:
        git("fetch", "--no-tags", args.remote, branch)
        parent = git("rev-parse", "FETCH_HEAD")
    git_dir = Path(git("rev-parse", "--absolute-git-dir"))
    # A separate index creates an exact site tree; no checkout/reset/stash.
    with tempfile.TemporaryDirectory(prefix="docs-index-", dir=git_dir) as scratch:
        env = os.environ.copy()
        env["GIT_INDEX_FILE"] = str(Path(scratch) / "index")
        env["GIT_WORK_TREE"] = str(site)
        env["GIT_DIR"] = str(git_dir)
        git("read-tree", "--empty", env=env)
        git("add", "--all", "--force", ".", env=env, cwd=site)
        tree = git("write-tree", env=env)
        command = ["commit-tree", tree]
        if parent:
            command += ["-p", parent]
        commit = git(*command, env=env, input=f"docs: publish source {revision}\n")
        # Normal fast-forward push: a concurrent update fails, never overwrites.
        git("push", args.remote, f"{commit}:{branch}")
    print(f"Published {commit} to gh-pages (source {revision})")


if __name__ == "__main__":
    main()
