#!/usr/bin/env python3
"""Fail a documentation build on missing pages, links, anchors or assets."""
from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.ids, self.links, self.math, self.images = set(), [], 0, 0
        self.remote_assets = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if "math" in attrs.get("class", "").split():
            self.math += 1
        if tag == "img":
            self.images += 1
        for key in ("href", "src"):
            if key in attrs and tag in ("a", "link", "script", "img", "iframe", "source"):
                self.links.append(attrs[key])
                if tag in ("script", "img", "iframe", "source") or (tag == "link" and attrs.get("rel") == "stylesheet"):
                    if urlsplit(attrs[key]).netloc:
                        self.remote_assets.append(attrs[key])


def check_site(site):
    site = Path(site).resolve()
    pages = {p.resolve(): Page(p.read_text()) for p in site.rglob("*.html")}
    errors = []
    for path, page in pages.items():
        errors += [f"{path.relative_to(site)}: external runtime asset {url}"
                   for url in page.remote_assets]
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not link or link.startswith("javascript:"):
                continue
            local = unquote(url.path)
            if local.startswith("/trex/"):
                target = site / local[len("/trex/"):]
            elif local.startswith("/"):
                target = site / local.lstrip("/")
            else:
                target = path.parent / local if local else path
            if target.is_dir():
                target /= "index.html"
            target = target.resolve()
            if not target.is_relative_to(site) or not target.exists():
                errors.append(f"{path.relative_to(site)}: missing {link}")
            elif url.fragment and target in pages:
                anchor = unquote(url.fragment)
                if anchor not in pages[target].ids and not anchor.startswith("/tabset-"):
                    errors.append(f"{path.relative_to(site)}: missing anchor {link}")
    for required in ["index.html", "reference/index.html", "search.json", "llms.txt", "build-info.json"]:
        if not (site / required).is_file():
            errors.append(f"Missing required site file: {required}")
    search = json.loads((site / "search.json").read_text())
    indexed = {row.get("href", "").split("#")[0] for row in search}
    source = Path(__file__).resolve().parent
    for directory in ("start", "guides", "reference"):
        for qmd in (source / directory).glob("*.qmd"):
            html = qmd.relative_to(source).with_suffix(".html").as_posix()
            if not (site / html).is_file():
                errors.append(f"Missing rendered source: {html}")
            if html not in indexed:
                errors.append(f"Missing search entry: {html}")
    math_count = sum(page.math for page in pages.values())
    if math_count < 30 or not list(site.rglob("katex.min.js")):
        errors.append("Missing expected mathematics or local KaTeX runtime")
    if errors:
        raise RuntimeError("Site checks failed:\n" + "\n".join(sorted(set(errors))))
    summary = {"html_pages": len(pages), "math_elements": math_count,
               "image_elements": sum(page.images for page in pages.values()),
               "checked_links": sum(len(page.links) for page in pages.values()),
               "search_entries": len(search)}
    print("Site checks passed: " + json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    check_site(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "_site")
