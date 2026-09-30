"""Fetch the pinned arXiv source into ignored local storage (Python >=3.10)."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

folder = Path(__file__).resolve().parent
manifest = json.loads((folder / "manifest.json").read_text())
with urlopen(manifest["source_url"], timeout=60) as response:
    archive = response.read()
if hashlib.sha256(archive).hexdigest() != manifest["sha256"]:
    raise RuntimeError("Source checksum changed; inspect before updating the manifest")
download = folder / "download"
download.mkdir(exist_ok=True)
path = download / "arxiv-2609.36445v1.tar.gz"
path.write_bytes(archive)
print(path)
