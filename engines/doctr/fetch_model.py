"""Download the pinned docTR checkpoints (129 MB) into a cache directory, on the host.

Usage: python engines/doctr/fetch_model.py <cache-dir>   (or make doctr-model)

The fast_base detection and crnn_vgg16_bn recognition checkpoints of docTR 1.1.0 are downloaded
from docTR's GitHub releases into <cache-dir>/models/doctr/models/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>. Every file is checked against
models.sha256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from doctr_engine import CHECKPOINTS, MODELS_DIR, SOURCES, sha256

CHECKSUMS = Path(__file__).with_name("models.sha256")


def main(cache: Path) -> None:
    target = cache / MODELS_DIR
    urls = {CHECKPOINTS[role]: url for role, url in SOURCES.items()}
    for line in CHECKSUMS.read_text().splitlines():
        digest, name = line.split()
        path = target / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch   {urls[name]}", flush=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(urls[name], part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from models.sha256")
        part.replace(path)
    print(f"docTR checkpoints in {target}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    main(Path(sys.argv[1]).resolve())
