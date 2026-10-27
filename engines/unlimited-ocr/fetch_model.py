"""Download the pinned Unlimited-OCR weights (6.7 GB) into a cache directory, on the host.

Usage: python engines/unlimited-ocr/fetch_model.py <cache-dir>   (or make unlimited-ocr-model)

The files of the pinned Hugging Face revision listed in models.sha256 are written to
<cache-dir>/models/unlimited-ocr/Unlimited-OCR/<revision>/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>. Every file is checked against
its SHA-256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from unlimited_ocr_engine import PINNED_DIR, REPOSITORY, REVISION, sha256

MODELS = Path(__file__).with_name("models.sha256")


def fetch(checksums: Path, target: Path) -> None:
    for line in checksums.read_text().splitlines():
        digest, name = line.split()
        path = target / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        print(f"fetch   {url}", flush=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from {checksums.name}")
        part.replace(path)
    print(f"{REPOSITORY} {REVISION} in {target}")


def main(args: list[str]) -> None:
    if len(args) != 1:
        sys.exit(__doc__.split("\n\n")[1])
    target = Path(args[0]).resolve() / PINNED_DIR
    target.mkdir(parents=True, exist_ok=True)
    fetch(MODELS, target)


if __name__ == "__main__":
    main(sys.argv[1:])
