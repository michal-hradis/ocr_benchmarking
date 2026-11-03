"""Download the pinned Nanonets-OCR2-3B weights (about 7.5 GB) into a cache directory, on the host.

Usage: python engines/nanonets-ocr2/fetch_model.py <cache-dir>   (or make nanonets-ocr2-model)

The files of the pinned Hugging Face revision listed in models.sha256 are written to
<cache-dir>/models/nanonets-ocr2/Nanonets-OCR2-3B/<revision>/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>. Every file is checked against
models.sha256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from nanonets_ocr2_engine import PINNED_DIR, REPOSITORY, REVISION, sha256

CHECKSUMS = Path(__file__).with_name("models.sha256")


def main(args: list[str]) -> None:
    if len(args) != 1:
        sys.exit(__doc__.split("\n\n")[1])
    target = Path(args[0]).resolve() / PINNED_DIR
    target.mkdir(parents=True, exist_ok=True)
    for line in CHECKSUMS.read_text().splitlines():
        digest, name = line.split()
        path = target / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
        print(f"fetch   {url}", flush=True)
        part = path.with_name(name + ".part")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from models.sha256")
        part.replace(path)
    print(f"Nanonets-OCR2-3B {REVISION} in {target}")


if __name__ == "__main__":
    main(sys.argv[1:])
