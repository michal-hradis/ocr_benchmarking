"""Download the pinned tessdata_best models (about 48 MB) into a cache directory, on the host.

Usage: python engines/tesseract/fetch_tessdata.py <cache-dir>   (or make tesseract-best)

The models of the pinned tesseract-ocr/tessdata_best revision are written to
<cache-dir>/models/tesseract/tessdata_best/<revision>/, which the engine reads as /cache/models/...
when the run is started with --cache <cache-dir>. Every file is checked against
tessdata_best.sha256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from tesseract_engine import BEST_DIR, BEST_REVISION, read_checksums, sha256

CHECKSUMS = Path(__file__).with_name("tessdata_best.sha256")
REPOSITORY = "tesseract-ocr/tessdata_best"


def main(cache: Path) -> None:
    target = cache / BEST_DIR
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in read_checksums(CHECKSUMS).items():
        path = target / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{BEST_REVISION}/{name}"
        print(f"fetch   {url}", flush=True)
        part = path.with_name(name + ".part")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from tessdata_best.sha256")
        part.replace(path)
    print(f"tessdata_best {BEST_REVISION} in {target}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    main(Path(sys.argv[1]).resolve())
