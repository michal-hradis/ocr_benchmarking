"""Download the pinned Surya checkpoints (about 3 GB) into a cache directory, on the host.

Usage: python engines/surya/fetch_model.py <cache-dir>   (or make surya-model)

The text detection, text recognition and layout checkpoints of Surya 0.17.1 are downloaded from
models.datalab.to into <cache-dir>/models/surya/<checkpoint>/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>. Every file is checked against
models.sha256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from surya_engine import MODELS_DIR, sha256

BASE_URL = "https://models.datalab.to"
CHECKSUMS = Path(__file__).with_name("models.sha256")


def main(cache: Path) -> None:
    target = cache / MODELS_DIR
    for line in CHECKSUMS.read_text().splitlines():
        digest, name = line.split()
        path = target / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{name}"
        print(f"fetch   {url}", flush=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from models.sha256")
        part.replace(path)
    print(f"Surya checkpoints in {target}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    main(Path(sys.argv[1]).resolve())
