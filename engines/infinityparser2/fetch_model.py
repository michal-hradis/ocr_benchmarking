"""Download the pinned Infinity-Parser2-Flash weights (about 4.4 GB) into a cache directory, on
the host.

Usage: python engines/infinityparser2/fetch_model.py <cache-dir>   (or make infinityparser2-model)

The files of the pinned Hugging Face revision are written to
<cache-dir>/models/infinityparser2/Infinity-Parser2-Flash/<revision>/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>. Every file is checked against
models.sha256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from infinityparser2_engine import PINNED_DIR, REPOSITORY, REVISION, sha256

CHECKSUMS = Path(__file__).with_name("models.sha256")


def main(cache: Path) -> None:
    target = cache / PINNED_DIR
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
    print(f"Infinity-Parser2-Flash {REVISION} in {target}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    main(Path(sys.argv[1]).resolve())
