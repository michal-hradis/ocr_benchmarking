"""Download the pinned dots.mocr weights and code (about 6.1 GB) into a cache directory, on
the host.

Usage: python engines/dots-mocr/fetch_model.py <cache-dir>   (or make dots-mocr-model)

The files of the pinned Hugging Face revision are written to
<cache-dir>/models/dots-mocr/dots-mocr/<revision>/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>. Every file is checked against
models.sha256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from dots_mocr_engine import PINNED_DIR, REPOSITORY, REVISION, sha256

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
    print(f"dots.mocr {REVISION} in {target}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    main(Path(sys.argv[1]).resolve())
