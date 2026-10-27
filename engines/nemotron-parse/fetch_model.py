"""Download the pinned Nemotron 3.5 Parse weights (2.5 GB) into a cache directory, on the host,
or the source of its nemotron-parse package into a directory, when the engine image is built.

Usage: python engines/nemotron-parse/fetch_model.py <cache-dir>   (or make nemotron-parse-model)
       python fetch_model.py --source <dir>                        (in the Dockerfile)

The files of the pinned Hugging Face revision listed in models.sha256 are written to
<cache-dir>/models/nemotron-parse/Nemotron-3.5-Parse/<revision>/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>; the files listed in
source.sha256 are written to <dir>. Every file is checked against its SHA-256 before it is moved
into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from nemotron_parse_engine import PINNED_DIR, REPOSITORY, REVISION, sha256

MODELS = Path(__file__).with_name("models.sha256")
SOURCE = Path(__file__).with_name("source.sha256")


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
    if len(args) == 2 and args[0] == "--source":
        fetch(SOURCE, Path(args[1]).resolve())
    elif len(args) == 1:
        target = Path(args[0]).resolve() / PINNED_DIR
        target.mkdir(parents=True, exist_ok=True)
        fetch(MODELS, target)
    else:
        sys.exit(__doc__.split("\n\n")[1])


if __name__ == "__main__":
    main(sys.argv[1:])
