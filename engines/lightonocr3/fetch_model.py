"""Download pinned LightOnOCR-3 checkpoints into a cache directory, on the host.

Usage: python engines/lightonocr3/fetch_model.py <cache-dir> [0.8B|1B|4B ...]
       (or make lightonocr3-model [CHECKPOINT=4B]; all three without a checkpoint)

The files of each checkpoint's pinned Hugging Face revision listed in models-<checkpoint>.sha256 are
written to <cache-dir>/models/lightonocr3/<name>/<revision>/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>. Every file is checked against
its SHA-256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from lightonocr3_engine import CHECKPOINTS, pinned_dir, sha256

HERE = Path(__file__).parent


def fetch(checksums: Path, target: Path, repository: str, revision: str) -> None:
    for line in checksums.read_text().splitlines():
        digest, name = line.split()
        path = target / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
        print(f"fetch   {url}", flush=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from {checksums.name}")
        part.replace(path)
    print(f"{repository} {revision} in {target}")


def main(args: list[str]) -> None:
    if not args or any(name not in CHECKPOINTS for name in args[1:]):
        sys.exit(__doc__.split("\n\n")[1])
    for checkpoint in args[1:] or list(CHECKPOINTS):
        spec = CHECKPOINTS[checkpoint]
        target = Path(args[0]).resolve() / pinned_dir(checkpoint)
        target.mkdir(parents=True, exist_ok=True)
        checksums = HERE / f"models-{checkpoint}.sha256"
        fetch(checksums, target, spec["repository"], spec["revision"])


if __name__ == "__main__":
    main(sys.argv[1:])
