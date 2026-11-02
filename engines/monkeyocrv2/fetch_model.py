"""Download the pinned MonkeyOCRv2 checkpoints or source files, checked against their SHA-256.

Usage: python engines/monkeyocrv2/fetch_model.py <cache-dir> [B-Parsing|B-Parsing-DFlash ...]
       (or make monkeyocrv2-model; both checkpoints without a name)
       python fetch_model.py --source <dir>   (in the Dockerfile: upstream's pinned source files)

The files of each checkpoint's pinned Hugging Face revision listed in models-<checkpoint>.sha256
are written to <cache-dir>/models/monkeyocrv2/MonkeyOCRv2-<checkpoint>/<revision>/, which the
engine reads as /cache/models/... when the run is started with --cache <cache-dir>. With --source,
the files of source.sha256 are downloaded from github.com/Yuliang-Liu/MonkeyOCRv2 at the pinned
commit. Every file is checked against its SHA-256 before it is moved into place; files already
present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from monkeyocrv2_engine import (
    CHECKPOINTS,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    pinned_dir,
    sha256,
)

HERE = Path(__file__).parent


def fetch(checksums: Path, target: Path, base_url: str) -> None:
    for line in checksums.read_text().splitlines():
        digest, name = line.split()
        path = target / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{name}"
        print(f"fetch   {url}", flush=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from {checksums.name}")
        part.replace(path)
    print(f"{base_url} in {target}")


def main(args: list[str]) -> None:
    if len(args) == 2 and args[0] == "--source":
        base = f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_COMMIT}"
        fetch(HERE / "source.sha256", Path(args[1]).resolve(), base)
        return
    if not args or args[0].startswith("-") or any(n not in CHECKPOINTS for n in args[1:]):
        sys.exit(__doc__.split("\n\n")[1])
    for checkpoint in args[1:] or list(CHECKPOINTS):
        spec = CHECKPOINTS[checkpoint]
        target = Path(args[0]).resolve() / pinned_dir(checkpoint)
        target.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}"
        fetch(HERE / f"models-{checkpoint}.sha256", target, base)


if __name__ == "__main__":
    main(sys.argv[1:])
