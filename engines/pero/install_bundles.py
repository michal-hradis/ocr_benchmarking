"""Copy user-supplied PERO OCR model bundles into a cache directory, on the host.

Usage: python engines/pero/install_bundles.py <bundles-dir> <cache-dir>   (or make pero-bundles)

Every bundle registered in bundles.sha256 is copied from <bundles-dir>/<bundle>/ to
<cache-dir>/models/pero/<bundle>/, which the engine reads as /cache/models/pero/... when the run is
started with --cache <cache-dir>. Only registered files are copied, each checked against
bundles.sha256 before it is moved into place; files already present and correct are kept. Nothing
is downloaded: PERO bundles are supplied by the user (README.md).
"""

import shutil
import sys
from pathlib import Path

from pero_engine import BUNDLES_DIR, read_checksums, sha256

CHECKSUMS = Path(__file__).with_name("bundles.sha256")


def main(source: Path, cache: Path) -> None:
    for bundle, files in read_checksums(CHECKSUMS).items():
        target = cache / BUNDLES_DIR / bundle
        target.mkdir(parents=True, exist_ok=True)
        for name, digest in files.items():
            path = target / name
            if path.is_file() and sha256(path) == digest:
                print(f"ok      {bundle}/{name}")
                continue
            origin = source / bundle / name
            if not origin.is_file():
                sys.exit(f"{origin} does not exist")
            print(f"copy    {origin}", flush=True)
            part = path.with_name(name + ".part")
            shutil.copyfile(origin, part)
            if sha256(part) != digest:
                part.unlink()
                sys.exit(f"{bundle}/{name}: SHA-256 differs from bundles.sha256")
            part.replace(path)
        print(f"PERO bundle {bundle} in {target}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__.split("\n\n")[1])
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
