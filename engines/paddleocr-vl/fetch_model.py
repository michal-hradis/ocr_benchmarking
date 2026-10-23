"""Download the pinned PaddleOCR-VL-1.6 and PP-DocLayoutV3 files (about 2 GB) into a cache
directory, on the host.

Usage: python engines/paddleocr-vl/fetch_model.py <cache-dir>   (or make paddleocr-vl-model)

The files of the pinned Hugging Face revisions are written to
<cache-dir>/models/paddleocr-vl/<repository>/<revision>/, which the engine reads as
/cache/models/... when the run is started with --cache <cache-dir>. Every file is checked against
models.sha256 before it is moved into place; files already present and correct are kept.
"""

import sys
import urllib.request
from pathlib import Path

from paddleocr_vl_engine import MODELS, MODELS_DIR, sha256

CHECKSUMS = Path(__file__).with_name("models.sha256")
# Directory under MODELS_DIR (<repository name>/<revision>): (repository, revision).
SOURCES = {
    f"{repository.split('/', 1)[1]}/{revision}": (repository, revision)
    for _, repository, revision in MODELS.values()
}


def main(cache: Path) -> None:
    root = cache / MODELS_DIR
    for line in CHECKSUMS.read_text().splitlines():
        digest, name = line.split()
        directory, file = name.rsplit("/", 1)
        repository, revision = SOURCES[directory]
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{file}"
        print(f"fetch   {url}", flush=True)
        part = path.with_name(file + ".part")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from models.sha256")
        part.replace(path)
    print(f"PaddleOCR-VL-1.6 and PP-DocLayoutV3 in {root}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    main(Path(sys.argv[1]).resolve())
