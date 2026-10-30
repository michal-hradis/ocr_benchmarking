"""Download the pinned Docling models (397 MB) into a cache directory, on the host.

Usage: python engines/docling/fetch_model.py <cache-dir>   (or make docling-model)

The Heron layout model and the accurate TableFormer from Hugging Face at pinned commits, and the
RapidOCR PP-OCRv5 Latin models of RapidOCR 3.10.0 from its ModelScope release, are downloaded
into <cache-dir>/models/docling/ in the directory layout of Docling's artifacts_path, which the
engine reads as /cache/models/docling when the run is started with --cache <cache-dir>. Every
file is checked against models.sha256 before it is moved into place; files already present and
correct are kept.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

CHECKSUMS = Path(__file__).with_name("models.sha256")
MODELS_DIR = "models/docling"
HF = "https://huggingface.co/docling-project"
# Directory below the artifacts directory: URL its files are downloaded from.
SOURCES = {
    # docling-layout-heron, main as of Docling 2.145.0 (which asks for "main")
    "docling-project--docling-layout-heron": (
        f"{HF}/docling-layout-heron/resolve/8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
    ),
    # docling-models tag v2.3.0, the revision Docling 2.145.0 asks for
    "docling-project--docling-models": (
        f"{HF}/docling-models/resolve/fc0f2d45e2218ea24bce5045f58a389aed16dc23"
    ),
    # The URLs RapidOCR 3.10.0's model registry gives for these files
    "RapidOcr": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx",
}
RAPIDOCR_PATHS = {
    "ch_PP-OCRv5_det_mobile.onnx": "PP-OCRv5/det",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": "PP-OCRv4/cls",
    "latin_PP-OCRv5_rec_mobile.onnx": "PP-OCRv5/rec",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def url(name: str) -> str:
    directory, rest = name.split("/", 1)
    if directory == "RapidOcr":
        return f"{SOURCES[directory]}/{RAPIDOCR_PATHS[rest]}/{rest}"
    return f"{SOURCES[directory]}/{rest}"


def main(cache: Path) -> None:
    target = cache / MODELS_DIR
    for line in CHECKSUMS.read_text().splitlines():
        digest, name = line.split()
        path = target / name
        if path.is_file() and sha256(path) == digest:
            print(f"ok      {name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch   {url(name)}", flush=True)
        part = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url(name), part)
        if sha256(part) != digest:
            part.unlink()
            sys.exit(f"{name}: SHA-256 differs from models.sha256")
        part.replace(path)
    print(f"Docling models in {target}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    main(Path(sys.argv[1]).resolve())
