"""docTR text detector and recognizer adapter; runs in the engine image.

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the pinned
docTR detection and recognition checkpoints from the /cache volume once (checked against
models.sha256), then for every sample image in job order detect and recognize the words, let
docTR group them into lines and blocks in its reading order, and write $OCRBENCH_OUTPUT/<id>/
with docTR's native page export (doctr.json), the standardized prediction.json and, last,
result.json. docTR's relative coordinates are converted to pixels of the input image (README.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_DOCTR_CACHE", "/cache"))
MODELS_DIR = "models/doctr"  # relative to the cache: docTR's DOCTR_CACHE_DIR
CHECKSUMS = Path(os.environ.get("OCRBENCH_DOCTR_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# docTR 1.1.0's default pair (doctr.models.ocr_predictor) and its pretrained checkpoints, by
# their path below DOCTR_CACHE_DIR; docTR checks the hash prefix in the file name itself.
ARCHS = {"detection": "fast_base", "recognition": "crnn_vgg16_bn"}
CHECKPOINTS = {
    "detection": "models/fast_base-688a8b34.pt",
    "recognition": "models/crnn_vgg16_bn-0417f351.pt",
}
# The GitHub release assets docTR's own URLs (doctr-static.mindee.com) redirect to.
SOURCES = {
    "detection": "https://github.com/mindee/doctr/releases/download/v0.8.1/fast_base-688a8b34.pt",
    "recognition": (
        "https://github.com/mindee/doctr/releases/download/v0.12.0/crnn_vgg16_bn-0417f351.pt"
    ),
}
# Engine parameters (job.json "params"), README.md. detection_size is the side of the square the
# detector resizes the page into (docTR: 1024, the fast_base training size).
DEFAULT_PARAMS: dict[str, Any] = {"device": "cuda:0", "detection_size": 1024}


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {
        **DEFAULT_PARAMS,
        **{key: value for key, value in params.items() if value is not None},
    }
    device = merged["device"]
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+|cpu", device):
        raise AdapterError(f"device must be 'cuda:<n>' or 'cpu': {device!r}")
    size = merged["detection_size"]
    if type(size) is not int or size < 32 or size % 32:
        raise AdapterError(f"detection_size must be a positive multiple of 32: {size!r}")
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_checksums() -> dict[str, str]:
    """``models/<file>``: SHA-256, from models.sha256."""
    lines = CHECKSUMS.read_text(encoding="utf-8").splitlines()
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}


def model_files() -> dict[str, str]:
    """SHA-256 of every pinned model file; raises AdapterError unless all are present and match."""
    directory = CACHE / MODELS_DIR
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make doctr-model and run with --cache (README.md)"
        )
    files, bad = {}, []
    for name, digest in pinned_checksums().items():
        path = directory / name
        files[name] = sha256(path) if path.is_file() else None
        if files[name] != digest:
            bad.append(name)
    if bad:
        raise AdapterError(f"model files missing or not matching models.sha256: {bad}")
    return files


def environment(device: str) -> dict[str, Any]:
    """Versions and GPU diagnostics; raises AdapterError when the requested GPU is unusable."""
    try:
        import torch
    except Exception as error:
        raise AdapterError(f"cannot import PyTorch: {error}") from error
    info: dict[str, Any] = {
        "doctr": metadata.version("python-doctr"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
    }
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        count = torch.cuda.device_count()
        if index >= count:
            raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
        info["gpu"] = torch.cuda.get_device_name(index)
        info["gpu_memory_mb"] = torch.cuda.get_device_properties(index).total_memory >> 20
    return info


def load_models(params: dict[str, Any]) -> Any:
    """docTR's OCR predictor with the pinned pretrained pair, from the local checkpoints only.

    docTR finds the checkpoints in DOCTR_CACHE_DIR and verifies their hash prefix instead of
    downloading them. Everything else keeps docTR's defaults: straight pages, no orientation,
    language or layout models, words grouped into lines. A detection_size other than the
    model's own input size replaces the detector's pre-processor with one built as docTR's does,
    for that size.
    """
    os.environ["DOCTR_CACHE_DIR"] = str(CACHE / MODELS_DIR)
    import torch
    from doctr.models import ocr_predictor
    from doctr.models.preprocessor import PreProcessor

    predictor = ocr_predictor(ARCHS["detection"], ARCHS["recognition"], pretrained=True)
    detector, size = predictor.det_predictor, params["detection_size"]
    if tuple(detector.model.cfg["input_shape"][1:]) != (size, size):
        old = detector.pre_processor
        detector.pre_processor = PreProcessor(
            (size, size),
            batch_size=old.batch_size,
            mean=detector.model.cfg["mean"],
            std=detector.model.cfg["std"],
            preserve_aspect_ratio=old.resize.preserve_aspect_ratio,
            symmetric_pad=old.resize.symmetric_pad,
        )
    return predictor.to(torch.device(params["device"]))


def read_image(path: Path) -> Any:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode == "RGB" else image.convert("RGB")


def run_doctr(predictor: Any, image: Any) -> dict[str, Any]:
    """``doctr.json``: docTR's export of the page, with its blocks in docTR's reading order."""
    import numpy as np

    document = predictor([np.asarray(image)])
    return document.pages[0].export()


def polygon(geometry: list, size: tuple[int, int]) -> list[list[float]]:
    """Pixel polygon of a docTR geometry, relative to the page: a box ``((x0, y0), (x1, y1))``
    becomes its four corners, a rotated box keeps its points."""
    width, height = size
    points = [(float(x) * width, float(y) * height) for x, y in geometry]
    if len(points) == 2:
        (x0, y0), (x1, y1) = points
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [[round(x, 2), round(y, 2)] for x, y in points]


def result_to_prediction(
    native: dict[str, Any], sample_id: str, size: tuple[int, int]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Standardized prediction (docs/PREDICTIONS.md) from ``doctr.json``, and counts.

    Every docTR block becomes a region and every line a line, in docTR's order, with their
    boxes in input-image pixels. A line's text is its words joined by spaces; lines without
    text are dropped, and so are blocks left without lines.
    """
    height, width = native["dimensions"]
    if (width, height) != size:
        raise AdapterError(f"docTR result is for a {width}x{height} image")
    regions, words, lines = [], 0, 0
    for block in native["blocks"]:
        entries = []
        for line in block["lines"]:
            words += len(line["words"])
            text = " ".join(w["value"] for w in line["words"] if w["value"].strip())
            if text:
                lines += 1
                entries.append(
                    {"id": f"l{lines}", "text": text, "polygon": polygon(line["geometry"], size)}
                )
        if entries:
            regions.append(
                {
                    "id": f"r{len(regions) + 1}",
                    "type": "text",
                    "polygon": polygon(block["geometry"], size),
                    "lines": entries,
                }
            )
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": width, "height": height},
        "text": "\n\n".join("\n".join(e["text"] for e in r["lines"]) for r in regions),
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    counts = {"blocks": len(native["blocks"]), "words": words, "lines": lines}
    return prediction, counts | {"regions": len(regions)}


def process(predictor: Any, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Run docTR on one sample, write its outputs and return ``result.json`` content."""
    image = read_image(source / sample["image"])
    start = time.monotonic()
    native = run_doctr(predictor, image)
    seconds = round(time.monotonic() - start, 3)
    (out / "doctr.json").write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    prediction, counts = result_to_prediction(native, sample["id"], image.size)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        "status": "ok",
        "primary": "prediction.json",
        "extra": counts | {"doctr_seconds": seconds},
    }


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    predictor, params, info, files, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        files = model_files()
        predictor = load_models(params)
        info["detection_input"] = list(predictor.det_predictor.pre_processor.resize.size)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load docTR: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {
        "archs": ARCHS,
        "checkpoints": CHECKPOINTS,
        "sources": SOURCES,
        "files": files,
        "job_model": job.get("model"),
    }
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(predictor, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=model_info, params=params)
        write_result(out, result)
    print(f"doctr adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
