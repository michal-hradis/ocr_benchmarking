"""Indra OCR CPU pipeline adapter; runs in the engine image.

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the pinned
Indra OCR models from the image once (checked against models.sha256), then for every sample
image in job order run Indra's image OCR (orientation and skew, line detection, script routing
and recognition, layout blocks and reading order) and write $OCRBENCH_OUTPUT/<id>/ with Indra's
native page JSON (indra.json, the fields of ``indra-ocr --format json``), the standardized
prediction.json and, last, result.json. Every layout block becomes a region with the lines it
consists of, in Indra's reading order (README.md).

Only Indra's raster-image path is used: the adapter passes the decoded image to the OCR
pipeline, so a PDF text layer is never read.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

MODELS = Path(os.environ.get("OCRBENCH_INDRA_MODELS", "/opt/ocrbench/models"))
CHECKSUMS = Path(os.environ.get("OCRBENCH_INDRA_CHECKSUMS", "/opt/ocrbench/models.sha256"))
CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
MODEL_REPOSITORY = "datalab-to/indra-ocr"
MODEL_TAG = "v1.1.0"  # indra_ocr.models.HF_REVISION of Indra OCR 1.1.2
MODEL_COMMIT = os.environ.get("OCRBENCH_INDRA_MODEL_COMMIT", "")
# Engine parameters (job.json "params"), README.md.
DEFAULT_PARAMS: dict[str, Any] = {"threads": 4}
# Indra layout block label -> standardized region type (docs/PREDICTIONS.md#region-types); any
# other label becomes "other".
REGION_TYPES = {
    "Text": "text",
    "Section-Header": "heading",
    "Page-Header": "page-header",
    "Page-Footer": "page-footer",
    "List-Group": "text",
    "Table": "table",
    "Caption": "caption",
    "Footnote": "footnote",
    "Picture": "figure",
}


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    threads = merged["threads"]
    if type(threads) is not int or threads < 1:
        raise AdapterError(f"threads must be an integer of at least 1: {threads!r}")
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files() -> dict[str, str]:
    """SHA-256 of every pinned model file; raises AdapterError unless all are present and match."""
    lines = CHECKSUMS.read_text(encoding="utf-8").splitlines()
    pinned = {name: digest for digest, name in (line.split() for line in lines if line.strip())}
    files, bad = {}, []
    for name, digest in pinned.items():
        path = MODELS / name
        files[name] = sha256(path) if path.is_file() else None
        if files[name] != digest:
            bad.append(name)
    if bad:
        raise AdapterError(f"model files missing or not matching models.sha256: {bad}")
    return files


def cpu_quota() -> str | None:
    """The container's cgroup v2 CPU limit (``cpu.max``, quota and period), when readable."""
    try:
        return CPU_MAX.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def environment(threads: int) -> dict[str, Any]:
    """Package versions and the CPU allocation the pipeline runs with: its threads, the CPUs
    the process may run on and the container's CPU quota (Docker's --cpus)."""
    info: dict[str, Any] = {
        name: metadata.version(name)
        for name in ("indra-ocr", "onnxruntime", "opencv-python-headless", "numpy")
    }
    info["threads"] = threads
    info["affinity_cpus"] = len(os.sched_getaffinity(0))
    info["cpu_max"] = cpu_quota()
    return info


def load_converter(threads: int) -> Any:
    """Indra's converter on the pinned local models: OCR mode, no math, CPU only."""
    import cv2
    import indra_ocr

    cv2.setNumThreads(threads)  # OpenCV's own pool; ONNX Runtime gets ``threads`` below
    return indra_ocr.load(
        path=MODELS,
        threads=threads,
        mode="ocr",
        math=False,
        furniture=False,
        providers=["CPUExecutionProvider"],
    )


def read_image(path: Path) -> Any:
    """The image as stored, in grayscale as Indra reads it (EXIF orientation is not applied,
    docs/PREDICTIONS.md#positions)."""
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if gray is None:
        raise AdapterError(f"cannot decode image {path.name}")
    return gray


def page_json(page: Any, name: str) -> dict[str, Any]:
    """``indra.json``: the page as ``indra-ocr --format json`` writes it."""
    return {
        "file": name,
        "page": 0,
        "source": page.source,
        "rotation": page.rotation,
        "skew": page.skew,
        "vertical": page.vertical,
        "note": page.note,
        "lines": [asdict(line) for line in page.lines],
        "markdown": page.markdown,
        "blocks": page.blocks,
        "timings": page.timings,
    }


def original_polygon(box: list[float], rotation: int, size: tuple[int, int]) -> list[list[float]]:
    """The 4-point polygon of an upright-page box ``x0, y0, x1, y1`` in original image pixels.

    Indra turns a page whose content is turned by ``rotation`` degrees counter-clockwise by that
    many degrees clockwise before reading it, and reports boxes on the turned page. Each quarter
    turn is undone counter-clockwise: a point ``(x, y)`` of a ``w`` wide image goes to
    ``(y, w - x)``.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    turns = rotation // 90
    width = size[1] if turns % 2 else size[0]  # of the turned page
    height = size[0] if turns % 2 else size[1]
    for _ in range(turns):
        points = [[y, width - x] for x, y in points]
        width, height = height, width
    return points


def page_to_prediction(
    native: dict[str, Any], sample_id: str, size: tuple[int, int]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Standardized prediction (docs/PREDICTIONS.md) from ``indra.json``, and counts.

    Indra's blocks partition its reading-ordered lines into consecutive runs (a block's text is
    its lines' texts joined by line breaks), so every non-picture block takes the next lines in
    reading order. Each block becomes a region with its box, label and lines, in Indra's block
    order; each line keeps its box and confidence. A picture block is a ``figure`` region
    without text. A page without layout blocks (Indra reads fewer than two lines without its
    layout model) keeps its lines in one region without a position. Boxes are mapped back to the
    original image when Indra turned the page. The page text is the regions' lines, with a blank
    line between regions.
    """
    if native["source"] != "ocr":
        raise AdapterError(f"Indra page source is {native['source']!r}: {native['note']}")
    rotation = 0 if native["vertical"] else native["rotation"]
    if rotation not in (0, 90, 180, 270):
        raise AdapterError(f"unexpected page rotation {rotation!r}")
    lines = native["lines"]
    line_ids = [f"l{index + 1}" for index in range(len(lines))]

    def entry(index: int) -> dict[str, Any] | None:
        line = lines[index]
        if not line["text"].strip():
            return None
        return {
            "id": line_ids[index],
            "text": line["text"],
            "polygon": original_polygon(line["box"], rotation, size),
            "confidence": line["conf"],
        }

    def entries(indices: range) -> list[dict[str, Any]]:
        return [line for line in map(entry, indices) if line is not None]

    regions: list[dict[str, Any]] = []
    used = 0
    for number, block in enumerate(native["blocks"]):
        label = block["label"]
        region: dict[str, Any] = {
            "id": f"r{number + 1}",
            "type": REGION_TYPES.get(label, "other"),
            "native_type": label,
            "polygon": original_polygon(block["bbox"], rotation, size),
        }
        if label != "Picture":
            count = block["text"].count("\n") + 1
            members = range(used, used + count)
            if used + count > len(lines) or block["text"] != "\n".join(
                lines[i]["text"] for i in members
            ):
                raise AdapterError(f"block {number + 1} does not match the next {count} lines")
            used += count
            if block_lines := entries(members):
                region["lines"] = block_lines
        regions.append(region)
    if used != len(lines):
        if native["blocks"]:
            raise AdapterError(f"the layout blocks hold {used} of {len(lines)} lines")
        if block_lines := entries(range(len(lines))):
            regions.append({"id": "r1", "lines": block_lines, "extra": {"grouping": "no-layout"}})
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": size[0], "height": size[1]},
        "text": "\n\n".join(
            "\n".join(line["text"] for line in region["lines"])
            for region in regions
            if "lines" in region
        ),
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    counts = {
        "lines": len(lines),
        "layout_blocks": len(native["blocks"]),
        "regions": len(regions),
        "rotation": native["rotation"],
    }
    return prediction, counts


def process(converter: Any, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Run Indra OCR on one sample, write its outputs and return ``result.json`` content."""
    path = source / sample["image"]
    gray = read_image(path)
    start = time.monotonic()
    page = converter.ocr(gray)
    seconds = round(time.monotonic() - start, 3)
    native = page_json(page, path.name)
    (out / "indra.json").write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    size = (int(gray.shape[1]), int(gray.shape[0]))
    prediction, counts = page_to_prediction(native, sample["id"], size)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        "status": "ok",
        "primary": "prediction.json",
        "extra": counts | {"indra_seconds": seconds, "timings": page.timings},
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
    converter, params, info, files, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["threads"])
        files = model_files()
        converter = load_converter(params["threads"])
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load Indra OCR: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {
        "repository": MODEL_REPOSITORY,
        "tag": MODEL_TAG,
        "commit": MODEL_COMMIT,
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
            result = process(converter, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=model_info, params=params)
        write_result(out, result)
    print(f"indra-ocr adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
