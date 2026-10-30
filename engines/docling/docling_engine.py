"""Docling engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, check the
pinned models in the /cache volume, build one Docling DocumentConverter for image input with the
standard pipeline (layout analysis, OCR of the layout regions, TableFormer, reading order) and
convert every sample image in job order. Each image is decoded as stored (EXIF orientation not
applied) and handed to Docling as an uncompressed TIFF at a fixed nominal resolution of 72 x
ocr_scale DPI, so that OCR sees the original pixels 1:1 whatever the file's own DPI says. Every
sample directory gets the native DoclingDocument (docling.json), its Markdown export
(docling.md), Docling's layout clusters with their OCR cells (docling_layout.json), the
standardized prediction.json and, last, result.json.

The prediction (``to_prediction``) has one region per provenance box of every DoclingDocument
item, in the document's reading order, and as lines the OCR cells of the layout cluster the box
came from: their raw OCR text and their boxes, mapped from Docling's page coordinates (points) to
the pixels of the input image. Docling's own item text, which joins the lines, removes hyphens at
line ends and replaces typographic quotes, stays in the native files.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

# The models are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_DOCLING_CACHE", "/cache"))
MODELS_DIR = "models/docling"  # relative to the cache: Docling's artifacts_path
CHECKSUMS = Path(os.environ.get("OCRBENCH_DOCLING_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# Engine parameters (job.json "params"), README.md.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "ocr_engine": "rapidocr",
    "ocr_lang": ["latin"],
    "ocr_mode": "default",
    "ocr_scale": 3.0,
    "table_structure": True,
    "document_timeout": None,
    "num_threads": 4,
}
OCR_ENGINES = ("rapidocr",)  # further OCR engines come with their presets (#78-#81)
OCR_MODES = ("default", "layout_regions", "full_page")
POINTS_PER_INCH = 72.0
# Docling item label: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "text": "text",
    "paragraph": "text",
    "reference": "text",
    "handwritten_text": "text",
    "title": "heading",
    "section_header": "heading",
    "list_item": "list-item",
    "caption": "caption",
    "footnote": "footnote",
    "page_header": "page-header",
    "page_footer": "page-footer",
    "table": "table",
    "document_index": "table",
    "formula": "formula",
    "picture": "figure",
    "chart": "figure",
}
# A provenance box and the layout cluster it came from describe the same rectangle; they differ
# only by the change of origin, computed in floating point (points).
BOX_TOLERANCE = 1e-3
MAX_WARNINGS = 50  # log records kept per sample in result.json


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    device = merged["device"]
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+|cpu", device):
        raise AdapterError(f"device must be 'cuda:<n>' or 'cpu': {device!r}")
    if merged["ocr_engine"] not in OCR_ENGINES:
        raise AdapterError(f"ocr_engine must be one of {OCR_ENGINES}: {merged['ocr_engine']!r}")
    lang = merged["ocr_lang"]
    if not isinstance(lang, list) or len(lang) != 1 or not isinstance(lang[0], str):
        raise AdapterError(f"ocr_lang must be a list of one language code: {lang!r}")
    if merged["ocr_mode"] not in OCR_MODES:
        raise AdapterError(f"ocr_mode must be one of {OCR_MODES}: {merged['ocr_mode']!r}")
    scale = merged["ocr_scale"]
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 10:
        raise AdapterError(f"ocr_scale must be a number with 0 < value <= 10: {scale!r}")
    if type(merged["table_structure"]) is not bool:
        raise AdapterError(f"table_structure must be true or false: {merged['table_structure']!r}")
    timeout = merged["document_timeout"]
    if timeout is not None and (
        type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
    ):
        raise AdapterError(f"document_timeout must be a positive number or null: {timeout!r}")
    threads = merged["num_threads"]
    if type(threads) is not int or threads < 1:
        raise AdapterError(f"num_threads must be a positive integer: {threads!r}")
    return merged


def nominal_dpi(params: dict[str, Any]) -> float:
    """The resolution the adapter declares: Docling renders OCR input at 72 DPI x ocr_scale."""
    return POINTS_PER_INCH * params["ocr_scale"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_checksums() -> dict[str, str]:
    """``<file below the artifacts directory>``: SHA-256, from models.sha256."""
    lines = CHECKSUMS.read_text(encoding="utf-8").splitlines()
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}


def model_files() -> dict[str, str]:
    """SHA-256 of every pinned model file; raises AdapterError unless all are present and match."""
    directory = CACHE / MODELS_DIR
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the models with "
            "make docling-model and run with --cache (README.md)"
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
        name: metadata.version(name)
        for name in ("docling-slim", "docling-core", "docling-ibm-models", "rapidocr")
    }
    info |= {
        "onnxruntime": metadata.version("onnxruntime"),
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


def pipeline_options(params: dict[str, Any]) -> Any:
    """Docling's standard pipeline options for the preset; everything else keeps the defaults.

    The models come from the artifacts directory only, remote services and external plugins are
    off, and RapidOCR runs on the CPU (onnxruntime) whatever the layout device is: the image has
    the CPU build of onnxruntime, and Docling would otherwise ask it for CUDA.
    """
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions

    ocr = RapidOcrOptions(
        lang=params["ocr_lang"],
        backend="onnxruntime",
        mode=OcrMode(params["ocr_mode"]),
        scale=params["ocr_scale"],
        rapidocr_params={
            "EngineConfig.onnxruntime.use_cuda": False,
            "Det.use_cuda": False,
            "Cls.use_cuda": False,
            "Rec.use_cuda": False,
        },
    )
    return PdfPipelineOptions(
        artifacts_path=str(CACHE / MODELS_DIR),
        accelerator_options=AcceleratorOptions(
            device=params["device"], num_threads=params["num_threads"]
        ),
        do_ocr=True,
        ocr_options=ocr,
        do_table_structure=params["table_structure"],
        document_timeout=params["document_timeout"],
        enable_remote_services=False,
        allow_external_plugins=False,
    )


def load_converter(params: dict[str, Any]) -> Any:
    """A DocumentConverter for image input only, with the pipeline loaded now, not on first use."""
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, ImageFormatOption

    converter = DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options(params))
        },
    )
    converter.initialize_pipeline(InputFormat.IMAGE)
    return converter


def image_stream(path: Path, dpi: float) -> tuple[io.BytesIO, tuple[int, int]]:
    """The decoded image as an uncompressed TIFF declaring ``dpi``, and its size.

    Pillow does not apply the EXIF orientation when it opens an image, and the TIFF carries no
    EXIF, so Docling (which would apply it) sees the pixels as stored
    (docs/PREDICTIONS.md#positions). The TIFF stores the resolution as an exact rational, so a
    page is ``size * 72 / dpi`` points.
    """
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
            size = image.size
            stream = io.BytesIO()
            image.save(stream, format="TIFF", dpi=(dpi, dpi))
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    stream.seek(0)
    return stream, size


def layout_dump(result: Any) -> dict[str, Any]:
    """Docling's layout clusters of every page, with their OCR cells and child clusters."""
    pages = []
    for page in result.pages:
        layout = page.predictions.layout
        pages.append(
            {
                "page_no": page.page_no,
                "size": page.size.model_dump(mode="json") if page.size else None,
                "clusters": [c.model_dump(mode="json") for c in layout.clusters] if layout else [],
            }
        )
    return {"pages": pages}


def corners(box: dict[str, Any], height: float) -> tuple[float, float, float, float]:
    """``(left, top, right, bottom)`` of a Docling bounding box, with the origin at the top."""
    if box["coord_origin"] == "TOPLEFT":
        return box["l"], box["t"], box["r"], box["b"]
    if box["coord_origin"] == "BOTTOMLEFT":
        return box["l"], height - box["t"], box["r"], height - box["b"]
    raise AdapterError(f"unknown coord_origin {box['coord_origin']!r}")


def rect_points(rect: dict[str, Any], height: float) -> list[tuple[float, float]]:
    """The four corners of a Docling bounding rectangle (a cell), with the origin at the top."""
    points = [(rect[f"r_x{i}"], rect[f"r_y{i}"]) for i in range(4)]
    if rect["coord_origin"] == "TOPLEFT":
        return points
    if rect["coord_origin"] == "BOTTOMLEFT":
        return [(x, height - y) for x, y in points]
    raise AdapterError(f"unknown coord_origin {rect['coord_origin']!r}")


def walk(document: dict[str, Any]) -> list[dict[str, Any]]:
    """The items of a DoclingDocument JSON in document order.

    Pre-order from the body (and then the furniture root): an item, then its children, as
    ``DoclingDocument.iterate_items`` reads them with every content layer included and with the
    children of pictures (text that Docling recognized inside a picture). Groups contribute only
    their children.
    """

    def resolve(ref: str) -> dict[str, Any]:
        parts = ref.lstrip("#/").split("/")
        node: Any = document
        for part in parts:
            node = node[int(part)] if isinstance(node, list) else node[part]
        return node

    items, seen = [], set()

    def visit(node: dict[str, Any]) -> None:
        if node["self_ref"] in seen:
            raise AdapterError(f"{node['self_ref']} appears twice in the document tree")
        seen.add(node["self_ref"])
        if "prov" in node:
            items.append(node)
        for child in node.get("children", []):
            visit(resolve(child["$ref"]))

    visit(document["body"])
    visit(document["furniture"])
    return items


def flatten(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every cluster, its children after it, depth first."""
    flat = []
    for cluster in clusters:
        flat.append(cluster)
        flat.extend(flatten(cluster.get("children", [])))
    return flat


def match_cluster(
    box: tuple[float, float, float, float],
    label: str,
    clusters: list[dict[str, Any]],
    height: float,
    used: set[int],
) -> dict[str, Any] | None:
    """The unused cluster with the same box (and preferably the same label), or None."""
    candidates = [
        c
        for c in clusters
        if id(c) not in used
        and all(
            abs(a - b) <= BOX_TOLERANCE
            for a, b in zip(box, corners(c["bbox"], height), strict=True)
        )
    ]
    for cluster in candidates:
        if cluster["label"] == label:
            return cluster
    return candidates[0] if candidates else None


def to_prediction(
    document: dict[str, Any], layout: dict[str, Any], sample_id: str, size: tuple[int, int]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Standardized prediction (docs/PREDICTIONS.md) from Docling's native output.

    ``document`` is the DoclingDocument JSON, ``layout`` the layout clusters of its page
    (``layout_dump``) and ``size`` the input image in pixels. Every provenance box of every item
    (in ``walk`` order) becomes a region with the item's label as native type; the box is
    matched to the layout cluster it came from, and the region's lines are that cluster's OCR
    cells with non-blank text (in Docling's cell order), except cells that belong to a child
    cluster which is itself an item's box (text inside a picture). Positions are converted from
    Docling's page coordinates to input pixels by ``size / page size`` on each axis. Also
    returns counts of regions, lines and of OCR cells not in any region.
    """
    width, height = size
    if len(document["pages"]) != 1 or len(layout["pages"]) != 1:
        raise AdapterError(f"expected one page, got {len(document['pages'])}")
    page = next(iter(document["pages"].values()))
    page_no = page["page_no"]
    page_w, page_h = page["size"]["width"], page["size"]["height"]
    scale_x, scale_y = width / page_w, height / page_h
    native_page = layout["pages"][0]
    if native_page["page_no"] != page_no or native_page["size"] != page["size"]:
        raise AdapterError("the layout clusters belong to another page")
    clusters = flatten(native_page["clusters"])

    def pixels(x: float, y: float) -> list[float]:
        return [round(x * scale_x, 3), round(y * scale_y, 3)]

    # Pass 1: the cluster behind every provenance box.
    used: set[int] = set()
    boxes = []
    for item in walk(document):
        for number, prov in enumerate(item["prov"]):
            if prov["page_no"] != page_no:
                raise AdapterError(f"{item['self_ref']} refers to page {prov['page_no']}")
            box = corners(prov["bbox"], page_h)
            cluster = match_cluster(box, item["label"], clusters, page_h, used)
            if cluster is None:
                raise AdapterError(f"no layout cluster matches {item['self_ref']} box {number}")
            used.add(id(cluster))
            boxes.append((item, number, box, cluster))

    def owned(cluster: dict[str, Any]) -> set[int]:
        """Cell indices of the matched clusters below ``cluster``."""
        cells: set[int] = set()
        for child in cluster.get("children", []):
            if id(child) in used:
                cells.update(cell["index"] for cell in child["cells"])
            cells |= owned(child)
        return cells

    # Pass 2: regions and lines.
    regions, assigned = [], set()
    for number, (item, prov_number, (left, top, right, bottom), cluster) in enumerate(boxes):
        excluded = owned(cluster)
        region: dict[str, Any] = {
            "id": f"r{number + 1}",
            "type": REGION_TYPES.get(item["label"], "other"),
            "native_type": item["label"],
            "polygon": [
                pixels(left, top),
                pixels(right, top),
                pixels(right, bottom),
                pixels(left, bottom),
            ],
        }
        lines = []
        for cell in cluster["cells"]:
            if cell["index"] in excluded or not cell["text"].strip():
                continue
            if cell["index"] in assigned:
                raise AdapterError(f"OCR cell {cell['index']} is in two regions")
            assigned.add(cell["index"])
            line: dict[str, Any] = {
                "id": f"l{cell['index']}",
                "text": cell["text"],
                "polygon": [pixels(x, y) for x, y in rect_points(cell["rect"], page_h)],
            }
            if cell.get("confidence") is not None:
                line["confidence"] = round(float(cell["confidence"]), 4)
            lines.append(line)
        if lines:
            region["lines"] = lines
        region["extra"] = {
            "self_ref": item["self_ref"],
            "prov": prov_number,
            "cluster": cluster["id"],
            "layout_confidence": round(float(cluster["confidence"]), 4),
        }
        regions.append(region)
    text = "\n\n".join(
        "\n".join(line["text"] for line in region["lines"])
        for region in regions
        if "lines" in region
    )
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": width, "height": height},
        "text": text,
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    cells = {cell["index"] for c in clusters for cell in c["cells"] if cell["text"].strip()}
    counts = {
        "regions": len(regions),
        "lines": len(assigned),
        "cells": len(cells),
        "cells_outside_regions": len(cells - assigned),
    }
    return prediction, counts


def check_page_size(document: dict[str, Any], size: tuple[int, int], dpi: float) -> None:
    """The page must be the input image at ``dpi``: anything else means another transformation."""
    for page in document["pages"].values():
        expected = (size[0] * POINTS_PER_INCH / dpi, size[1] * POINTS_PER_INCH / dpi)
        actual = (page["size"]["width"], page["size"]["height"])
        if any(abs(a - e) > 1e-6 * max(e, 1) for a, e in zip(actual, expected, strict=True)):
            raise AdapterError(
                f"Docling page is {actual[0]:.3f} x {actual[1]:.3f} pt, expected "
                f"{expected[0]:.3f} x {expected[1]:.3f} pt for {size[0]} x {size[1]} px"
            )


class Warnings(logging.Handler):
    """Collects the warnings and errors that Docling and RapidOCR log during one sample."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[str] = []
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.records) < MAX_WARNINGS:
            self.records.append(f"{record.name}: {record.getMessage()}"[:500])
        else:
            self.dropped += 1


def process(
    converter: Any, params: dict[str, Any], sample: dict[str, Any], source: Path, out: Path
) -> dict[str, Any]:
    """Convert one sample, write its outputs and return the ``result.json`` content."""
    import torch
    from docling.datamodel.base_models import ConversionStatus, DocumentStream

    dpi = nominal_dpi(params)
    stream, size = image_stream(source / sample["image"], dpi)
    cuda = params["device"].startswith("cuda:")
    if cuda:
        torch.cuda.reset_peak_memory_stats(params["device"])
    collector = Warnings()
    loggers = [logging.getLogger(name) for name in ("docling", "RapidOCR")]
    for logger in loggers:
        logger.addHandler(collector)
    try:
        result = converter.convert(
            DocumentStream(name=f"{sample['id']}.tif", stream=stream), raises_on_error=False
        )
    finally:
        for logger in loggers:
            logger.removeHandler(collector)
    errors = [f"{e.component_type}: {e.error_message}" for e in result.errors]
    extra: dict[str, Any] = {
        "conversion_status": result.status.value,
        "errors": errors,
        "warnings": collector.records,
        "warnings_dropped": collector.dropped,
        "nominal_dpi": dpi,
    }
    if cuda:
        extra["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(params["device"]) >> 20
    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        message = "; ".join(errors) or f"conversion status {result.status.value}"
        return {"status": "failed", "message": f"Docling: {message}"[:2000], "extra": extra}
    document = result.document.export_to_dict()
    layout = layout_dump(result)
    (out / "docling.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    (out / "docling.md").write_text(result.document.export_to_markdown(), encoding="utf-8")
    (out / "docling_layout.json").write_text(
        json.dumps(layout, ensure_ascii=False), encoding="utf-8"
    )
    check_page_size(document, size, dpi)
    prediction, counts = to_prediction(document, layout, sample["id"], size)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    extra["counts"] = counts
    if result.status is ConversionStatus.PARTIAL_SUCCESS:
        message = "; ".join(errors) or "Docling reported a partial conversion"
        return {
            "status": "partial",
            "primary": "prediction.json",
            "message": message[:2000],
            "extra": extra,
        }
    return {"status": "ok", "primary": "prediction.json", "extra": extra}


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    converter, params, info, files, options, init_error = None, None, {}, None, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        files = model_files()
        converter = load_converter(params)
        options = json.loads(pipeline_options(params).model_dump_json())
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load Docling: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {"files": files, "job_model": job.get("model")}
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(converter, params, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"[:2000]}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(
            environment=info, models=model_info, params=params, pipeline_options=options
        )
        write_result(out, result)
    print(f"docling adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
