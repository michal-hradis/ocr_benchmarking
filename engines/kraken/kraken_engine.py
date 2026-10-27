"""Kraken historical OCR/HTR engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the Kraken
baseline segmentation and line recognition models once, run segmentation and recognition on every
sample image in job order and write $OCRBENCH_OUTPUT/<id>/ with Kraken's native PAGE XML
(kraken.xml) and segmentation JSON with the recognized text (kraken.json), the standardized
prediction.json and, last, result.json. Regions keep Kraken's region types and its reading order.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
import re
import sys
import time
from importlib import metadata, resources
from pathlib import Path
from typing import Any

MODELS_DIR = Path(os.environ.get("OCRBENCH_KRAKEN_MODELS", "/opt/ocrbench/models"))
CACHE_DIR = "/cache/"
# Recognition models in the image (checked against models.sha256 when it is built): their record,
# license and the material they suit, recorded with every sample. Not interchangeable (README.md).
BUNDLED_MODELS: dict[str, dict[str, Any]] = {
    "catmus-print-fondue-large.mlmodel": {
        "record": "https://doi.org/10.5281/zenodo.10592716",
        "license": "CC-BY-4.0",
        "scripts": ["Latn"],
        "material": "print",
        "suitable_for": "Latin-script printed books and newspapers, 16th-21st c.; no háček letters",
    },
    "McCATMuS_nfd_nofix_V1.mlmodel": {
        "record": "https://doi.org/10.5281/zenodo.13788177",
        "license": "CC-BY-4.0",
        "scripts": ["Latn"],
        "material": "handwriting, print, typewriting",
        "suitable_for": "Latin-script manuscripts (HTR), 16th-21st c.; no háček letters",
    },
    "german_print.mlmodel": {
        "record": "https://doi.org/10.5281/zenodo.10519596",
        "license": "CC0-1.0",
        "scripts": ["Latf", "Latn"],
        "material": "print",
        "suitable_for": "German and Latin prints, incunabula to 20th c., Fraktur",
    },
}
# Engine parameters (job.json "params"), README.md.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "recognition_model": "catmus-print-fondue-large.mlmodel",
    "segmentation_model": None,  # Kraken's default blla.mlmodel
    "batch_size": 1,
    "segmentation_min_height": 1200,
    "segmentation_tile_height": 0,  # 0: segment the whole page at once
}
INT_PARAMS = {"batch_size": 1, "segmentation_min_height": 0, "segmentation_tile_height": 0}
MIN_TILE_HEIGHT = 600
# Kraken region class -> standardized region type (docs/PREDICTIONS.md#region-types), by the
# lower-cased class without a SegmOnto ":subtype"; any other class becomes "other".
REGION_TYPES = {
    "text": "text",
    "paragraph": "text",
    "maintext": "text",
    "mainzone": "text",
    "margintextzone": "text",
    "heading": "heading",
    "title": "heading",
    "titlepagezone": "heading",
    "caption": "caption",
    "footnote": "footnote",
    "header": "page-header",
    "runningtitlezone": "page-header",
    "footer": "page-footer",
    "pagenumber": "page-number",
    "numberingzone": "page-number",
    "table": "table",
    "tablezone": "table",
    "illustration": "figure",
    "image": "figure",
    "figure": "figure",
    "graphiczone": "figure",
}


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
    for key, low in INT_PARAMS.items():
        value = merged[key]
        if type(value) is not int or value < low:
            raise AdapterError(f"{key} must be an integer of at least {low}: {value!r}")
    if 0 < merged["segmentation_tile_height"] < MIN_TILE_HEIGHT:
        raise AdapterError(f"segmentation_tile_height must be 0 or at least {MIN_TILE_HEIGHT}")
    model = merged["recognition_model"]
    if not isinstance(model, str) or not (model in BUNDLED_MODELS or _cache_path(model)):
        bundled = ", ".join(BUNDLED_MODELS)
        raise AdapterError(f"recognition_model must be one of {bundled} or a {CACHE_DIR} path")
    model = merged["segmentation_model"]
    if model is not None and not (isinstance(model, str) and _cache_path(model)):
        raise AdapterError(
            f"segmentation_model must be null (Kraken's default) or a {CACHE_DIR} path"
        )
    return merged


def _cache_path(value: str) -> bool:
    return value.startswith(CACHE_DIR) and ".." not in Path(value).parts


def model_paths(params: dict[str, Any]) -> dict[str, Path]:
    recognition = params["recognition_model"]
    segmentation = params["segmentation_model"]
    return {
        "segmentation": Path(segmentation or resources.files("kraken").joinpath("blla.mlmodel")),
        "recognition": Path(recognition if "/" in recognition else MODELS_DIR / recognition),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment(device: str) -> dict[str, Any]:
    """Versions and GPU diagnostics; raises AdapterError when the requested GPU is unusable."""
    try:
        import kraken  # noqa: F401
        import torch
    except Exception as error:
        raise AdapterError(f"cannot import Kraken/PyTorch: {error}") from error
    info: dict[str, Any] = {
        "kraken": metadata.version("kraken"),
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
    return info


def load_models(params: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Segmentation and recognition task models, and their files with checksums."""
    from kraken.tasks import RecognitionTaskModel, SegmentationTaskModel

    paths = model_paths(params)
    for role, path in paths.items():
        if not path.is_file():
            raise AdapterError(f"{role} model {path} does not exist (models: README.md)")
    files = {role: {"path": str(path), "sha256": sha256(path)} for role, path in paths.items()}
    if bundled := BUNDLED_MODELS.get(params["recognition_model"]):
        files["recognition"] |= bundled
    try:
        segmenter = SegmentationTaskModel.load_model(paths["segmentation"])
        recognizer = RecognitionTaskModel.load_model(paths["recognition"])
    except ValueError as error:  # not a model, or a model for another task
        raise AdapterError(f"unsupported model: {error}") from error
    if recognizer.seg_type == "bbox":
        raise AdapterError(
            "unsupported model: the recognition model was trained on bounding-box lines; "
            "only baseline models are supported"
        )
    return segmenter, recognizer, files


def inference_configs(params: dict[str, Any]) -> tuple[Any, Any]:
    from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig

    device = params["device"]
    target = {"accelerator": "cpu", "device": "auto"}
    if device.startswith("cuda:"):
        target = {"accelerator": "gpu", "device": [int(device.split(":", 1)[1])]}
    return (
        SegmentationInferenceConfig(**target),
        RecognitionInferenceConfig(**target, batch_size=params["batch_size"]),
    )


def read_image(path: Path) -> Any:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode in ("L", "RGB") else image.convert("RGB")


def segmentation_canvas(image: Any, min_height: int) -> tuple[Any, int]:
    """The image centred vertically on a canvas of at least ``min_height`` pixels, and its offset.

    The default segmentation model scales every page to a fixed height, so a short image such as
    a paragraph crop would be enlarged until no lines are found. The canvas is filled with the
    median colour of the image's top and bottom rows.
    """
    width, height = image.size
    if height >= min_height:
        return image, 0
    from PIL import Image, ImageStat

    rows = Image.new(image.mode, (width, 2))
    rows.paste(image.crop((0, 0, width, 1)), (0, 0))
    rows.paste(image.crop((0, height - 1, width, height)), (0, 1))
    fill = tuple(int(v) for v in ImageStat.Stat(rows).median)
    canvas = Image.new(image.mode, (width, min_height), fill if len(fill) > 1 else fill[0])
    offset = (min_height - height) // 2
    canvas.paste(image, (0, offset))
    return canvas, offset


def shift_segmentation(segmentation: Any, offset: int, height: int) -> Any:
    """Kraken's segmentation of the canvas moved back into the original image's pixels."""
    if not offset:
        return segmentation

    def points(values):
        return values and [[x, min(max(y - offset, 0), height)] for x, y in values]

    def move(item, *fields):
        return dataclasses.replace(item, **{f: points(getattr(item, f)) for f in fields})

    lines = [move(line, "baseline", "boundary") for line in segmentation.lines]
    regions = {
        kind: [move(region, "boundary") for region in regions]
        for kind, regions in segmentation.regions.items()
    }
    return dataclasses.replace(segmentation, lines=lines, regions=regions)


def segmentation_tiles(height: int, tile_height: int) -> list[tuple[int, int, int, int]]:
    """Full-width bands ``(top, bottom, start, end)`` covering a page ``height`` pixels high.

    Bands are ``tile_height`` pixels high, evenly spaced and overlapping by at least a quarter of
    that; each owns the rows from ``start`` to ``end``, split in the middle of every overlap.
    ``0`` or a page no higher than ``tile_height`` gives the whole page as one band.
    """
    if not tile_height or height <= tile_height:
        return [(0, height, 0, height)]
    gaps = math.ceil((height - tile_height) / (tile_height - tile_height // 4))
    tops = [round(index * (height - tile_height) / gaps) for index in range(gaps + 1)]
    cuts = [0] + [(top + previous + tile_height) // 2 for previous, top in itertools.pairwise(tops)]
    cuts.append(height)
    return [(top, top + tile_height, cuts[i], cuts[i + 1]) for i, top in enumerate(tops)]


def baseline_middle(points: list) -> float:
    """The y coordinate halfway along a baseline, where Kraken assigns a line to a region."""
    remaining = sum(math.dist(p, q) for p, q in itertools.pairwise(points)) / 2
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        length = math.dist((x0, y0), (x1, y1))
        if length >= remaining and length:
            return y0 + (y1 - y0) * remaining / length
        remaining -= length
    return float(points[-1][1])


def clip_rows(points: list, start: int, end: int) -> list[list[int]]:
    """A polygon cut to the rows from ``start`` to ``end`` (Sutherland-Hodgman clipping)."""

    def clip(points, inside, y):
        out = []
        for previous, point in zip(points[-1:] + points[:-1], points, strict=True):
            if inside(point) != inside(previous):
                (x0, y0), (x1, y1) = previous, point
                out.append([round(x0 + (x1 - x0) * (y - y0) / (y1 - y0)), y])
            if inside(point):
                out.append(list(point))
        return out

    points = clip(list(points), lambda p: p[1] >= start, start)
    return clip(points, lambda p: p[1] <= end, end) if points else []


def segment(segmenter: Any, config: Any, image: Any, params: dict) -> tuple[Any, int, int]:
    """Kraken's segmentation of ``image`` in input-image pixels, the canvas offset and tile count.

    A page higher than ``segmentation_tile_height`` is segmented in overlapping full-width bands
    (``segmentation_tiles``), each scaled to the model's input height on its own (on a canvas
    like a short image), so small text is seen at a higher resolution. A band keeps the lines
    whose baseline middle lies in its own rows and its regions cut to those rows; the lines of
    all bands are then put into Kraken's polygonal reading order over the whole page. Reading
    orders from a reading order model of a custom segmentation model are not used, as they
    cover only one band.
    """
    min_height = params["segmentation_min_height"]
    tiles = segmentation_tiles(image.height, params["segmentation_tile_height"])
    if len(tiles) == 1:
        canvas, offset = segmentation_canvas(image, min_height)
        segmentation = segmenter.predict(canvas, config)
        return shift_segmentation(segmentation, offset, image.height), offset, 1
    from kraken.lib.segmentation import polygonal_reading_order

    lines, regions, first = [], {}, None
    for top, bottom, start, end in tiles:
        canvas, offset = segmentation_canvas(image.crop((0, top, image.width, bottom)), min_height)
        band = segmenter.predict(canvas, config)
        band = shift_segmentation(band, offset - top, image.height)
        first = first or band
        lines += [line for line in band.lines if start <= baseline_middle(line.baseline) < end]
        for kind, found in band.regions.items():
            for region in found:
                boundary = clip_rows(region.boundary, start, end)
                if len(boundary) >= 3:
                    region = dataclasses.replace(region, boundary=boundary)
                    regions.setdefault(kind, []).append(region)
    flat = [region for found in regions.values() for region in found]
    order = polygonal_reading_order(lines, first.text_direction[-2:], flat) if lines else []
    segmentation = dataclasses.replace(
        first, lines=[lines[i] for i in order], regions=regions, line_orders=[]
    )
    return segmentation, 0, len(tiles)


def native_result(segmentation: Any, records: list[Any]) -> dict[str, Any]:
    """``kraken.json``: Kraken's segmentation with each line's text, and per-character confidences.

    ``segmentation`` loads back with ``kraken.containers.Segmentation(**...)``.
    """
    lines = [dataclasses.asdict(record) | {"text": record.prediction} for record in records]
    data = dataclasses.asdict(dataclasses.replace(segmentation, lines=[]))
    data["lines"] = lines
    return {"segmentation": data, "confidences": [list(r.confidences) for r in records]}


def region_type(kraken_type: str) -> str:
    return REGION_TYPES.get(kraken_type.split(":", 1)[0].lower(), "other")


def _polygon(points: list | None, minimum: int) -> list[list[float]] | None:
    if points and len(points) >= minimum:
        return [[float(x), float(y)] for x, y in points]
    return None


def result_to_prediction(native: dict[str, Any], sample_id: str, size: tuple[int, int]) -> dict:
    """Standardized prediction (docs/PREDICTIONS.md) from ``kraken.json``.

    Lines with non-blank text keep their boundary polygon, baseline, mean character confidence
    and non-default line type. They are read in Kraken's reading order (the last of its
    ``line_orders`` when a reading order model added one), and each run of consecutive lines in
    the same Kraken region becomes one region, so the page text keeps that order exactly. A
    Kraken region whose lines are not consecutive therefore gives several regions with the same
    ``extra.kraken_id``; consecutive lines outside any region form a region without a polygon.
    Regions without recognized text (for example image regions) stay only in the native outputs.
    """
    segmentation, confidences = native["segmentation"], native["confidences"]
    lines = segmentation["lines"]
    if len(lines) != len(confidences):
        raise AdapterError("lines and confidences differ in length")
    kraken_regions = {
        region["id"]: (kind, region)
        for kind, regions in (segmentation.get("regions") or {}).items()
        for region in regions
    }
    orders = segmentation.get("line_orders") or []
    order = orders[-1] if orders else range(len(lines))
    regions: list[dict[str, Any]] = []
    previous = None
    for index in order:
        line, chars = lines[index], confidences[index]
        text = line.get("text") or ""
        if not text.strip():
            continue
        parent = next((r for r in line.get("regions") or [] if r in kraken_regions), None)
        if regions and parent == previous:
            region = regions[-1]
        elif parent is None:
            region = {"id": "", "type": "text", "lines": [], "extra": {"grouping": "unassigned"}}
            regions.append(region)
        else:
            kind, native_region = kraken_regions[parent]
            region = {"id": "", "type": region_type(kind), "native_type": kind}
            if polygon := _polygon(native_region.get("boundary"), 3):
                region["polygon"] = polygon
            region |= {"lines": [], "extra": {"kraken_id": parent}}
            regions.append(region)
        previous = parent
        entry: dict[str, Any] = {"id": "", "text": text}
        if polygon := _polygon(line.get("boundary"), 3):
            entry["polygon"] = polygon
        if baseline := _polygon(line.get("baseline"), 2):
            entry["baseline"] = baseline
        if chars:
            entry["confidence"] = round(sum(chars) / len(chars), 4)
        kind = ((line.get("tags") or {}).get("type") or [{}])[0].get("type")
        if kind and kind != "default":
            entry["type"] = kind
        entry["extra"] = {"kraken_id": line["id"]}
        region["lines"].append(entry)
    count = 0
    for number, region in enumerate(regions, 1):
        region["id"] = f"r{number}"
        for entry in region["lines"]:
            count += 1
            entry["id"] = f"l{count}"
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": size[0], "height": size[1]},
        "text": "\n\n".join("\n".join(e["text"] for e in r["lines"]) for r in regions),
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    return prediction


def page_xml(segmentation: Any, records: list[Any], name: str, size: tuple[int, int]) -> str:
    """Kraken's own PAGE XML serialization of the recognition result."""
    from kraken import serialization

    results = dataclasses.replace(segmentation, lines=records, imagename=name)
    return serialization.serialize(results, image_size=size, template="pagexml")


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Segment and recognize one sample, write its outputs and return ``result.json`` content."""
    segmenter, recognizer, segmentation_config, recognition_config = models
    image = read_image(source / sample["image"])
    segmentation, offset, tiles = segment(segmenter, segmentation_config, image, params)
    records = list(recognizer.predict(image, segmentation, recognition_config))
    if len(records) != len(segmentation.lines):
        raise AdapterError(f"Kraken recognized {len(records)} of {len(segmentation.lines)} lines")
    xml = page_xml(segmentation, records, Path(sample["image"]).name, image.size)
    (out / "kraken.xml").write_text(xml, encoding="utf-8")
    native = native_result(segmentation, records)
    (out / "kraken.json").write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    prediction = result_to_prediction(native, sample["id"], image.size)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    regions = prediction.get("regions", [])
    counts = {
        "canvas_offset": offset,
        "segmentation_tiles": tiles,
        "kraken_lines": len(records),
        "kraken_regions": sum(len(r) for r in segmentation.regions.values()),
        "lines": sum(len(region["lines"]) for region in regions),
        "regions": len(regions),
    }
    return {"status": "ok", "primary": "prediction.json", "extra": counts}


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    models, params, info, files, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        segmenter, recognizer, files = load_models(params)
        models = (segmenter, recognizer, *inference_configs(params))
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load Kraken: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {"files": files, "job_model": job.get("model")}
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(models, params, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=model_info, params=params)
        write_result(out, result)
    print(f"kraken adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
