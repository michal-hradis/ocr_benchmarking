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
import json
import os
import re
import sys
import time
from importlib import metadata, resources
from pathlib import Path
from typing import Any

MODELS_DIR = Path(os.environ.get("OCRBENCH_KRAKEN_MODELS", "/opt/ocrbench/models"))
CACHE_DIR = "/cache/"
BUNDLED_MODELS = (
    "catmus-print-fondue-large.mlmodel",
    "McCATMuS_nfd_nofix_V1.mlmodel",
    "german_print.mlmodel",
)
# Engine parameters (job.json "params"), README.md.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "recognition_model": BUNDLED_MODELS[0],
    "segmentation_model": None,  # Kraken's default blla.mlmodel
    "batch_size": 1,
    "segmentation_min_height": 1200,
}
INT_PARAMS = {"batch_size": 1, "segmentation_min_height": 0}  # name: minimum
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
    ``line_orders`` when a reading order model added one) and grouped into their Kraken region;
    consecutive lines outside any region form a region of their own without a polygon. Regions
    are in the order of their first line, which is the reading order. Regions without recognized
    text (for example image regions) stay only in the native outputs.
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
    by_kraken_id: dict[str, dict[str, Any]] = {}
    previous = None
    for index in order:
        line, chars = lines[index], confidences[index]
        text = line.get("text") or ""
        if not text.strip():
            continue
        parent = next((r for r in line.get("regions") or [] if r in kraken_regions), None)
        if parent is None:
            if previous is not None or not regions:
                regions.append(
                    {"id": "", "type": "text", "lines": [], "extra": {"grouping": "unassigned"}}
                )
            region = regions[-1]
        elif parent in by_kraken_id:
            region = by_kraken_id[parent]
        else:
            kind, native_region = kraken_regions[parent]
            region = {"id": "", "type": region_type(kind), "native_type": kind}
            if polygon := _polygon(native_region.get("boundary"), 3):
                region["polygon"] = polygon
            region |= {"lines": [], "extra": {"kraken_id": parent}}
            regions.append(region)
            by_kraken_id[parent] = region
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
    canvas, offset = segmentation_canvas(image, params["segmentation_min_height"])
    segmentation = segmenter.predict(canvas, segmentation_config)
    segmentation = shift_segmentation(segmentation, offset, image.height)
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
