"""Surya OCR detector, recognizer and layout/reading order adapter; runs in the engine image.

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the pinned
Surya text line detection, text recognition and layout models from the /cache volume once
(checked against models.sha256), then for every sample image in job order detect text lines,
recognize each detected line, detect the layout blocks with their reading order and write
$OCRBENCH_OUTPUT/<id>/ with Surya's native results (surya.json), the standardized
prediction.json and, last, result.json. Lines go into the layout block they lie in, and blocks
are read in Surya's layout reading order (README.md).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_SURYA_CACHE", "/cache"))
MODELS_DIR = "models/surya"  # relative to the cache, as written by fetch_model.py
CHECKSUMS = Path(os.environ.get("OCRBENCH_SURYA_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# Surya 0.17.1's default checkpoints (surya/settings.py), at models.datalab.to/<checkpoint>/.
CHECKPOINTS = {
    "detection": "text_detection/2025_05_07",
    "recognition": "text_recognition/2025_09_23",
    "layout": "layout/2025_09_23",
}
# Engine parameters (job.json "params"), README.md.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "math_mode": False,
    "recognition_batch_size": None,  # Surya's default for the device
}
# Surya layout label -> standardized region type (docs/PREDICTIONS.md#region-types); any other
# label becomes "other".
REGION_TYPES = {
    "Text": "text",
    "SectionHeader": "heading",
    "ListItem": "list-item",
    "Caption": "caption",
    "Footnote": "footnote",
    "PageHeader": "page-header",
    "PageFooter": "page-footer",
    "Table": "table",
    "Equation": "formula",
    "Picture": "figure",
    "Figure": "figure",
}
MIN_INSIDE = 0.5  # share of a line's box that must lie in a layout block to belong to it
SAME_LINE_OVERLAP = 0.8  # vertical overlap (of the smaller height) that puts boxes on one line
# Inline formatting the recognizer writes as HTML; the prediction keeps only the text.
FORMATTING_TAGS = re.compile(r"</?(?:b|i|u|s|del|mark|sup|sub|small)\b[^<>]*>", re.IGNORECASE)
LINE_BREAK_TAGS = re.compile(r"<br\s*/?>", re.IGNORECASE)


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
    if type(merged["math_mode"]) is not bool:
        raise AdapterError(f"math_mode must be true or false: {merged['math_mode']!r}")
    size = merged["recognition_batch_size"]
    if size is not None and (type(size) is not int or size < 1):
        raise AdapterError(f"recognition_batch_size must be an integer of at least 1: {size!r}")
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_checksums() -> dict[str, str]:
    """``<checkpoint>/<file>``: SHA-256, from models.sha256."""
    lines = CHECKSUMS.read_text(encoding="utf-8").splitlines()
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}


def model_files() -> dict[str, str]:
    """SHA-256 of every pinned model file; raises AdapterError unless all are present and match."""
    directory = CACHE / MODELS_DIR
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make surya-model and run with --cache (README.md)"
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
        import transformers  # noqa: F401
    except Exception as error:
        raise AdapterError(f"cannot import transformers/PyTorch: {error}") from error
    info: dict[str, Any] = {
        "surya": metadata.version("surya-ocr"),
        "transformers": metadata.version("transformers"),
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


def load_models(params: dict[str, Any]) -> tuple[Any, Any, Any]:
    """The detection, recognition and layout predictors, from the local checkpoints only."""
    device = params["device"]
    # Surya reads its settings from the environment when it is first imported; its batch size
    # defaults are keyed by the device type.
    os.environ["TORCH_DEVICE"] = device.split(":", 1)[0]
    os.environ["DISABLE_TQDM"] = "true"
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor

    paths = {role: str(CACHE / MODELS_DIR / name) for role, name in CHECKPOINTS.items()}
    detector = DetectionPredictor(checkpoint=paths["detection"], device=device)
    recognizer = RecognitionPredictor(
        FoundationPredictor(checkpoint=paths["recognition"], device=device)
    )
    layout = LayoutPredictor(FoundationPredictor(checkpoint=paths["layout"], device=device))
    return detector, recognizer, layout


def read_image(path: Path) -> Any:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode == "RGB" else image.convert("RGB")


def run_surya(models: tuple, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """``surya.json``: Surya's detection, recognition and layout results for one image.

    Recognition runs on the detected line polygons, exactly as Surya's own OCR does when given a
    detector; the detection result is kept separately for its line confidences.
    """
    from surya.common.surya.schema import TaskNames

    detector, recognizer, layout = models
    detection = detector([image])[0]
    polygons = [box.polygon for box in detection.bboxes]
    recognition = recognizer(
        [image],
        task_names=[TaskNames.ocr_with_boxes],
        polygons=[polygons],
        recognition_batch_size=params["recognition_batch_size"],
        math_mode=params["math_mode"],
    )[0]
    blocks = layout([image])[0]
    return {
        "detection": detection.model_dump(mode="json", exclude={"heatmap", "affinity_map"}),
        "recognition": recognition.model_dump(mode="json"),
        "layout": blocks.model_dump(mode="json"),
    }


def plain_text(text: str) -> str:
    """A recognized line without Surya's inline formatting tags and with entities decoded."""
    return html.unescape(FORMATTING_TAGS.sub("", LINE_BREAK_TAGS.sub(" ", text)))


def bbox(polygon: list[list[float]]) -> list[float]:
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def assign_lines(boxes: list[list[float]], blocks: list[list[float]]) -> list[int | None]:
    """Index of the layout block each line box belongs to, or None.

    A line belongs to the block that contains the largest part of its box, provided that is at
    least half of it; ties go to the block earlier in reading order.
    """
    assigned: list[int | None] = []
    for box in boxes:
        area = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
        best, best_area = None, 0.0
        for index, block in enumerate(blocks):
            width = min(box[2], block[2]) - max(box[0], block[0])
            height = min(box[3], block[3]) - max(box[1], block[1])
            if width > 0 and height > 0 and width * height > best_area:
                best, best_area = index, width * height
        assigned.append(best if area and best_area >= MIN_INSIDE * area else None)
    return assigned


def line_order(indices: list[int], boxes: list[list[float]]) -> list[int]:
    """Lines top to bottom, side by side boxes of one visual line left to right.

    Boxes sorted by their top edge join the current visual line while they overlap it
    vertically by at least 80 % of the smaller height.
    """
    rows: list[tuple[list[float], list[int]]] = []
    for index in sorted(indices, key=lambda i: (boxes[i][1], boxes[i][0])):
        box = boxes[index]
        if rows:
            row_box, members = rows[-1]
            shared = min(row_box[3], box[3]) - max(row_box[1], box[1])
            smaller = min(row_box[3] - row_box[1], box[3] - box[1])
            if smaller > 0 and shared / smaller >= SAME_LINE_OVERLAP:
                members.append(index)
                row_box[1], row_box[3] = min(row_box[1], box[1]), max(row_box[3], box[3])
                continue
        rows.append(([*box], [index]))
    return [i for _, members in rows for i in sorted(members, key=lambda i: boxes[i][0])]


def _polygon(points: list | None) -> list[list[float]] | None:
    if points and len(points) >= 3:
        return [[float(x), float(y)] for x, y in points]
    return None


def result_to_prediction(
    native: dict[str, Any], sample_id: str, size: tuple[int, int]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Standardized prediction (docs/PREDICTIONS.md) from ``surya.json``, and line counts.

    Every layout block becomes a region, in Surya's layout reading order (``position``), with
    its polygon and label. Every recognized line with non-blank text, its polygon and
    confidence, goes into the block that contains most of it (``assign_lines``), ordered as
    ``line_order``. Lines outside every block follow, top to bottom, in one region without a
    polygon (``extra.grouping: "unassigned"``), so no recognized text is lost.
    """
    lines = native["recognition"]["text_lines"]
    detected = native["detection"]["bboxes"]
    if len(lines) != len(detected):
        raise AdapterError(f"Surya recognized {len(lines)} of {len(detected)} detected lines")
    for name in ("detection", "recognition", "layout"):
        width, height = native[name]["image_bbox"][2:]
        if (round(width), round(height)) != size:
            raise AdapterError(f"{name} result is for a {width}x{height} image")
    blocks = sorted(native["layout"]["bboxes"], key=lambda block: block["position"])
    texts = [plain_text(line["text"]) for line in lines]
    kept = [i for i, text in enumerate(texts) if text.strip()]
    boxes = [bbox(line["polygon"]) for line in lines]
    owners = assign_lines([boxes[i] for i in kept], [bbox(b["polygon"]) for b in blocks])
    members: dict[int | None, list[int]] = {}
    for index, owner in zip(kept, owners, strict=True):
        members.setdefault(owner, []).append(index)

    def entry(index: int) -> dict[str, Any]:
        line = {"id": f"l{index + 1}", "text": texts[index]}
        if polygon := _polygon(lines[index]["polygon"]):
            line["polygon"] = polygon
        if lines[index].get("confidence") is not None:
            line["confidence"] = round(float(lines[index]["confidence"]), 4)
        return line

    regions = []
    for number, block in enumerate(blocks):
        label = block["label"]
        region: dict[str, Any] = {
            "id": f"r{number + 1}",
            "type": REGION_TYPES.get(label, "other"),
            "native_type": label,
        }
        if polygon := _polygon(block["polygon"]):
            region["polygon"] = polygon
        if number in members:
            region["lines"] = [entry(i) for i in line_order(members[number], boxes)]
        region["extra"] = {"position": block["position"]}
        regions.append(region)
    if None in members:
        regions.append(
            {
                "id": f"r{len(blocks) + 1}",
                "type": "text",
                "lines": [entry(i) for i in line_order(members[None], boxes)],
                "extra": {"grouping": "unassigned"},
            }
        )
    text = "\n\n".join(
        "\n".join(line["text"] for line in region["lines"])
        for region in regions
        if "lines" in region
    )
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": size[0], "height": size[1]},
        "text": text,
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    counts = {
        "detected_lines": len(detected),
        "lines": len(kept),
        "unassigned_lines": len(members.get(None, [])),
        "layout_blocks": len(blocks),
        "regions": len(regions),
    }
    return prediction, counts


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Run Surya on one sample, write its outputs and return ``result.json`` content."""
    image = read_image(source / sample["image"])
    start = time.monotonic()
    native = run_surya(models, params, image)
    seconds = round(time.monotonic() - start, 3)
    (out / "surya.json").write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    prediction, counts = result_to_prediction(native, sample["id"], image.size)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {
        "status": "ok",
        "primary": "prediction.json",
        "extra": counts | {"surya_seconds": seconds},
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
    models, params, info, files, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        files = model_files()
        models = load_models(params)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load Surya: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {
        "checkpoints": CHECKPOINTS,
        "source": "https://models.datalab.to",
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
            result = process(models, params, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=model_info, params=params)
        write_result(out, result)
    print(f"surya adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
