"""PaddleOCR PP-StructureV3 engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the
PP-StructureV3 layout parsing pipeline (layout and region detection, text detection, text line
orientation and recognition) once, run it on every sample image in job order and write
$OCRBENCH_OUTPUT/<id>/ with the native result (ppstructurev3.json), the standardized
prediction.json and, last, result.json. Every layout block becomes a region in PP-StructureV3's
reading order with its box; the recognized text lines of the overall OCR result are assigned to
the blocks they overlap. Table, formula, seal and chart recognition are off. The text detection
and recognition models are one of two pinned sets (parameter ocr_models).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

MODELS_DIR = Path(os.environ.get("OCRBENCH_PPSTRUCTUREV3_MODELS", "/opt/ocrbench/models"))
# PPStructureV3() argument prefix: model name (the PP-StructureV3 defaults, see README.md).
MODELS = {
    "layout_detection": "PP-DocLayout_plus-L",
    "region_detection": "PP-DocBlockLayout",
    "textline_orientation": "PP-LCNet_x1_0_textline_ori",
}
# Parameter ocr_models: text detection and recognition models. PP-OCRv5-latin is what
# PPStructureV3(lang=<a Latin-script language>) selects; PP-OCRv6-medium is the paddleocr engine's.
OCR_MODELS = {
    "PP-OCRv5-latin": {
        "text_detection": "PP-OCRv5_server_det",
        "text_recognition": "latin_PP-OCRv5_mobile_rec",
    },
    "PP-OCRv6-medium": {
        "text_detection": "PP-OCRv6_medium_det",
        "text_recognition": "PP-OCRv6_medium_rec",
    },
}
# Engine parameters (job.json "params"): None keeps the pipeline default (README.md).
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "gpu:0",
    "ocr_models": "PP-OCRv5-latin",
    "layout_threshold": None,
    "text_det_limit_side_len": None,
    "text_det_limit_type": None,
    "text_det_thresh": None,
    "text_det_box_thresh": None,
    "text_det_unclip_ratio": None,
    "text_rec_score_thresh": None,
    "text_recognition_batch_size": None,
}
INT_PARAMS = {"text_det_limit_side_len": 1, "text_recognition_batch_size": 1}  # name: minimum
# name: (check, description); scores are in [0, 1], detection parameters must be positive.
FLOAT_PARAMS = {
    "layout_threshold": (lambda v: 0 < v <= 1, "a number with 0 < value <= 1"),
    "text_det_thresh": (lambda v: 0 < v <= 1, "a number with 0 < value <= 1"),
    "text_det_box_thresh": (lambda v: 0 < v <= 1, "a number with 0 < value <= 1"),
    "text_det_unclip_ratio": (lambda v: v > 0, "a finite number greater than 0"),
    "text_rec_score_thresh": (lambda v: 0 <= v <= 1, "a number from 0 to 1"),
}
# PP-StructureV3 block label: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "text": "text",
    "abstract": "text",
    "content": "text",
    "reference": "text",
    "reference_content": "text",
    "algorithm": "text",
    "aside_text": "text",
    "doc_title": "heading",
    "paragraph_title": "heading",
    "abstract_title": "heading",
    "reference_title": "heading",
    "content_title": "heading",
    "figure_title": "caption",
    "table_title": "caption",
    "chart_title": "caption",
    "figure_table_chart_title": "caption",
    "footnote": "footnote",
    "header": "page-header",
    "footer": "page-footer",
    "number": "page-number",
    "table": "table",
    "formula": "formula",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "seal": "figure",
    "header_image": "figure",
    "footer_image": "figure",
}
PREPROCESSING_STEPS = ("use_doc_orientation_classify", "use_doc_unwarping")  # both transform
MIN_OVERLAP = 3  # pixels on both axes, as PaddleX matches OCR boxes to blocks
SAME_LINE_OVERLAP = 0.8  # vertical overlap (of the smaller height) that puts boxes on one line


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    device = merged["device"]
    if not isinstance(device, str) or not re.fullmatch(r"gpu:[0-9]+", device):
        raise AdapterError(f"device must be 'gpu:<n>' (the image is a CUDA build): {device!r}")
    if merged["ocr_models"] not in OCR_MODELS:
        names = " or ".join(repr(name) for name in OCR_MODELS)
        raise AdapterError(f"ocr_models must be {names}: {merged['ocr_models']!r}")
    if merged["text_det_limit_type"] not in (None, "min", "max"):
        limit_type = merged["text_det_limit_type"]
        raise AdapterError(f"text_det_limit_type must be 'min' or 'max': {limit_type!r}")
    for key, low in INT_PARAMS.items():
        value = merged[key]
        if value is not None and (type(value) is not int or value < low):
            raise AdapterError(f"{key} must be an integer of at least {low}: {value!r}")
    for key, (check, description) in FLOAT_PARAMS.items():
        value = merged[key]
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not check(value):
            raise AdapterError(f"{key} must be {description}: {value!r}")
    return merged


def pipeline_arguments(params: dict[str, Any]) -> dict[str, Any]:
    """PPStructureV3() keyword arguments: pinned models, layout and OCR only.

    Document orientation correction and unwarping are off because they change the image the
    positions refer to. Table, formula, seal and chart recognition are off; their models are not
    in the image. Region detection and the text line orientation classifier stay on, as in the
    PP-StructureV3 defaults.
    """
    arguments: dict[str, Any] = {}
    for prefix, name in models(params).items():
        arguments[f"{prefix}_model_name"] = name
        arguments[f"{prefix}_model_dir"] = str(MODELS_DIR / name)
    arguments |= {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": True,
        "use_region_detection": True,
        "use_table_recognition": False,
        "use_formula_recognition": False,
        "use_seal_recognition": False,
        "use_chart_recognition": False,
    }
    overrides = {k: v for k, v in params.items() if v is not None and k != "ocr_models"}
    return arguments | overrides


def models(params: dict[str, Any]) -> dict[str, str]:
    """PPStructureV3() argument prefix: model name, with the selected OCR models."""
    return MODELS | OCR_MODELS[params["ocr_models"]]


def overlap(a: list[float], b: list[float]) -> tuple[float, float]:
    """Width and height of the intersection of two ``[x1, y1, x2, y2]`` boxes (may be < 0)."""
    return min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])


def assign_lines(boxes: list[list[float]], blocks: list[list[float]]) -> list[int | None]:
    """Index of the block each line box belongs to, or None.

    A line belongs to the blocks it overlaps by more than 3 pixels on both axes (the PaddleX
    rule for matching OCR boxes to layout blocks) and, of those, to the one with the largest
    intersection; ties go to the block earlier in reading order.
    """
    assigned: list[int | None] = []
    for box in boxes:
        best, best_area = None, 0.0
        for index, block in enumerate(blocks):
            width, height = overlap(box, block)
            if width > MIN_OVERLAP and height > MIN_OVERLAP and width * height > best_area:
                best, best_area = index, width * height
        assigned.append(best)
    return assigned


def line_order(indices: list[int], boxes: list[list[float]]) -> list[int]:
    """Lines of one block top to bottom, side by side boxes of one visual line left to right.

    As PP-StructureV3 orders the text of a horizontal block: boxes sorted by their top edge
    join the current visual line while they overlap it vertically by at least 80 % of the
    smaller height.
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


def result_to_prediction(
    native: dict[str, Any], sample_id: str, size: tuple[int, int]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Standardized prediction (docs/PREDICTIONS.md) from one PP-StructureV3 result.

    Every block of ``parsing_res_list`` becomes a region in that order with its box as polygon
    and its label as native type. PP-StructureV3 sorts every block into that list for reading
    (``xycut_enhanced``, which also places figure titles and unordered labels such as
    ``aside_text``) and writes its Markdown in it; ``block_order`` only numbers some labels on
    the layout image (null for, e.g., ``image``, ``figure_title`` or ``header``) and must agree
    with the list. Every recognized line of
    ``overall_ocr_res`` with non-blank text, its polygon and score, goes into the block it
    overlaps most (``assign_lines``), ordered as ``line_order``; lines outside every block are
    left out, as PP-StructureV3 leaves them out of its blocks. Also returns line counts.
    """
    res = native["res"]
    if res.get("model_settings", {}).get("use_doc_preprocessor"):
        # PPStructureV3() runs the preprocessor even with both of its steps off.
        steps = res.get("doc_preprocessor_res", {}).get("model_settings")
        if steps is None or any(steps.get(k, True) for k in PREPROCESSING_STEPS):
            raise AdapterError("document preprocessing may have changed the image")
    if (res.get("width"), res.get("height")) != size:
        raise AdapterError(f"result is for a {res.get('width')}x{res.get('height')} image")
    ocr = res["overall_ocr_res"]
    texts, polys, scores, boxes = (
        ocr["rec_texts"],
        ocr["rec_polys"],
        ocr["rec_scores"],
        ocr["rec_boxes"],
    )
    if not len(texts) == len(polys) == len(scores) == len(boxes):
        raise AdapterError("rec_texts, rec_polys, rec_scores and rec_boxes differ in length")
    blocks = res["parsing_res_list"]
    orders = [b["block_order"] for b in blocks if b.get("block_order") is not None]
    if orders != sorted(set(orders)):
        raise AdapterError("block_order does not increase along parsing_res_list")
    kept = [i for i, text in enumerate(texts) if text.strip()]
    owners = assign_lines([boxes[i] for i in kept], [b["block_bbox"] for b in blocks])
    members: dict[int, list[int]] = {}
    for index, owner in zip(kept, owners, strict=True):
        if owner is not None:
            members.setdefault(owner, []).append(index)
    regions = []
    for number, block in enumerate(blocks):
        label = block["block_label"]
        x1, y1, x2, y2 = (float(v) for v in block["block_bbox"])
        region: dict[str, Any] = {
            "id": f"b{number + 1}",
            "type": REGION_TYPES.get(label, "other"),
            "native_type": label,
            "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        }
        lines = [
            {
                "id": f"l{index + 1}",
                "text": texts[index],
                "polygon": [[float(x), float(y)] for x, y in polys[index]],
                "confidence": round(float(scores[index]), 4),
            }
            for index in line_order(members.get(number, []), boxes)
        ]
        if lines:
            region["lines"] = lines
        region["extra"] = {
            "block_id": block.get("block_id"),
            "block_order": block.get("block_order"),
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
        "image": {"width": size[0], "height": size[1]},
        "text": text,
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    assigned = sum(len(m) for m in members.values())
    counts = {"lines": len(kept), "assigned": assigned, "outside_blocks": len(kept) - assigned}
    return prediction, counts


def environment(device: str) -> dict[str, Any]:
    """Versions and GPU diagnostics; raises AdapterError when the GPU is unusable."""
    try:
        import paddle
        import paddleocr
        import paddlex
    except Exception as error:  # e.g. libcuda.so.1 missing without --gpus
        raise AdapterError(f"cannot import PaddlePaddle/PaddleOCR: {error}") from error
    info: dict[str, Any] = {
        "paddleocr": paddleocr.__version__,
        "paddlex": paddlex.__version__,
        "paddle": paddle.__version__,
        "cuda": paddle.version.cuda(),
        "cudnn": paddle.version.cudnn(),
        "device": device,
    }
    index = int(device.split(":", 1)[1])
    count = paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0
    if index >= count:
        raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
    info["gpu"] = paddle.device.cuda.get_device_name(index)
    return info


def load_pipeline(params: dict[str, Any]) -> Any:
    from paddleocr import PPStructureV3

    return PPStructureV3(**pipeline_arguments(params))


def error_message(error: Exception) -> str:
    """``<type>: <message>``; of Paddle's long C++ errors only the summary at the end."""
    message = str(error)
    if "Error Message Summary:" in message:
        summary = message.rsplit("Error Message Summary:", 1)[1]
        message = " ".join(summary.lstrip("-\n").split())
    return f"{type(error).__name__}: {message}"


def read_image(path: Path) -> Any:
    """BGR array as stored: EXIF orientation is not applied (docs/PREDICTIONS.md#positions)."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise AdapterError(f"cannot decode image {path.name}")
    return image


def process(pipeline: Any, sample: dict[str, Any], source: Path, out: Path) -> dict[str, Any]:
    """Run the pipeline on one sample, write its outputs and return the ``result.json`` content."""
    image = read_image(source / sample["image"])
    results = list(pipeline.predict(image))
    if len(results) != 1:
        raise AdapterError(f"PP-StructureV3 returned {len(results)} results, expected 1")
    native = results[0].json
    (out / "ppstructurev3.json").write_text(
        json.dumps(native, ensure_ascii=False), encoding="utf-8"
    )
    height, width = image.shape[:2]
    prediction, counts = result_to_prediction(native, sample["id"], (width, height))
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    res = native["res"]
    extra = {
        "model_settings": res.get("model_settings"),
        "text_det_params": res["overall_ocr_res"].get("text_det_params"),
        "blocks": len(res["parsing_res_list"]),
        "lines": counts,
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
    pipeline, params, info, init_error = None, None, {}, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        pipeline = load_pipeline(params)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load PP-StructureV3: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    used = models(params) if params else {}
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(pipeline, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": error_message(error)}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(
            environment=info, models=used | {"job_model": job.get("model")}, params=params
        )
        write_result(out, result)
    print(f"ppstructurev3 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
