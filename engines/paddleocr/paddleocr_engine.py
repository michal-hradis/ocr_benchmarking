"""PaddleOCR PP-OCRv6 engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the
PaddleOCR general OCR pipeline (text detection + recognition) once, run it on every sample image
in job order and write $OCRBENCH_OUTPUT/<id>/ with the native result (paddleocr.json), the
standardized prediction.json and, last, result.json. PP-OCR reports text lines without any
layout grouping; all lines of a page go into one region in PaddleOCR's order. The detector input
size is a parameter (text_det_max_side_limit, set in the PaddleX pipeline configuration); the
detected polygons are in pixels of the input image either way.
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

MODELS_DIR = Path(os.environ.get("OCRBENCH_PADDLEOCR_MODELS", "/opt/ocrbench/models"))
DET_MODEL = "PP-OCRv6_medium_det"
REC_MODEL = "PP-OCRv6_medium_rec"
# Engine parameters (job.json "params"): None keeps the pipeline default (README.md).
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "gpu:0",
    "text_det_limit_side_len": None,
    "text_det_limit_type": None,
    "text_det_max_side_limit": None,
    "text_det_thresh": None,
    "text_det_box_thresh": None,
    "text_det_unclip_ratio": None,
    "text_rec_score_thresh": None,
    "text_recognition_batch_size": None,
}
# name: minimum; the detector input is rounded to multiples of 32 pixels.
INT_PARAMS = {
    "text_det_limit_side_len": 1,
    "text_det_max_side_limit": 32,
    "text_recognition_batch_size": 1,
}
# name: (check, description); scores are in [0, 1], detection parameters must be positive.
FLOAT_PARAMS = {
    "text_det_thresh": (lambda v: 0 < v <= 1, "a number with 0 < value <= 1"),
    "text_det_box_thresh": (lambda v: 0 < v <= 1, "a number with 0 < value <= 1"),
    "text_det_unclip_ratio": (lambda v: v > 0, "a finite number greater than 0"),
    "text_rec_score_thresh": (lambda v: 0 <= v <= 1, "a number from 0 to 1"),
}


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
    """PaddleOCR() keyword arguments: pinned models, no document preprocessing.

    Document orientation correction and unwarping are off because they change the image the
    positions refer to; the text line orientation classifier is off because its model is not
    in the image. ``text_det_max_side_limit`` is not a PaddleOCR() argument; it goes into the
    PaddleX configuration (``paddlex_config``).
    """
    arguments = {
        "text_detection_model_name": DET_MODEL,
        "text_detection_model_dir": str(MODELS_DIR / DET_MODEL),
        "text_recognition_model_name": REC_MODEL,
        "text_recognition_model_dir": str(MODELS_DIR / REC_MODEL),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    overrides = {
        k: v for k, v in params.items() if v is not None and k != "text_det_max_side_limit"
    }
    return arguments | overrides


def paddlex_config(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """The PaddleX OCR pipeline configuration with ``text_det_max_side_limit`` applied.

    The detector scales the image by ``text_det_limit_side_len``/``text_det_limit_type`` and
    then down so that its longer side is at most ``max_side_limit`` (4000 by default).
    """
    limit = params["text_det_max_side_limit"]
    if limit is not None:
        config["SubModules"]["TextDetection"]["max_side_limit"] = limit
    return config


def result_to_prediction(native: dict[str, Any], sample_id: str, size: tuple[int, int]) -> dict:
    """Standardized prediction (docs/PREDICTIONS.md) from one PaddleOCR result (``res.json``).

    Every recognized line (``rec_texts``, ``rec_polys``, ``rec_scores``) with non-blank text
    becomes a line with its detected quadrilateral and recognition score; all lines go into one
    region ``lines`` in PaddleOCR's order, without a polygon. Page text is the lines joined by
    newlines.
    """
    res = native["res"]
    if res.get("model_settings", {}).get("use_doc_preprocessor"):
        raise AdapterError("document preprocessing changed the image; positions would not match")
    texts, polys, scores = res["rec_texts"], res["rec_polys"], res["rec_scores"]
    if not len(texts) == len(polys) == len(scores):
        raise AdapterError("rec_texts, rec_polys and rec_scores differ in length")
    lines = []
    for index, (text, poly, score) in enumerate(zip(texts, polys, scores, strict=True)):
        if not text.strip():
            continue
        lines.append(
            {
                "id": f"l{index + 1}",
                "text": text,
                "polygon": [[float(x), float(y)] for x, y in poly],
                "confidence": round(float(score), 4),
            }
        )
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": size[0], "height": size[1]},
        "text": "\n".join(line["text"] for line in lines),
    }
    if lines:
        prediction["regions"] = [
            {"id": "lines", "type": "text", "lines": lines, "extra": {"grouping": "all-lines"}}
        ]
    return prediction


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
    from paddleocr import PaddleOCR
    from paddlex.inference import load_pipeline_config

    config = paddlex_config(load_pipeline_config("OCR"), params)
    return PaddleOCR(paddlex_config=config, **pipeline_arguments(params))


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


def process(
    pipeline: Any, sample: dict[str, Any], source: Path, out: Path, params: dict[str, Any]
) -> dict[str, Any]:
    """Run the pipeline on one sample, write its outputs and return the ``result.json`` content."""
    image = read_image(source / sample["image"])
    results = list(pipeline.predict(image))
    if len(results) != 1:
        raise AdapterError(f"PaddleOCR returned {len(results)} results, expected 1")
    native = results[0].json
    (out / "paddleocr.json").write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    limit = params["text_det_max_side_limit"]
    used = native["res"].get("text_det_params", {}).get("max_side_limit")
    if limit is not None and used != limit:
        raise AdapterError(f"text_det_max_side_limit {limit} requested but PaddleOCR used {used}")
    height, width = image.shape[:2]
    prediction = result_to_prediction(native, sample["id"], (width, height))
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    res = native["res"]
    settings = {key: res.get(key) for key in ("text_det_params", "text_rec_score_thresh")}
    return {"status": "ok", "primary": "prediction.json", "extra": settings}


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
        init_error = f"cannot load PaddleOCR: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    models = {"det": DET_MODEL, "rec": REC_MODEL, "job_model": job.get("model")}
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(pipeline, sample, source, out, params)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": error_message(error)}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=models, params=params)
        write_result(out, result)
    print(f"paddleocr adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
