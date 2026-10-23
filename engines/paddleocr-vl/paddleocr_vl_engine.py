"""PaddleOCR-VL-1.6 document parsing engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the
official PaddleOCR-VL v1.6 pipeline once (PP-DocLayoutV3 layout detection and reading order, then
the PaddleOCR-VL-1.6-0.9B model on every layout block, run natively by PaddlePaddle) from the
/cache volume, checked against models.sha256, run it on every sample image in job order and
write $OCRBENCH_OUTPUT/<id>/ with the native result (paddleocr_vl.json), the standardized
prediction.json and, last, result.json. Every layout block becomes a region in the pipeline's
order with its box or polygon from the layout model and the text the VL model generated for it;
there are no line positions. A sample is truncated when the VL model stops at max_new_tokens on
any block.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# The models are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_PADDLEOCR_VL_CACHE", "/cache"))
MODELS_DIR = "models/paddleocr-vl"  # relative to the cache; models.sha256 is relative to it
CHECKSUMS = Path(os.environ.get("OCRBENCH_PADDLEOCR_VL_CHECKSUMS", "/opt/ocrbench/models.sha256"))
PIPELINE_VERSION = "v1.6"
# PaddleOCRVL() argument prefix: (PaddleX model name, Hugging Face repository, commit).
MODELS = {
    "layout_detection": (
        "PP-DocLayoutV3",
        "PaddlePaddle/PP-DocLayoutV3",
        "241f8bdfc77a7c7bee915a5057aaee58c235a8d3",
    ),
    "vl_rec": (
        "PaddleOCR-VL-1.6-0.9B",
        "PaddlePaddle/PaddleOCR-VL-1.6",
        "c5630abae1d940eafe0697512a0325494b02ab42",
    ),
}
# Engine parameters (job.json "params"), README.md. None keeps the pipeline default; 4096 is
# the pipeline's own default for max_new_tokens, made explicit to detect truncation.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "gpu:0",
    "max_new_tokens": 4096,
    "min_pixels": None,
    "max_pixels": None,
    "layout_threshold": None,
}
INT_PARAMS = {"max_new_tokens": 1, "min_pixels": 1, "max_pixels": 1}  # name: minimum
# PP-DocLayoutV3 block label: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "text": "text",
    "abstract": "text",
    "content": "text",
    "reference": "text",
    "reference_content": "text",
    "algorithm": "text",
    "aside_text": "text",
    "vertical_text": "text",
    "doc_title": "heading",
    "paragraph_title": "heading",
    "figure_title": "caption",
    "footnote": "footnote",
    "vision_footnote": "footnote",
    "header": "page-header",
    "footer": "page-footer",
    "number": "page-number",
    "table": "table",
    "display_formula": "formula",
    "inline_formula": "formula",
    "image": "figure",
    "chart": "figure",
    "seal": "figure",
    "header_image": "figure",
    "footer_image": "figure",
}
FORMULA_LABELS = {"display_formula", "inline_formula", "formula_number"}
PREPROCESSING = ("use_doc_orientation_classify", "use_doc_unwarping")  # both transform


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    if merged["max_new_tokens"] is None:
        merged["max_new_tokens"] = DEFAULT_PARAMS["max_new_tokens"]
    device = merged["device"]
    if not isinstance(device, str) or not re.fullmatch(r"gpu:[0-9]+", device):
        raise AdapterError(f"device must be 'gpu:<n>' (the image is a CUDA build): {device!r}")
    for key, low in INT_PARAMS.items():
        value = merged[key]
        if value is not None and (type(value) is not int or value < low):
            raise AdapterError(f"{key} must be an integer of at least {low}: {value!r}")
    low, high = merged["min_pixels"], merged["max_pixels"]
    if low is not None and high is not None and low > high:
        raise AdapterError("min_pixels must not be larger than max_pixels")
    threshold = merged["layout_threshold"]
    if threshold is not None and (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise AdapterError(f"layout_threshold must be a number with 0 < value <= 1: {threshold!r}")
    return merged


def model_dir(prefix: str) -> str:
    """Directory of a pinned model, relative to the cache."""
    _, repository, revision = MODELS[prefix]
    return f"{MODELS_DIR}/{repository.split('/', 1)[1]}/{revision}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_models() -> dict[str, str]:
    """SHA-256 of the files of both pinned model directories, which must hold exactly the files
    of models.sha256, all matching: PaddleX may load other files it finds there."""
    pinned = dict(
        reversed(line.split()) for line in CHECKSUMS.read_text().splitlines() if line.strip()
    )
    root = CACHE / MODELS_DIR
    files: dict[str, str] = {}
    for prefix in MODELS:
        directory = CACHE / model_dir(prefix)
        if not directory.is_dir():
            raise AdapterError(
                f"model directory {directory} does not exist; download the models with "
                "make paddleocr-vl-model and run with --cache (README.md)"
            )
        files |= {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted(directory.rglob("*"))
            if p.is_file()
        }
    bad = sorted(name for name, digest in pinned.items() if files.get(name) != digest)
    if bad:
        raise AdapterError(f"model files missing or not matching models.sha256: {bad}")
    extra = sorted(set(files) - set(pinned))
    if extra:
        raise AdapterError(f"model files not in models.sha256: {extra}")
    return files


def pipeline_arguments(params: dict[str, Any]) -> dict[str, Any]:
    """PaddleOCRVL() keyword arguments: pipeline v1.6 with the pinned models, no preprocessing.

    Document orientation correction and unwarping are off because they change the image the
    positions refer to (they are off in the v1.6 defaults too). Everything else keeps the
    pipeline defaults, including the native Paddle backend for the VL model.
    """
    arguments: dict[str, Any] = {"pipeline_version": PIPELINE_VERSION, "device": params["device"]}
    for prefix, (name, _, _) in MODELS.items():
        arguments[f"{prefix}_model_name"] = name
        arguments[f"{prefix}_model_dir"] = str(CACHE / model_dir(prefix))
    return arguments | dict.fromkeys(PREPROCESSING, False)


def predict_arguments(params: dict[str, Any]) -> dict[str, Any]:
    """PaddleOCRVL.predict() keyword arguments; None keeps the pipeline default."""
    keys = ("max_new_tokens", "min_pixels", "max_pixels", "layout_threshold")
    return {key: params[key] for key in keys if params[key] is not None}


class Generations:
    """Records every generation of the VL model: generated tokens and whether it ended.

    The pipeline runs the native model one block at a time and returns only the decoded text,
    so the adapter wraps the model's ``generate`` to see whether a block stopped at
    ``max_new_tokens`` without its end token (the output is then cut off).
    """

    def __init__(self, model: Any, eos_token_id: int) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.eos = eos_token_id
        generate = model.generate

        def recording(inputs: Any, **kwargs: Any) -> Any:
            output = generate(inputs, **kwargs)
            for ids in output[0].tolist():  # generated tokens only, padded after the end token
                ended = self.eos in ids
                self.calls.append((ids.index(self.eos) if ended else len(ids), ended))
            return output

        model.generate = recording

    def summary(self, max_new_tokens: int) -> dict[str, Any]:
        tokens = [count for count, _ in self.calls]
        return {
            "vl_calls": len(self.calls),
            "generated_tokens": sum(tokens),
            "max_generated_tokens": max(tokens, default=0),
            "truncated_blocks": sum(1 for _, ended in self.calls if not ended),
            "max_new_tokens": max_new_tokens,
        }


def instrument(pipeline: Any) -> Generations:
    """Hook the VL model of a PaddleOCRVL pipeline (PaddleX's local DocVLM predictor)."""
    predictor = pipeline.paddlex_pipeline.vl_rec_model
    return Generations(predictor.infer, predictor.processor.tokenizer.eos_token_id)


class _TableText(HTMLParser):
    """Rows of an HTML table as lines, cells separated by tabs, as the host converter does."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell is not None:
            self.rows[-1].append(" ".join("".join(self.cell).split()))
            self.cell = None

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)


def table_text(content: str) -> str:
    """Plain text of a table block: the pipeline converts the model's table to HTML; content
    that is not an HTML table is kept."""
    if "<table" not in content.lower():
        return content.strip()
    parser = _TableText()
    parser.feed(content)
    parser.close()
    return "\n".join("\t".join(row) for row in parser.rows if any(row))


def formula_text(content: str) -> str:
    """TeX source of a formula block, without the pipeline's ``$$`` or ``$`` delimiters."""
    text = content.strip()
    for mark in ("$$", "$"):
        if len(text) > 2 * len(mark) and text.startswith(mark) and text.endswith(mark):
            return text[len(mark) : -len(mark)].strip()
    return text


def block_text(label: str, content: str) -> str:
    if label == "table":
        return table_text(content)
    if label in FORMULA_LABELS:
        return formula_text(content)
    return content.strip()


def polygon(block: dict[str, Any]) -> list[list[float]]:
    """The block's polygon from the layout model when it has one, otherwise its box."""
    points = block.get("block_polygon_points")
    if points is not None and len(points) >= 3:
        return [[float(x), float(y)] for x, y in points]
    x1, y1, x2, y2 = (float(v) for v in block["block_bbox"])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def result_to_prediction(
    native: dict[str, Any], sample_id: str, size: tuple[int, int]
) -> dict[str, Any]:
    """Standardized prediction (docs/PREDICTIONS.md) from one PaddleOCR-VL result.

    Every block of ``parsing_res_list`` becomes a region in that order, which is the reading
    order PP-DocLayoutV3 predicts (with adjacent text blocks merged by the pipeline), with its
    polygon and its label as native type. Its text is the VL model's output for the block, as
    the pipeline post-processed it, in plain text: an HTML table becomes one line per row with
    tab-separated cells, a formula loses its ``$$`` delimiters. Blocks without text (figures)
    stay regions without text. ``block_order`` only numbers some labels and is kept in extra.
    """
    res = native["res"]
    if res.get("model_settings", {}).get("use_doc_preprocessor"):
        raise AdapterError("document preprocessing may have changed the image")
    if (res.get("width"), res.get("height")) != size:
        raise AdapterError(f"result is for a {res.get('width')}x{res.get('height')} image")
    regions = []
    for number, block in enumerate(res["parsing_res_list"]):
        label = block["block_label"]
        region: dict[str, Any] = {
            "id": f"b{number + 1}",
            "type": REGION_TYPES.get(label, "other"),
            "native_type": label,
            "polygon": polygon(block),
        }
        text = block_text(label, block.get("block_content") or "")
        if text:
            region["text"] = text
        region["extra"] = {
            "block_id": block.get("block_id"),
            "block_order": block.get("block_order"),
            "group_id": block.get("group_id"),
        }
        regions.append(region)
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": size[0], "height": size[1]},
        "text": "\n\n".join(region["text"] for region in regions if "text" in region),
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
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


def load_pipeline(params: dict[str, Any]) -> tuple[Any, Generations]:
    from paddleocr import PaddleOCRVL

    pipeline = PaddleOCRVL(**pipeline_arguments(params))
    return pipeline, instrument(pipeline)


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
    models: tuple[Any, Generations],
    params: dict[str, Any],
    sample: dict[str, Any],
    source: Path,
    out: Path,
) -> dict[str, Any]:
    """Run the pipeline on one sample, write its outputs and return the ``result.json`` content."""
    pipeline, generations = models
    image = read_image(source / sample["image"])
    generations.calls.clear()
    results = list(pipeline.predict(image, **predict_arguments(params)))
    if len(results) != 1:
        raise AdapterError(f"PaddleOCR-VL returned {len(results)} results, expected 1")
    native = results[0].json
    (out / "paddleocr_vl.json").write_text(json.dumps(native, ensure_ascii=False), encoding="utf-8")
    height, width = image.shape[:2]
    prediction = result_to_prediction(native, sample["id"], (width, height))
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    res = native["res"]
    extra = {
        "model_settings": res.get("model_settings"),
        "blocks": len(res["parsing_res_list"]),
        "layout_boxes": len(res.get("layout_det_res", {}).get("boxes", [])),
        **generations.summary(params["max_new_tokens"]),
    }
    result: dict[str, Any] = {"status": "ok", "primary": "prediction.json", "extra": extra}
    if extra["truncated_blocks"]:
        result["status"] = "truncated"
        result["message"] = (
            f"{extra['truncated_blocks']} of {extra['vl_calls']} blocks stopped at "
            f"max_new_tokens ({params['max_new_tokens']} tokens)"
        )
    return result


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
        files = check_models()
        models = load_pipeline(params)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load PaddleOCR-VL: {error_message(error)}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        **{
            prefix: {"name": name, "repository": repository, "revision": revision}
            for prefix, (name, repository, revision) in MODELS.items()
        },
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
            result = {"status": "failed", "message": error_message(error)}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=model_info, params=params)
        write_result(out, result)
    print(f"paddleocr-vl adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
