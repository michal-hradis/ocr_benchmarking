"""PyLaia HTR engine adapter: Kraken page segmentation, PyLaia line recognition (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load Kraken's
default baseline segmentation model and start the PyLaia recognizer process (pylaia_worker.py,
in the image's separate Python 3.10 venv) once, then for every sample image in job order segment
the page exactly as the Kraken engine does, crop every line's bounding box from the original
image, recognize the crops with PyLaia and write $OCRBENCH_OUTPUT/<id>/ with Kraken's
segmentation (segmentation.json), PyLaia's per-line output (pylaia.json), the standardized
prediction.json and, last, result.json. The Kraken engine's adapter (kraken_engine.py) provides
the segmentation canvas, the image decoding and the mapping of lines to regions.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path
from typing import Any

import kraken_engine as kr

HERE = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("OCRBENCH_PYLAIA_MODELS", "/opt/ocrbench/models"))
PYLAIA_PYTHON = os.environ.get("OCRBENCH_PYLAIA_PYTHON", "/opt/pylaia/bin/python")
WORKER = os.environ.get("OCRBENCH_PYLAIA_WORKER", str(HERE / "pylaia_worker.py"))
# Bundled recognition models (README.md) and the fixed line height they were trained with.
BUNDLED_MODELS = {"newseye-austrian": 128}
MODEL_FILES = ("model", "weights.ckpt", "syms.txt")
# Engine parameters (job.json "params"), README.md.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "recognition_model": "newseye-austrian",
    "batch_size": 8,
    "segmentation_min_height": 1200,
}
INT_PARAMS = {"batch_size": 1, "segmentation_min_height": 0}  # name: minimum


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
    if merged["recognition_model"] not in BUNDLED_MODELS:
        raise AdapterError(f"recognition_model must be one of {', '.join(BUNDLED_MODELS)}")
    return merged


def model_files(params: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The segmentation and recognition model files with their checksums."""
    paths = {"segmentation": Path(resources.files("kraken").joinpath("blla.mlmodel"))}
    directory = MODELS_DIR / params["recognition_model"]
    paths |= {f"recognition/{name}": directory / name for name in MODEL_FILES}
    for role, path in paths.items():
        if not path.is_file():
            raise AdapterError(f"{role} model file {path} does not exist (models: README.md)")
    return {role: {"path": str(path), "sha256": kr.sha256(path)} for role, path in paths.items()}


class Recognizer:
    """The PyLaia recognizer process (pylaia_worker.py), restarted after it exits."""

    def __init__(self, params: dict[str, Any]) -> None:
        name = params["recognition_model"]
        self.args = json.dumps(
            {
                "model_dir": str(MODELS_DIR / name),
                "device": params["device"],
                "batch_size": params["batch_size"],
                "fixed_height": BUNDLED_MODELS[name],
            }
        )
        self.process: subprocess.Popen | None = None
        self.info: dict[str, Any] = {}

    def start(self) -> None:
        self.process = subprocess.Popen(
            [PYLAIA_PYTHON, WORKER, self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        reply = self._reply()
        if not reply.get("ready"):
            self.close()
            raise AdapterError(reply.get("error") or "PyLaia did not start")
        self.info = {key: value for key, value in reply.items() if key != "ready"}

    def recognize(self, crops: list[Any]) -> list[dict[str, Any]]:
        """PyLaia's symbols, text and per-character confidences for grayscale line images."""
        if self.process is None or self.process.poll() is not None:
            self.start()
        assert self.process
        assert self.process.stdin
        header = {"lines": [list(crop.size) for crop in crops]}
        try:
            self.process.stdin.write(json.dumps(header).encode() + b"\n")
            for crop in crops:
                self.process.stdin.write(crop.tobytes())
            self.process.stdin.flush()
        except BrokenPipeError:
            pass  # the reply reports the exit
        reply = self._reply()
        if "error" in reply:
            raise AdapterError(f"PyLaia: {reply['error']}")
        lines = reply["lines"]
        if len(lines) != len(crops):
            raise AdapterError(f"PyLaia recognized {len(lines)} of {len(crops)} lines")
        return lines

    def _reply(self) -> dict[str, Any]:
        assert self.process
        assert self.process.stdout
        line = self.process.stdout.readline()
        if not line:
            code = self.process.wait()
            self.process = None
            raise AdapterError(f"the PyLaia process exited with code {code}")
        return json.loads(line)

    def close(self) -> None:
        if self.process is not None:
            self.process.stdin.close()
            self.process.wait()
            self.process = None


def crop_box(boundary: list | None, size: tuple[int, int]) -> list[int] | None:
    """The integer bounding box ``[x0, y0, x1, y1]`` of a line polygon, clipped to the image;
    ``None`` without a polygon or for an empty box."""
    if not boundary or len(boundary) < 3:
        return None
    xs = [x for x, _ in boundary]
    ys = [y for _, y in boundary]
    x0, y0 = max(math.floor(min(xs)), 0), max(math.floor(min(ys)), 0)
    x1, y1 = min(math.ceil(max(xs)), size[0]), min(math.ceil(max(ys)), size[1])
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def recognize_lines(recognizer: Any, image: Any, lines: list[dict]) -> list[dict[str, Any]]:
    """``pylaia.json`` lines: each segmented line's crop box and PyLaia's result, in Kraken's
    line order. A line without a usable polygon gets ``crop: null`` and no text."""
    gray = image.convert("L")
    boxes = [crop_box(line.get("boundary"), image.size) for line in lines]
    crops = [gray.crop(tuple(box)) for box in boxes if box]
    results = iter(recognizer.recognize(crops) if crops else [])
    records = []
    for line, box in zip(lines, boxes, strict=True):
        record: dict[str, Any] = {"kraken_id": line["id"], "crop": box}
        if box:
            result = next(results)
            record |= {key: result[key] for key in ("symbols", "text", "confidences")}
        else:
            record |= {"symbols": [], "text": "", "confidences": []}
        records.append(record)
    return records


def result_to_prediction(
    segmentation: dict[str, Any], records: list[dict], sample_id: str, size: tuple[int, int]
) -> dict[str, Any]:
    """Standardized prediction from ``segmentation.json`` and ``pylaia.json``: Kraken's mapping
    (kraken_engine.result_to_prediction) with PyLaia's text and character confidences."""
    lines = segmentation["lines"]
    if len(lines) != len(records):
        raise AdapterError("segmentation and PyLaia lines differ in length")
    texts = [line | {"text": record["text"]} for line, record in zip(lines, records, strict=True)]
    native = {
        "segmentation": segmentation | {"lines": texts},
        "confidences": [record["confidences"] for record in records],
    }
    return kr.result_to_prediction(native, sample_id, size)


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Segment and recognize one sample, write its outputs and return ``result.json`` content."""
    segmenter, config, recognizer = models
    image = kr.read_image(source / sample["image"])
    canvas, offset = kr.segmentation_canvas(image, params["segmentation_min_height"])
    segmentation = segmenter.predict(canvas, config)
    segmentation = kr.shift_segmentation(segmentation, offset, image.height)
    native = dataclasses.asdict(segmentation)
    (out / "segmentation.json").write_text(json.dumps(native, ensure_ascii=False), "utf-8")
    records = recognize_lines(recognizer, image, native["lines"])
    pylaia = {"lines": records}
    (out / "pylaia.json").write_text(json.dumps(pylaia, ensure_ascii=False), encoding="utf-8")
    prediction = result_to_prediction(native, records, sample["id"], image.size)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    regions = prediction.get("regions", [])
    counts = {
        "canvas_offset": offset,
        "kraken_lines": len(records),
        "kraken_regions": sum(len(r) for r in segmentation.regions.values()),
        "uncropped_lines": sum(record["crop"] is None for record in records),
        "lines": sum(len(region["lines"]) for region in regions),
        "regions": len(regions),
    }
    return {"status": "ok", "primary": "prediction.json", "extra": counts}


def load(params: dict[str, Any]) -> tuple[tuple, dict[str, Any]]:
    """Kraken's segmentation model, its configuration and the started PyLaia recognizer."""
    from kraken.tasks import SegmentationTaskModel

    segmenter = SegmentationTaskModel.load_model(Path(model_files(params)["segmentation"]["path"]))
    config, _ = kr.inference_configs(params)
    recognizer = Recognizer(params)
    recognizer.start()
    return (segmenter, config, recognizer), recognizer.info


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    models, params, info, files, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = kr.environment(params["device"])
        files = model_files(params)
        models, worker_info = load(params)
        info |= worker_info
    except (AdapterError, kr.AdapterError) as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load the models: {type(error).__name__}: {error}"
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
        except (AdapterError, kr.AdapterError) as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=model_info, params=params)
        kr.write_result(out, result)
    if models:
        models[2].close()
    print(f"pylaia adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
