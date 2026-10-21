"""Tesseract engine adapter; standard library only, runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, run the
``tesseract`` executable on every sample image in job order and write $OCRBENCH_OUTPUT/<id>/
with the native outputs (page.txt, page.hocr, page.tsv), the standardized prediction.json built
from the hOCR and, last, result.json. Tesseract paragraphs become regions and its text lines
become lines, with their boxes and baselines in pixels of the input image.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Engine parameters (job.json "params") and their defaults.
DEFAULT_PARAMS = {"lang": "eng", "psm": 3, "oem": 1, "dpi": None}
LANG_RE = re.compile(r"^[A-Za-z0-9_]+(\+[A-Za-z0-9_]+)*$")
LINE_CLASSES = {"ocr_line", "ocr_header", "ocr_caption", "ocr_textfloat"}
NATIVE_FORMATS = ("txt", "hocr", "tsv")  # tesseract writes page.<format> for each
STDERR_TAIL = 2000


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    if not isinstance(merged["lang"], str) or not LANG_RE.match(merged["lang"]):
        raise AdapterError(f"lang must be traineddata names joined by '+': {merged['lang']!r}")
    for key, low, high in (("psm", 0, 13), ("oem", 0, 3)):
        value = merged[key]
        if type(value) is not int or not low <= value <= high:
            raise AdapterError(f"{key} must be an integer from {low} to {high}: {value!r}")
    dpi = merged["dpi"]
    if dpi is not None and (type(dpi) is not int or not 1 <= dpi <= 2400):
        raise AdapterError(f"dpi must be an integer from 1 to 2400: {dpi!r}")
    return merged


def tesseract_command(image: Path, outbase: Path, params: dict[str, Any]) -> list[str]:
    command = ["tesseract", str(image), str(outbase), "-l", params["lang"]]
    command += ["--psm", str(params["psm"]), "--oem", str(params["oem"])]
    if params["dpi"] is not None:
        command += ["--dpi", str(params["dpi"])]
    return command + list(NATIVE_FORMATS)


def _title(element: ET.Element) -> dict[str, list[str]]:
    """hOCR ``title`` properties: ``bbox 1 2 3 4; x_wconf 96`` -> {"bbox": [...], ...}."""
    properties = {}
    for item in (element.get("title") or "").split(";"):
        if item.strip():
            name, *values = item.split()
            properties[name] = values
    return properties


def _class(element: ET.Element) -> str:
    return element.get("class") or ""


def _box(values: list[str]) -> list[list[float]]:
    x0, y0, x1, y1 = (float(v) for v in values)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _baseline(bbox: list[str], values: list[str]) -> list[list[float]]:
    """hOCR baseline ``slope offset`` is relative to the bottom-left corner of the line box."""
    x0, _, x1, y1 = (float(v) for v in bbox)
    slope, offset = float(values[0]), float(values[1])
    return [[x0, y1 + offset], [x1, y1 + offset + slope * (x1 - x0)]]


def _line(element: ET.Element) -> dict[str, Any] | None:
    words, confidences = [], []
    for word in element.iter():
        if _class(word) != "ocrx_word":
            continue
        text = "".join(word.itertext()).strip()
        if text:
            words.append(text)
            confidences += [float(c) for c in _title(word).get("x_wconf", [])]
    if not words:
        return None
    title = _title(element)
    line: dict[str, Any] = {"text": " ".join(words)}
    if element.get("id"):
        line["id"] = element.get("id")
    if _class(element) != "ocr_line":
        line["type"] = _class(element).removeprefix("ocr_")
    if "bbox" in title:
        line["polygon"] = _box(title["bbox"])
        if "baseline" in title:
            line["baseline"] = _baseline(title["bbox"], title["baseline"])
    if confidences:
        line["confidence"] = round(sum(confidences) / len(confidences), 2)
    return line


def hocr_to_prediction(hocr: str, sample_id: str) -> dict[str, Any]:
    """Standardized prediction (docs/PREDICTIONS.md) from Tesseract hOCR of one page.

    Each ``ocr_par`` is a ``text`` region with the box of the paragraph and its non-empty lines;
    the enclosing block ID is kept in ``extra``. Page text is the line text, lines joined by a
    newline and paragraphs by a blank line.
    """
    root = ET.fromstring(hocr)
    pages = [e for e in root.iter() if _class(e) == "ocr_page"]
    if len(pages) != 1:
        raise AdapterError(f"hOCR has {len(pages)} pages, expected 1")
    bbox = _title(pages[0]).get("bbox")
    if not bbox:
        raise AdapterError("hOCR page has no bbox")
    regions = []
    for block in pages[0].iter():
        if _class(block) != "ocr_carea":
            continue
        for paragraph in block.iter():
            if _class(paragraph) != "ocr_par":
                continue
            lines = [_line(e) for e in paragraph.iter() if _class(e) in LINE_CLASSES]
            lines = [line for line in lines if line]
            if not lines:
                continue
            region_id = paragraph.get("id") or f"r{len(regions) + 1}"
            region = {"id": region_id, "type": "text", "native_type": "paragraph"}
            title = _title(paragraph)
            if "bbox" in title:
                region["polygon"] = _box(title["bbox"])
            region["lines"] = lines
            region["extra"] = {"block": block.get("id"), "lang": paragraph.get("lang")}
            regions.append(region)
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": int(bbox[2]), "height": int(bbox[3])},
        "text": "\n\n".join("\n".join(line["text"] for line in r["lines"]) for r in regions),
    }
    if regions:
        prediction["regions"] = regions
    return prediction


def tesseract_version() -> str:
    try:
        done = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    except OSError as error:
        return f"unavailable: {error}"
    return (done.stdout or done.stderr).splitlines()[0] if done.returncode == 0 else "unknown"


def process(sample: dict[str, Any], source: Path, out: Path, params: dict) -> dict[str, Any]:
    """Run Tesseract on one sample, write its outputs and return the ``result.json`` content."""
    command = tesseract_command(source / sample["image"], out / "page", params)
    try:
        done = subprocess.run(command, capture_output=True, text=True, errors="replace")
    except OSError as error:
        raise AdapterError(f"cannot run tesseract: {error}") from error
    (out / "stderr.txt").write_text(done.stderr, encoding="utf-8")
    if done.returncode != 0:
        tail = done.stderr.strip()[-STDERR_TAIL:]
        raise AdapterError(f"tesseract exited with status {done.returncode}: {tail}")
    hocr = out / "page.hocr"
    if not hocr.is_file():
        raise AdapterError("tesseract wrote no hOCR output")
    try:
        prediction = hocr_to_prediction(hocr.read_text(encoding="utf-8"), sample["id"])
    except (ET.ParseError, ValueError, IndexError, UnicodeDecodeError) as error:
        raise AdapterError(f"unreadable hOCR: {error}") from error
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"status": "ok", "primary": "prediction.json", "extra": {"command": command[3:]}}


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    version = tesseract_version()
    try:
        params, params_error = parse_params(job.get("params") or {}), None
    except AdapterError as error:
        params, params_error = None, str(error)
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if params_error:
                raise AdapterError(params_error)
            result = process(sample, source, out, params)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(tesseract=version, params=params)
        write_result(out, result)
    print(f"tesseract adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
