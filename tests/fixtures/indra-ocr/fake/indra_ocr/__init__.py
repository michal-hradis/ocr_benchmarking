"""Fake Indra OCR: returns committed real Indra OCR 1.1.2 pages chosen by the image size."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2

FIXTURES = Path(__file__).resolve().parents[2]
PAGES = {
    (900, 420): "page.json",
    (1000, 420): "columns.json",
    (1500, 620): "newspaper.json",
    (300, 200): "blank.json",
    (420, 900): "turned.json",
    (520, 60): "line.json",
}


@dataclass
class Line:
    box: list
    text: str
    conf: float


@dataclass
class Page:
    lines: list
    timings: dict = field(default_factory=dict)
    markdown: str | None = None
    source: str = "ocr"
    note: str = ""
    blocks: list = field(default_factory=list)
    rotation: int = 0
    skew: float = 0.0
    vertical: bool = False


class Converter:
    def ocr(self, gray):
        width, height = gray.shape[1], gray.shape[0]
        if width == 666:
            raise RuntimeError("onnxruntime failed (fake)")
        name = PAGES.get((width, height))
        if name is None:
            return Page([], {"total_ms": 0.0}, markdown="", note="order=model")
        data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
        lines = [Line(**line) for line in data.pop("lines")]
        for key in ("file", "page"):
            del data[key]
        return Page(lines, **data)


def load(
    path=None,
    threads=4,
    dpi=200.0,
    mode="auto",
    image_ocr=True,
    providers=None,
    furniture=False,
    math=False,
):
    if not (Path(path) / "meta.json").is_file():
        raise FileNotFoundError(Path(path) / "meta.json")
    if log := os.environ.get("FAKE_INDRA_LOG"):
        record = {
            "path": str(path),
            "threads": threads,
            "mode": mode,
            "math": math,
            "furniture": furniture,
            "providers": providers,
            "cv2_threads": cv2.THREADS,
        }
        Path(log).write_text(json.dumps(record), encoding="utf-8")
    return Converter()
