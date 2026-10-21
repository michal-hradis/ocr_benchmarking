"""Fake PaddleOCR pipeline returning the committed result for 900-pixel-wide images."""

import json
from pathlib import Path
from types import SimpleNamespace

__version__ = "3.7.0-fake"
PAGE = Path(__file__).resolve().parent.parent / "page.json"
EMPTY = {
    "res": {
        "model_settings": {"use_doc_preprocessor": False},
        "rec_texts": [],
        "rec_scores": [],
        "rec_polys": [],
        "text_det_params": {},
    }
}


class PaddleOCR:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def predict(self, image):
        width = image.shape[1]
        if width == 666:
            raise RuntimeError("fake pipeline error")
        native = json.loads(PAGE.read_text(encoding="utf-8")) if width == 900 else EMPTY
        return [SimpleNamespace(json=native)]
