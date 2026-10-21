"""Fake task models returning the committed result (../../page.json) for 900x420 images.

A model file's content selects its kind: ``segmentation``, ``recognition`` or ``bbox`` (a
recognition model trained on bounding boxes); anything else is not a model. The segmenter
returns the committed segmentation for a 900-pixel-wide canvas, moved down to where the
adapter centred the 420-pixel-high image, an empty one for other widths, and raises for a
666-pixel-wide image.
"""

import json
from dataclasses import replace
from pathlib import Path

from kraken.configs import RecognitionInferenceConfig, SegmentationInferenceConfig
from kraken.containers import BaselineLine, BaselineOCRRecord, Region, Segmentation

PAGE = json.loads((Path(__file__).resolve().parents[2] / "page.json").read_text(encoding="utf-8"))


def _kind(path):
    return Path(path).read_text(encoding="utf-8").strip()


def _shift(points, offset):
    return [[x, y + offset] for x, y in points]


class SegmentationTaskModel:
    @classmethod
    def load_model(cls, path):
        if _kind(path) != "segmentation":
            raise ValueError("No segmentation models in model list [].")
        return cls()

    def predict(self, im, config):
        assert isinstance(config, SegmentationInferenceConfig)
        width, height = im.size
        if width == 666:
            raise RuntimeError("fake segmentation error")
        native = PAGE["segmentation"]
        if width != 900:
            return Segmentation("baselines", None, "horizontal-lr", False, [], {}, [])
        offset = (height - 420) // 2
        lines = [BaselineLine(**line | {"text": None}) for line in native["lines"]]
        lines = [
            replace(
                line, baseline=_shift(line.baseline, offset), boundary=_shift(line.boundary, offset)
            )
            for line in lines
        ]
        regions = {
            kind: [Region(**r | {"boundary": _shift(r["boundary"], offset)}) for r in regs]
            for kind, regs in native["regions"].items()
        }
        return Segmentation("baselines", None, "horizontal-lr", False, lines, regions, [])


class RecognitionTaskModel:
    def __init__(self, seg_type):
        self.seg_type = seg_type

    @classmethod
    def load_model(cls, path):
        kind = _kind(path)
        if kind not in ("recognition", "bbox"):
            raise ValueError("No recognition model in model list [].")
        return cls("bbox" if kind == "bbox" else "baselines")

    def predict(self, im, segmentation, config):
        assert isinstance(config, RecognitionInferenceConfig)
        assert im.size[1] == 420 or not segmentation.lines  # the original image, not the canvas
        native = {line["id"]: line for line in PAGE["segmentation"]["lines"]}
        confidences = dict(zip(native, PAGE["confidences"], strict=True))
        for line in segmentation.lines:
            text = native[line.id]["text"]
            yield BaselineOCRRecord(text, confidences[line.id], dict(vars(line)))
