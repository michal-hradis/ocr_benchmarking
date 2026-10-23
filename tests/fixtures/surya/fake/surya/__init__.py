"""Fake Surya 0.17.1: predictors that return committed results chosen by the image width.

Every predictor loads only from a checkpoint directory with a config.json, and records the
device; the recognizer checks it is given the detected line polygons.
"""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

FIXTURES = Path(__file__).resolve().parents[2]
RESULTS = {900: "page", 1000: "columns", 1500: "newspaper", 300: "blank"}


def native(image):
    """The committed surya.json for this image width, or an empty result of the image's size."""
    width, height = image.size
    if width == 666:
        raise RuntimeError("CUDA out of memory (fake)")
    if width in RESULTS:
        return json.loads((FIXTURES / f"{RESULTS[width]}.json").read_text(encoding="utf-8"))
    bbox = [0, 0, width, height]
    return {
        "detection": {"bboxes": [], "image_bbox": bbox},
        "recognition": {"text_lines": [], "image_bbox": bbox},
        "layout": {"bboxes": [], "image_bbox": bbox, "sliced": False},
    }


class Result:
    def __init__(self, data):
        self.data = data
        self.bboxes = [SimpleNamespace(polygon=b["polygon"]) for b in data.get("bboxes", [])]

    def model_dump(self, mode="python", exclude=None):
        assert mode == "json"
        return {k: v for k, v in copy.deepcopy(self.data).items() if k not in (exclude or ())}


class Predictor:
    def __init__(self, checkpoint=None, device=None):
        if not (Path(checkpoint) / "config.json").is_file():
            raise OSError(f"no checkpoint at {checkpoint}")
        assert os.environ["TORCH_DEVICE"] == device.split(":")[0]
        self.checkpoint, self.device = checkpoint, device
