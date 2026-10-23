"""Fake doctr.models.ocr_predictor: loads only when both pinned checkpoints are in DOCTR_CACHE_DIR
and returns the committed doctr.json of the image width (fake/README.md)."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from doctr.models.preprocessor import PreProcessor

FIXTURES = Path(__file__).resolve().parents[3]
RESULTS = {900: "page", 1000: "columns", 1500: "newspaper", 300: "blank"}
CHECKPOINTS = ("fast_base-688a8b34.pt", "crnn_vgg16_bn-0417f351.pt")


def export(image):
    """The committed doctr.json for this image width, or an empty page of the image's size."""
    width, height = image.size
    if width == 666:
        raise RuntimeError("CUDA out of memory (fake)")
    if width in RESULTS:
        return json.loads((FIXTURES / f"{RESULTS[width]}.json").read_text(encoding="utf-8"))
    return {
        "page_idx": 0,
        "dimensions": [height, width],
        "orientation": {"value": None, "confidence": None},
        "language": {"value": None, "confidence": None},
        "blocks": [],
        "layout": [],
        "tables": [],
    }


class OCRPredictor:
    device = None

    def __init__(self):
        cfg = {"input_shape": (3, 1024, 1024), "mean": (0.8, 0.8, 0.8), "std": (0.3, 0.3, 0.3)}
        self.det_predictor = SimpleNamespace(
            model=SimpleNamespace(cfg=cfg),
            pre_processor=PreProcessor((1024, 1024), 2, cfg["mean"], cfg["std"], True, True),
        )

    def to(self, device):
        self.device = device
        return self

    def __call__(self, pages):
        assert self.device is not None
        return SimpleNamespace(
            pages=[SimpleNamespace(export=lambda page=page: export(page)) for page in pages]
        )


def ocr_predictor(det_arch, reco_arch, pretrained=False):
    assert (det_arch, reco_arch, pretrained) == ("fast_base", "crnn_vgg16_bn", True)
    for name in CHECKPOINTS:
        path = Path(os.environ["DOCTR_CACHE_DIR"]) / "models" / name
        if not path.is_file():
            raise OSError(f"no checkpoint at {path}")
    return OCRPredictor()
