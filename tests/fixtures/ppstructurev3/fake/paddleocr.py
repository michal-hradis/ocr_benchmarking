"""Fake PaddleOCR PPStructureV3 returning committed results, chosen by the image width.

900: page.json; 1000: columns.json; 1500: newspaper.json; 666: a pipeline error; 777: a Paddle
CUDA out-of-memory error; any other width: blank.json with that image size. Used together with
the fake paddle, paddlex and cv2 modules of tests/fixtures/paddleocr/fake.
"""

import json
from pathlib import Path
from types import SimpleNamespace

__version__ = "3.7.0-fake"
FIXTURES = Path(__file__).resolve().parent.parent
RESULTS = {900: "page.json", 1000: "columns.json", 1500: "newspaper.json"}
OOM = (
    "\n\n--------------------------------------\nC++ Traceback (most recent call last):\n"
    "--------------------------------------\n0   paddle::AnalysisPredictor::ZeroCopyRun(bool)\n"
    "\n----------------------\nError Message Summary:\n----------------------\n"
    "ResourceExhaustedError: \n\nOut of memory error on GPU 0. Cannot allocate 21.535400GB "
    "memory on GPU 0.\n (at /paddle/paddle/phi/core/memory/allocation/cuda_allocator.cc:76)\n"
)


class PPStructureV3:
    def __init__(self, **kwargs):
        for name in ("table", "formula", "seal", "chart"):
            assert kwargs[f"use_{name}_recognition"] is False
        assert Path(kwargs["layout_detection_model_dir"]).name == "PP-DocLayout_plus-L"
        self.kwargs = kwargs

    def predict(self, image):
        height, width = image.shape[:2]
        if width == 666:
            raise RuntimeError("fake pipeline error")
        if width == 777:
            raise MemoryError(OOM)
        native = json.loads((FIXTURES / RESULTS.get(width, "blank.json")).read_text("utf-8"))
        if width not in RESULTS:
            native["res"] |= {"width": width, "height": height}
        return [SimpleNamespace(json=native)]
