"""Fake PaddleOCR pipeline returning committed results, chosen by the image width.

900: page.json; 5400: page-x6-limit<4000|8000>.json for the configured max_side_limit;
666: a pipeline error; 777: a Paddle CUDA out-of-memory error; any other width: no text.
FAKE_PADDLEOCR_IGNORE_LIMIT=1 reports max_side_limit 4000 whatever the configuration says.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

__version__ = "3.7.0-fake"
FIXTURES = Path(__file__).resolve().parent.parent
EMPTY = {
    "res": {
        "model_settings": {"use_doc_preprocessor": False},
        "rec_texts": [],
        "rec_scores": [],
        "rec_polys": [],
        "text_det_params": {},
    }
}
OOM = (
    "\n\n--------------------------------------\nC++ Traceback (most recent call last):\n"
    "--------------------------------------\n0   paddle::AnalysisPredictor::ZeroCopyRun(bool)\n"
    "\n----------------------\nError Message Summary:\n----------------------\n"
    "ResourceExhaustedError: \n\nOut of memory error on GPU 0. Cannot allocate 21.535400GB "
    "memory on GPU 0.\n (at /paddle/paddle/phi/core/memory/allocation/cuda_allocator.cc:76)\n"
)


class PaddleOCR:
    def __init__(self, paddlex_config, **kwargs):
        self.kwargs = kwargs
        self.limit = paddlex_config["SubModules"]["TextDetection"]["max_side_limit"]
        if os.environ.get("FAKE_PADDLEOCR_IGNORE_LIMIT"):
            self.limit = 4000

    def predict(self, image):
        width = image.shape[1]
        if width == 666:
            raise RuntimeError("fake pipeline error")
        if width == 777:
            raise MemoryError(OOM)
        if width == 900:
            native = json.loads((FIXTURES / "page.json").read_text(encoding="utf-8"))
        elif width == 5400:
            path = FIXTURES / f"page-x6-limit{self.limit}.json"
            native = json.loads(path.read_text(encoding="utf-8"))
        else:
            native = json.loads(json.dumps(EMPTY))
        native["res"]["text_det_params"]["max_side_limit"] = self.limit
        return [SimpleNamespace(json=native)]
