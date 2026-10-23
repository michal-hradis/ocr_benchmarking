"""Fake PaddleOCR PaddleOCRVL returning committed results, chosen by the image width.

900: page.json; 1000: columns.json; 1500: newspaper.json; 666: a pipeline error; 777: a Paddle
CUDA out-of-memory error; any other width: blank.json with that image size. For every block with
text, predict calls the VL model's generate as the real pipeline does: one token per word and the
end token (2), or, beyond max_new_tokens, that many tokens without it and the content cut to as
many words. Used together with the fake paddle, paddlex and cv2 modules of
tests/fixtures/paddleocr/fake.
"""

import json
from pathlib import Path
from types import SimpleNamespace

__version__ = "3.7.0-fake"
FIXTURES = Path(__file__).resolve().parent.parent
RESULTS = {900: "page.json", 1000: "columns.json", 1500: "newspaper.json"}
EOS = 2
OOM = (
    "\n\n--------------------------------------\nC++ Traceback (most recent call last):\n"
    "--------------------------------------\n0   paddle::AnalysisPredictor::ZeroCopyRun(bool)\n"
    "\n----------------------\nError Message Summary:\n----------------------\n"
    "ResourceExhaustedError: \n\nOut of memory error on GPU 0. Cannot allocate 21.535400GB "
    "memory on GPU 0.\n (at /paddle/paddle/phi/core/memory/allocation/cuda_allocator.cc:76)\n"
)


class Ids:
    def __init__(self, rows):
        self.rows = rows

    def tolist(self):
        return [list(row) for row in self.rows]


class Model:
    """The native VL model: generate returns (generated ids, scores) like PaddleX's."""

    def generate(self, inputs, max_new_tokens, use_cache):
        assert use_cache is True
        words = len(inputs["content"].split())
        if words > max_new_tokens:
            return Ids([[7] * max_new_tokens]), None
        return Ids([[7] * words + [EOS, 0, 0]]), None  # padded after the end token


class PaddleOCRVL:
    def __init__(self, **kwargs):
        assert kwargs["pipeline_version"] == "v1.6"
        assert kwargs["layout_detection_model_name"] == "PP-DocLayoutV3"
        assert kwargs["vl_rec_model_name"] == "PaddleOCR-VL-1.6-0.9B"
        assert Path(kwargs["vl_rec_model_dir"]).parent.name == "PaddleOCR-VL-1.6"
        assert Path(kwargs["layout_detection_model_dir"]).parent.name == "PP-DocLayoutV3"
        assert kwargs["use_doc_orientation_classify"] is False
        assert kwargs["use_doc_unwarping"] is False
        self.kwargs = kwargs
        tokenizer = SimpleNamespace(eos_token_id=EOS)
        vl_rec_model = SimpleNamespace(
            infer=Model(), processor=SimpleNamespace(tokenizer=tokenizer)
        )
        self.paddlex_pipeline = SimpleNamespace(vl_rec_model=vl_rec_model)

    def predict(self, image, max_new_tokens=4096, **kwargs):
        assert set(kwargs) <= {"min_pixels", "max_pixels", "layout_threshold"}
        height, width = image.shape[:2]
        if width == 666:
            raise RuntimeError("fake pipeline error")
        if width == 777:
            raise MemoryError(OOM)
        native = json.loads((FIXTURES / RESULTS.get(width, "blank.json")).read_text("utf-8"))
        if width not in RESULTS:
            native["res"] |= {"width": width, "height": height}
        model = self.paddlex_pipeline.vl_rec_model.infer
        for block in native["res"]["parsing_res_list"]:
            if block["block_content"]:
                inputs = {"content": block["block_content"]}
                ids = model.generate(inputs, max_new_tokens=max_new_tokens, use_cache=True)
                if EOS not in ids[0].tolist()[0]:
                    words = block["block_content"].split()[:max_new_tokens]
                    block["block_content"] = " ".join(words)
        return [SimpleNamespace(json=native)]
