"""Fake PaddleX pipeline configuration: the OCR detection defaults of PaddleX 3.7.2."""


def load_pipeline_config(name):
    assert name == "OCR"
    detection = {"limit_side_len": 64, "limit_type": "min", "max_side_limit": 4000}
    return {"pipeline_name": "OCR", "SubModules": {"TextDetection": detection}}
