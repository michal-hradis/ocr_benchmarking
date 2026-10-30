"""Fake DocumentConverter returning the committed Docling outputs, chosen by the image width.

900: page; 520: paragraph; 1000: columns; 1500: newspaper; 1200: table; 666: a CUDA
out-of-memory error; 777: a failed conversion; 888: a partial conversion (blank page); 999: a
page whose size is the image turned by 90 degrees; 910: a blank page with one OCR cell in a
cluster that no item uses; any other width: the blank page with that
image size. The input must be a TIFF at 72 x ocr_scale DPI, and every conversion logs a warning.
The pipeline's RapidOCR reader has RapidOCR 3.10.0's settings and the pinned models; the OCR model
is asked for its crops once per page: the whole page in full_page mode, else the layout clusters.
RapidOCR preprocesses each crop of non-zero area at 3 pixels per point, without resizing it.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from docling.datamodel.base_models import ConversionStatus, InputFormat

FIXTURES = Path(__file__).resolve().parents[2]
RESULTS = {900: "page", 520: "paragraph", 1000: "columns", 1500: "newspaper", 1200: "table"}
RAPIDOCR = {
    "Global": {
        "text_score": 0.5,
        "use_det": True,
        "use_cls": True,
        "use_rec": True,
        "use_preprocess_img": True,
        "min_side_len": 30,
        "max_side_len": 2000,
        "min_height": 30,
        "width_height_ratio": 8,
    },
    "Det": {
        "limit_side_len": 736,
        "limit_type": "min",
        "thresh": 0.3,
        "box_thresh": 0.5,
        "unclip_ratio": 1.6,
        "max_candidates": 1000,
        "use_dilation": True,
        "score_mode": "fast",
        "model_path": "RapidOcr/ch_PP-OCRv5_det_mobile.onnx",
    },
    "Cls": {
        "cls_thresh": 0.9,
        "cls_batch_num": 6,
        "model_path": "RapidOcr/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    },
    "Rec": {
        "rec_img_shape": (3, 48, 320),
        "rec_batch_num": 6,
        "rec_keys_path": None,
        "model_path": "RapidOcr/latin_PP-OCRv5_rec_mobile.onnx",
    },
}


class ImageFormatOption:
    def __init__(self, pipeline_options):
        self.pipeline_options = pipeline_options


class _Native:
    def __init__(self, data):
        self.data = data

    def model_dump(self, mode):
        assert mode == "json"
        return self.data


class _Document:
    def __init__(self, data):
        self.data = data

    def export_to_dict(self):
        return self.data

    def export_to_markdown(self):
        return "\n\n".join(t["text"] for t in self.data["texts"])


class DocumentConverter:
    def __init__(self, allowed_formats, format_options):
        assert allowed_formats == [InputFormat.IMAGE]
        self.options = format_options[InputFormat.IMAGE].pipeline_options
        assert (
            self.options.ocr_options.rapidocr_params["EngineConfig.onnxruntime.use_cuda"] is False
        )
        self.initialized = False
        reader = SimpleNamespace(cfg=None, preprocess_img=lambda image: (image, {}))
        self.ocr_model = SimpleNamespace(reader=reader, get_ocr_rects=None)

    def initialize_pipeline(self, input_format):
        assert input_format == InputFormat.IMAGE
        self.initialized = True
        cfg = {section: dict(values) for section, values in RAPIDOCR.items()}
        for values in cfg.values():
            if "model_path" in values:
                values["model_path"] = Path(self.options.artifacts_path) / values["model_path"]
        self.ocr_model.reader.cfg = cfg
        self.ocr_model.get_ocr_rects = self._ocr_rects

    def _ocr_rects(self, page):
        if self.options.ocr_options.mode.value == "full_page":
            size = page["size"]
            boxes = [{"l": 0, "t": 0, "r": size["width"], "b": size["height"]}]
        else:
            boxes = [cluster["bbox"] for cluster in page["clusters"]]
        origin = SimpleNamespace(value="TOPLEFT")
        return [SimpleNamespace(coord_origin=origin, **{k: b[k] for k in "ltrb"}) for b in boxes]

    def _get_pipeline(self, input_format):
        assert self.initialized
        assert input_format == InputFormat.IMAGE
        return SimpleNamespace(ocr_model=self.ocr_model)

    def convert(self, source, raises_on_error):
        assert self.initialized
        assert raises_on_error is False
        with Image.open(source.stream) as image:
            assert image.format == "TIFF"
            dpi = 72 * self.options.ocr_options.scale
            assert tuple(float(v) for v in image.info["dpi"]) == (dpi, dpi)
            width, height = image.size
        logging.getLogger("docling.fake").warning("fake warning for %s", source.name)
        if width == 666:
            raise RuntimeError("CUDA out of memory (fake)")
        if width == 777:
            error = SimpleNamespace(component_type="pipeline", error_message="fake failure")
            return SimpleNamespace(status=ConversionStatus.FAILURE, errors=[error])
        name = RESULTS.get(width, "blank")
        document = json.loads((FIXTURES / name / "docling.json").read_text(encoding="utf-8"))
        layout = json.loads((FIXTURES / name / "docling_layout.json").read_text(encoding="utf-8"))
        if name == "blank":
            size = {"width": width * 72 / dpi, "height": height * 72 / dpi}
            if width == 999:
                size = {"width": size["height"], "height": size["width"]}
            document["pages"]["1"]["size"] = layout["pages"][0]["size"] = size
            if width == 910:
                rect = {"r_x0": 1, "r_y0": 9, "r_x1": 9, "r_y1": 9, "r_x2": 9, "r_y2": 1}
                rect |= {"r_x3": 1, "r_y3": 1, "coord_origin": "TOPLEFT"}
                bbox = {"l": 1, "t": 1, "r": 9, "b": 9, "coord_origin": "TOPLEFT"}
                cell = {"index": 0, "text": "orphan", "rect": rect, "confidence": 0.9}
                layout["pages"][0]["clusters"] = [
                    {
                        "id": 0,
                        "label": "text",
                        "bbox": bbox,
                        "confidence": 0.9,
                        "cells": [cell],
                        "children": [],
                    }
                ]
        page = layout["pages"][0]
        for rect in self.ocr_model.get_ocr_rects(page):
            crop = (round((rect.b - rect.t) * 3), round((rect.r - rect.l) * 3), 3)
            if crop[0] and crop[1]:
                self.ocr_model.reader.preprocess_img(SimpleNamespace(shape=crop))
        pages = [
            SimpleNamespace(
                page_no=page["page_no"],
                size=_Native(page["size"]),
                predictions=SimpleNamespace(
                    layout=SimpleNamespace(clusters=[_Native(c) for c in page["clusters"]])
                ),
            )
        ]
        status, errors = ConversionStatus.SUCCESS, []
        if width == 888:
            status = ConversionStatus.PARTIAL_SUCCESS
            errors = [SimpleNamespace(component_type="timeout", error_message="fake timeout")]
        return SimpleNamespace(
            status=status, errors=errors, document=_Document(document), pages=pages
        )
