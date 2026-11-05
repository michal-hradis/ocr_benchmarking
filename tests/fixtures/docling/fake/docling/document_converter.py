"""Fake DocumentConverter returning the committed Docling outputs, chosen by the image width.

900: page; 520: paragraph; 1000: columns; 1500: newspaper; 1200: table; 666: a CUDA
out-of-memory error; 777: a failed conversion; 888: a partial conversion (blank page); 999: a
page whose size is the image turned by 90 degrees; 910: a blank page with one OCR cell in a
cluster that no item uses; any other width: the blank page with that
image size. The input must be a TIFF at 72 x ocr_scale DPI, and every conversion logs a warning.

With RapidOCR, the outputs are those of preset docling (../../<name>/). The pipeline's RapidOCR
reader has RapidOCR 3.10.0's settings and the pinned models; the OCR model is asked for its crops
once per page: the whole page in full_page mode, else the layout clusters. RapidOCR preprocesses
each crop of non-zero area at 3 pixels per point, without resizing it.

With the Tesseract command line (tesseract) or the tesserocr bindings, the outputs are those of
preset docling-tesseract-cli (../../tesseract-cli/<name>/) or docling-tesserocr
(../../tesserocr/<name>/) for page, paragraph, newspaper and table, else blank. The OCR model
reports the languages in TESSDATA_PREFIX and the Tesseract version FAKE_TESSERACT_VERSION (default
5.3.4). The command line is called once per call recorded in tesseract.json, with that call's
image size, OSD orientation and TSV rows; at width 555 the call fails, which Docling logs and
skips. The bindings read the whole page once, with one OSD pass, finding the page's OCR cells.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from docling.datamodel.base_models import ConversionStatus, InputFormat

FIXTURES = Path(__file__).resolve().parents[2]
RESULTS = {900: "page", 520: "paragraph", 1000: "columns", 1500: "newspaper", 1200: "table"}
TESSERACT_RESULTS = {"tesseract": "tesseract-cli", "tesserocr": "tesserocr"}
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


class _Rows:
    """The part of a pandas DataFrame of TSV rows that the adapter reads."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def to_dict(self, orient):
        assert orient == "records"
        return [dict(record) for record in self.records]


class _Reader:
    """A tesserocr PyTessBaseAPI: an image, its text lines and the settings it was made with."""

    def __init__(self, options, lines, osd=False):
        self.options, self.lines, self.osd = options, lines, osd

    def SetImage(self, image):
        self.image = image

    def DetectOrientationScript(self):
        assert self.osd
        return {"orient_deg": 0, "script_name": "Latin"}

    def GetComponentImages(self, level, text_only):
        return [None] * self.lines()

    def GetDatapath(self):
        return self.options.path + "/"

    def GetInitLanguagesAsString(self):
        return "osd" if self.osd else "+".join(self.options.lang)

    def GetPageSegMode(self):
        return 0 if self.osd else self.options.psm

    def oem(self):
        return 3


class DocumentConverter:
    def __init__(self, allowed_formats, format_options):
        assert allowed_formats == [InputFormat.IMAGE]
        self.options = format_options[InputFormat.IMAGE].pipeline_options
        self.kind = self.options.ocr_options.kind
        if self.kind == "rapidocr":
            params = self.options.ocr_options.rapidocr_params
            assert params["EngineConfig.onnxruntime.use_cuda"] is False
        self.initialized = False
        reader = SimpleNamespace(cfg=None, preprocess_img=lambda image: (image, {}))
        self.ocr_model = SimpleNamespace(reader=reader, get_ocr_rects=None)
        self.calls = []

    def initialize_pipeline(self, input_format):
        assert input_format == InputFormat.IMAGE
        self.initialized = True
        model = self.ocr_model
        model.get_ocr_rects = self._ocr_rects
        if self.kind == "rapidocr":
            cfg = {section: dict(values) for section, values in RAPIDOCR.items()}
            for values in cfg.values():
                if "model_path" in values:
                    values["model_path"] = Path(self.options.artifacts_path) / values["model_path"]
            model.reader.cfg = cfg
            return
        options = self.options.ocr_options
        tessdata = Path(os.environ["TESSDATA_PREFIX"])
        assert Path(options.path) == tessdata
        model.options = options
        model._auto_script = not options.lang
        model._tesseract_vocabulary = sorted(p.stem for p in tessdata.glob("*.traineddata"))
        if self.kind == "tesseract":
            version = os.environ.get("FAKE_TESSERACT_VERSION", "5.3.4")
            model._get_name_and_version = lambda: ("tesseract", version)
            model._safe_tesseract_cmd = options.tesseract_cmd
            model._safe_tessdata_path = str(Path(options.path).resolve())
            model._native_codes = list(options.lang)
            model._run_tesseract = self._run_tesseract
        else:
            model._reader = _Reader(options, lambda: self.lines)
            model._osd_reader = _Reader(options, lambda: 0, osd=True)

    def _ocr_rects(self, page):
        size = page["size"]
        if self.options.ocr_options.mode.value == "full_page":
            boxes = [{"l": 0, "t": 0, "r": size["width"], "b": size["height"]}]
        elif self.kind != "rapidocr":  # one crop per recorded call, the bindings: the page once
            crops = len(self.calls) if self.kind == "tesseract" else int(bool(page["clusters"]))
            boxes = [{"l": 0, "t": 0, "r": size["width"], "b": size["height"]}] * crops
        else:
            boxes = [cluster["bbox"] for cluster in page["clusters"]]
        origin = SimpleNamespace(value="TOPLEFT")
        return [SimpleNamespace(coord_origin=origin, **{k: b[k] for k in "ltrb"}) for b in boxes]

    def _run_tesseract(self, filename, osd):
        if self.width == 555:
            raise RuntimeError("tesseract returned non-zero exit status 1 (fake)")
        return _Rows(self.rows.pop(0))

    def _tesseract_ocr(self, page):
        """Docling's Tesseract models: per crop, OSD, then the command line or the bindings."""
        model = self.ocr_model
        for rect, call in zip(model.get_ocr_rects(page), self.calls or [None], strict=False):
            size = (round((rect.r - rect.l) * 3), round((rect.b - rect.t) * 3))
            if self.kind == "tesserocr":
                image = Image.new("L", size, 255)
                model._osd_reader.SetImage(image)
                model._osd_reader.DetectOrientationScript()
                model._reader.SetImage(image)
                self.lines = sum(len(c["cells"]) for c in page["clusters"])
                model._reader.GetComponentImages(None, True)
                continue
            orientation = call["osd_orientation"]
            osd = None
            if orientation is not None:
                osd = {"key": ["Orientation in degrees"], "value": [f" {orientation}"]}
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file:
                Image.new("L", call["image"], 255).save(file, format="PNG")
            try:
                model._run_tesseract(file.name, osd)
            except RuntimeError as error:
                logging.getLogger("docling.fake").error("tesseract OCR failed: %s", error)
            finally:
                os.remove(file.name)

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
        directory = FIXTURES / name
        self.width, self.calls, self.rows = width, [], []
        if self.kind != "rapidocr":
            directory = FIXTURES / TESSERACT_RESULTS[self.kind] / name
            if not directory.is_dir():
                name, directory = "blank", directory.parent / "blank"
        document = json.loads((directory / "docling.json").read_text(encoding="utf-8"))
        layout = json.loads((directory / "docling_layout.json").read_text(encoding="utf-8"))
        if self.kind == "tesseract":
            native = json.loads((directory / "tesseract.json").read_text(encoding="utf-8"))
            self.calls = native["calls"]
            self.rows = [call["rows"] for call in self.calls]
            if width == 555:  # a failing call on an otherwise blank page
                self.calls = [{"image": [width, height], "osd_orientation": 0, "rows": []}]
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
        if self.kind != "rapidocr":
            self._tesseract_ocr(page)
        for rect in self.ocr_model.get_ocr_rects(page) if self.kind == "rapidocr" else []:
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
