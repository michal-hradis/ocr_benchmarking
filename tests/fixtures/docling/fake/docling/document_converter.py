"""Fake DocumentConverter returning the committed Docling outputs, chosen by the image width.

900: page; 520: paragraph; 1000: columns; 1500: newspaper; 1200: table; 666: a CUDA
out-of-memory error; 777: a failed conversion; 888: a partial conversion (blank page); 999: a
page whose size is the image turned by 90 degrees; any other width: the blank page with that
image size. The input must be a TIFF at 72 x ocr_scale DPI, and every conversion logs a warning.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from docling.datamodel.base_models import ConversionStatus, InputFormat

FIXTURES = Path(__file__).resolve().parents[2]
RESULTS = {900: "page", 520: "paragraph", 1000: "columns", 1500: "newspaper", 1200: "table"}


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

    def initialize_pipeline(self, input_format):
        assert input_format == InputFormat.IMAGE
        self.initialized = True

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
        page = layout["pages"][0]
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
