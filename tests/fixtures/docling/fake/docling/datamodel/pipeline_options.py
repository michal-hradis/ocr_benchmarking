"""Fake docling.datamodel.pipeline_options: option objects that keep their keyword arguments."""

import json
from enum import Enum, StrEnum


class OcrMode(StrEnum):
    FULL_PAGE = "full_page"
    LAYOUT_REGIONS = "layout_regions"
    PDF_AWARE_LAYOUT_REGIONS = "pdf_aware_layout_regions"
    DEFAULT = "default"


class _Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def dump(self):
        def plain(value):
            if isinstance(value, _Options):
                return value.dump()
            return value.value if isinstance(value, Enum) else value

        return {key: plain(value) for key, value in self.__dict__.items()}

    def model_dump_json(self):
        return json.dumps(self.dump())


class RapidOcrOptions(_Options):
    pass


class PdfPipelineOptions(_Options):
    pass
