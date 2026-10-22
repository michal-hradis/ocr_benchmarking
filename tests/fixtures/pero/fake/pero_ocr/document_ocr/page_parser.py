"""Fake PageParser: the layout stage finds the committed page on 900-pixel-wide images.

The stage is the one the INI enables. Layout analysis finds the committed regions and lines
(../../../page.xml) on a 900-pixel-wide image, nothing on other widths, and raises on a
666-pixel-wide image; OCR checks that layout analysis ran first.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

PAGE = ET.parse(Path(__file__).resolve().parents[3] / "page.xml")
NS = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}"


class PageParser:
    def __init__(self, config, device=None, config_path=""):
        parser = config["PAGE_PARSER"]
        self.stage = "layout" if parser.getboolean("RUN_LAYOUT_PARSER", False) else "ocr"
        assert Path(config_path).is_dir()
        assert device.type in ("cuda", "cpu")

    def process_page(self, image, page_layout):
        height, width = image.shape[:2]
        assert page_layout.page_size == (height, width)
        if self.stage == "layout":
            if width == 666:
                raise RuntimeError("fake layout error")
            if width in (900, 555):
                page_layout.regions = [
                    SimpleNamespace(lines=region.findall(f"{NS}TextLine"))
                    for region in PAGE.iter(f"{NS}TextRegion")
                ]
        else:
            assert page_layout.stages == ["layout"]
        page_layout.stages.append(self.stage)
        return page_layout
