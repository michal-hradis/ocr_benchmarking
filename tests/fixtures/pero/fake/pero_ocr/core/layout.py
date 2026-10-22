"""Fake PageLayout serializing the committed PERO output (../../../page.xml, alto.xml)."""

from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[3]
EMPTY = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
    '<Page imageFilename="{id}" imageWidth="{w}" imageHeight="{h}"/></PcGts>\n'
)


class PageLayout:
    def __init__(self, id=None, page_size=(0, 0), file=None):
        self.id = id
        self.page_size = page_size  # (height, width)
        self.regions = []
        self.stages = []  # the fake parsers that processed this layout, in order

    def to_pagexml_string(self):
        if self.regions:
            return (FIXTURES / "page.xml").read_text(encoding="utf-8")
        height, width = self.page_size
        return EMPTY.format(id=self.id, w=width, h=height)

    def to_altoxml_string(self, word_splitters=None):
        assert word_splitters is None  # as in PERO's parse_folder.py
        if self.page_size[1] == 555:
            raise ValueError("fake ALTO error")
        return (FIXTURES / "alto.xml").read_text(encoding="utf-8") if self.regions else "<alto/>"
