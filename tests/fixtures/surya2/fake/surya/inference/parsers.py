from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass
class ParsedFullPageBlock:
    label: str
    bbox: tuple
    html: str


class _Blocks(HTMLParser):
    """Top-level divs with their inner HTML as written."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.blocks, self.depth, self.current = [], 0, None

    def handle_starttag(self, tag, attrs):
        if tag == "div" and self.depth == 0:
            self.current = (dict(attrs), [])
        elif self.current:
            self.current[1].append(self.get_starttag_text())
        self.depth += tag == "div"

    def handle_startendtag(self, tag, attrs):
        if self.current:
            self.current[1].append(self.get_starttag_text())

    def handle_endtag(self, tag):
        self.depth -= tag == "div"
        if tag == "div" and self.depth == 0 and self.current:
            self.blocks.append(self.current)
            self.current = None
        elif self.current:
            self.current[1].append(f"</{tag}>")

    def handle_data(self, data):
        if self.current:
            self.current[1].append(data)

    def handle_entityref(self, name):
        self.handle_data(f"&{name};")

    def handle_charref(self, name):
        self.handle_data(f"&#{name};")


def parse_full_page_html(text):
    parser = _Blocks()
    parser.feed(text)
    parser.close()
    if parser.current:  # an unclosed last div, as BeautifulSoup closes it
        parser.blocks.append(parser.current)
    out = []
    for attrs, parts in parser.blocks:
        label, box = attrs.get("data-label"), (attrs.get("data-bbox") or "").split()
        if label and len(box) == 4:
            out.append(ParsedFullPageBlock(label, tuple(map(float, box)), "".join(parts).strip()))
    return out
