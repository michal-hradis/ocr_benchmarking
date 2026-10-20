"""Shared host-side conversion of native plain text, Markdown and HTML into predictions.

Engines that emit Markdown or HTML (typically image-to-text models) store that native output
unchanged; the host converts it after inference, so a fix to a rule here means re-converting,
never re-running a model. The result is plain text without markup plus typed regions without
positions. Rules are documented in docs/PREDICTIONS.md; any change to them must increment
CONVERTER_VERSION, which is recorded in every converted prediction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.dollarmath import dollarmath_plugin

from ocrbench.prediction import PREDICTION_FORMAT, PREDICTION_VERSION, REGION_SEPARATOR

CONVERTER_VERSION = 1
FORMATS = ("text", "markdown", "html")
SUFFIXES = {".txt": "text", ".md": "markdown", ".markdown": "markdown", ".html": "html"}
SUFFIXES[".htm"] = "html"
CELL_SEPARATOR = "\t"


@dataclass
class _Block:
    type: str
    native_type: str
    text: str


def convert_native(content: str, source_format: str, sample_id: str) -> dict[str, Any]:
    """Convert one native output to prediction JSON (not yet validated)."""
    if source_format == "text":
        blocks: list[_Block] = []
        text = content
    elif source_format in ("markdown", "html"):
        blocks = _markdown(content) if source_format == "markdown" else _html(content)
        blocks = [block for block in blocks if block.text.strip()]
        text = REGION_SEPARATOR.join(block.text for block in blocks)
    else:
        raise ValueError(f"unknown native format {source_format!r}; expected one of {FORMATS}")
    data: dict[str, Any] = {
        "format": PREDICTION_FORMAT,
        "version": PREDICTION_VERSION,
        "sample_id": sample_id,
        "text": text,
    }
    if blocks:
        data["regions"] = [
            {
                "id": f"r{index}",
                "type": block.type,
                "native_type": block.native_type,
                "text": block.text,
            }
            for index, block in enumerate(blocks, 1)
        ]
    data["conversion"] = {"from": source_format, "version": CONVERTER_VERSION}
    return data


def _tidy(text: str) -> str:
    """Strip each line and drop blank lines inside a block."""
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


# Markdown ------------------------------------------------------------------------------------

_BR_RE = re.compile(r"^<br\s*/?>$", re.IGNORECASE)


def _parser() -> MarkdownIt:
    parser = MarkdownIt("commonmark").enable(["table", "strikethrough"])
    return parser.use(dollarmath_plugin, allow_space=True, double_inline=True)


def _inline(token: Token) -> str:
    parts = []
    for child in token.children or []:
        if child.type in ("text", "code_inline", "math_inline", "math_inline_double"):
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak") or (
            child.type == "html_inline" and _BR_RE.match(child.content.strip())
        ):
            parts.append("\n")
        # images (alt text is not page text), markup and other inline HTML tags add nothing
    return "".join(parts)


def _markdown(content: str) -> list[_Block]:
    blocks: list[_Block] = []
    tokens = _parser().parse(content)
    items: list[tuple[_Block, str]] = []  # open list items and their printed number
    heading: str | None = None
    row: list[str] | None = None
    table: _Block | None = None
    for token in tokens:
        kind = token.type
        if kind == "heading_open":
            heading = token.tag
        elif kind == "heading_close":
            heading = None
        elif kind == "list_item_open":
            # an ordered item's number is printed text; a bullet marker is not
            prefix = f"{token.info}{token.markup} " if token.info else ""
            block = _Block("list-item", "li", "")
            blocks.append(block)
            items.append((block, prefix))
        elif kind == "list_item_close":
            items.pop()
        elif kind == "table_open":
            table = _Block("table", "table", "")
            blocks.append(table)
        elif kind == "table_close":
            table = None
        elif kind == "tr_open":
            row = []
        elif kind == "tr_close" and table is not None and row is not None:
            line = CELL_SEPARATOR.join(row)
            table.text = f"{table.text}\n{line}" if table.text else line
            row = None
        elif kind == "inline":
            text = _inline(token)
            if row is not None:
                row.append(" ".join(text.split()))
            elif heading:
                blocks.append(_Block("heading", heading, _tidy(text)))
            elif items:
                item, prefix = items[-1]  # text goes to the innermost item
                item.text += f"\n{_tidy(text)}" if item.text else prefix + _tidy(text)
            else:
                blocks.append(_Block("text", "p", _tidy(text)))
        elif kind == "math_block" or kind == "math_block_label":
            blocks.append(_Block("formula", "math_block", token.content.strip()))
        elif kind in ("fence", "code_block"):
            blocks.append(_Block("other", kind, token.content.rstrip("\n")))
        elif kind == "html_block":
            blocks.extend(_html(token.content))
        # hr, blockquote and list containers add no text of their own
    return blocks


# HTML ----------------------------------------------------------------------------------------

_VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "source",
    "wbr",
}
_SKIP = {"script", "style", "head", "title", "template", "noscript"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCKS = _HEADINGS | {"p", "li", "table", "caption", "figcaption", "pre", "math"}
_BLOCKS |= {"address", "article", "aside", "blockquote", "body", "dd", "div", "dl", "dt"}
_BLOCKS |= {"figure", "footer", "header", "html", "main", "nav", "ol", "section", "ul"}
# Opening a tag implicitly closes these open elements, as browsers do for common cases.
_AUTO_CLOSE = {"li": {"li", "p"}, "dt": {"dt", "dd", "p"}, "dd": {"dt", "dd", "p"}}
_AUTO_CLOSE |= {"tr": {"tr", "td", "th"}, "td": {"td", "th"}, "th": {"td", "th"}}
_TYPES = {"p": "text", "li": "list-item", "caption": "caption", "figcaption": "caption"}
_TYPES |= {"pre": "other", "table": "table", "math": "formula"} | dict.fromkeys(
    _HEADINGS, "heading"
)


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str | None]
    children: list[_Node | str]


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", {}, [])
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        closes = _AUTO_CLOSE.get(tag, set())
        if tag in _BLOCKS:
            closes = closes | {"p"}
        while len(self.stack) > 1 and self.stack[-1].tag in closes:
            self.stack.pop()
        node = _Node(tag, dict(attrs), [])
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.stack[-1].children.append(_Node(tag, dict(attrs), []))

    def handle_endtag(self, tag: str) -> None:
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth].tag == tag:
                del self.stack[depth:]
                return
        # a stray end tag is ignored

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def _is_block(node: _Node | str) -> bool:
    if isinstance(node, str):
        return False
    if node.tag == "math":
        return node.attrs.get("display") == "block"
    return node.tag in _BLOCKS


def _math_text(node: _Node) -> str:
    """MathML as its TeX annotation when present, otherwise as its text content."""
    stack: list[_Node | str] = [node]
    while stack:
        child = stack.pop()
        if isinstance(child, _Node):
            if child.tag == "annotation" and "tex" in (child.attrs.get("encoding") or ""):
                return "".join(c for c in child.children if isinstance(c, str)).strip()
            stack.extend(reversed(child.children))
    return " ".join(_text(node, pre=False).split())


def _text(node: _Node, *, pre: bool) -> str:
    """Text content with HTML whitespace collapsing; <br> and block boundaries become newlines."""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child if pre else re.sub(r"\s+", " ", child))
        elif child.tag in _SKIP or child.tag in ("img", "annotation"):
            continue
        elif child.tag == "br":
            parts.append("\n")
        elif child.tag == "math":
            parts.append(_math_text(child))
        elif child.tag in ("td", "th"):
            parts.append(" " + _text(child, pre=pre) + " ")
        else:
            inner = _text(child, pre=pre or child.tag == "pre")
            parts.append(f"\n{inner}\n" if _is_block(child) or child.tag == "tr" else inner)
    return "".join(parts)


def _rows(node: _Node) -> list[_Node]:
    """Table rows, not descending into nested tables."""
    rows = []
    for child in node.children:
        if isinstance(child, _Node):
            if child.tag == "tr":
                rows.append(child)
            elif child.tag != "table":
                rows.extend(_rows(child))
    return rows


def _walk(node: _Node, blocks: list[_Block]) -> None:
    inline: list[_Node | str] = []

    def flush() -> None:
        if inline:
            text = _tidy(_text(_Node(node.tag, {}, list(inline)), pre=False))
            blocks.append(_Block("text", node.tag, text))
            inline.clear()

    number = None
    if node.tag == "ol":
        start = (node.attrs.get("start") or "1").strip()
        number = int(start) if start.lstrip("-").isdigit() else 1
    for child in node.children:
        if not _is_block(child) and not (isinstance(child, _Node) and child.tag in _SKIP):
            inline.append(child)
            continue
        flush()
        if isinstance(child, str) or child.tag in _SKIP:
            continue
        tag = child.tag
        if tag == "table":
            for caption in (c for c in child.children if isinstance(c, _Node)):
                if caption.tag == "caption":
                    blocks.append(_Block("caption", "caption", _tidy(_text(caption, pre=False))))
            cells = [
                [
                    " ".join(_text(c, pre=False).split())
                    for c in row.children
                    if isinstance(c, _Node) and c.tag in ("td", "th")
                ]
                for row in _rows(child)
            ]
            text = "\n".join(CELL_SEPARATOR.join(row) for row in cells if any(row))
            blocks.append(_Block("table", "table", text))
        elif tag == "math":
            blocks.append(_Block("formula", "math", _math_text(child)))
        elif tag == "pre":
            blocks.append(_Block("other", "pre", _text(child, pre=True).strip("\n")))
        elif tag in _TYPES:
            nested = [c for c in child.children if isinstance(c, _Node) and c.tag in ("ul", "ol")]
            own = _Node(tag, child.attrs, [c for c in child.children if c not in nested])
            text = _tidy(_text(own, pre=False))
            if tag == "li" and number is not None:
                text = f"{number}. {text}"
                number += 1
            blocks.append(_Block(_TYPES[tag], tag, text))
            for sub in nested:
                _walk(sub, blocks)
        else:  # containers: div, section, ul, ol, blockquote, …
            _walk(child, blocks)
    flush()


def _html(content: str) -> list[_Block]:
    builder = _TreeBuilder()
    builder.feed(content)
    builder.close()
    blocks: list[_Block] = []
    _walk(builder.root, blocks)
    return blocks
