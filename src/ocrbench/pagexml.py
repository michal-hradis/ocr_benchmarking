"""Read PAGE XML ground truth into ordered regions, lines and reference text.

Extraction rules are documented in docs/DATASETS.md. The reader never normalizes text: Unicode
strings are kept exactly as annotated, and normalization belongs to the evaluator (#8).
Coordinates are kept as annotated as well; out-of-bounds points are only reported.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ocrbench.issues import Issue, Severity

PAGE_NAMESPACE_RE = re.compile(
    r"^http://schema\.primaresearch\.org/PAGE/gts/pagecontent/(\d{4}-\d{2}-\d{2})$"
)

Point = tuple[float, float]
Polygon = tuple[Point, ...]

_GROUPS = {"OrderedGroup", "UnorderedGroup", "OrderedGroupIndexed", "UnorderedGroupIndexed"}
_REFS = {"RegionRef", "RegionRefIndexed"}


@dataclass(frozen=True)
class TextLine:
    id: str
    text: str | None  # None when the line has no TextEquiv/Unicode
    coords: Polygon | None
    baseline: Polygon | None


@dataclass(frozen=True)
class TextRegion:
    id: str
    type: str | None
    coords: Polygon | None
    text: str | None  # region-level TextEquiv, used only when no line has text
    lines: tuple[TextLine, ...]
    parent_id: str | None  # enclosing TextRegion for nested regions


@dataclass(frozen=True)
class ReferenceLine:
    """One unit of reference text in reading order."""

    text: str
    region_id: str
    line_id: str | None  # None when the text comes from the region-level TextEquiv
    coords: Polygon | None


@dataclass(frozen=True)
class PageGT:
    schema_version: str  # PAGE namespace date, e.g. "2019-07-15"
    image_filename: str | None
    width: int
    height: int
    regions: tuple[TextRegion, ...]  # in reading order
    reading_order: Literal["page", "document"]  # from <ReadingOrder> or XML document order
    reference_lines: tuple[ReferenceLine, ...]


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _number(token: str) -> float:
    try:
        return int(token)
    except ValueError:
        return float(token)


class _Reader:
    def __init__(self, path: Path, sample_id: str | None) -> None:
        self.path = path
        self.sample_id = sample_id
        self.issues: list[Issue] = []

    def issue(self, severity: Severity, code: str, message: str, element_id: str | None = None):
        location = f"{self.path.name}#{element_id}" if element_id else self.path.name
        self.issues.append(Issue(severity, code, message, self.sample_id, location))

    def children(self, element: ET.Element, name: str) -> Iterator[ET.Element]:
        return (child for child in element if _local(child.tag) == name)

    def child(self, element: ET.Element, name: str) -> ET.Element | None:
        return next(self.children(element, name), None)

    def polygon(self, element: ET.Element | None, owner_id: str | None) -> Polygon | None:
        if element is None:
            return None
        points = element.get("points")
        try:
            if points is not None:
                pairs = (pair.split(",") for pair in points.split())
                polygon = tuple((_number(x), _number(y)) for x, y in pairs)
            else:  # PAGE 2009/2010 use <Point x="" y=""/> children
                polygon = tuple(
                    (_number(p.get("x", "")), _number(p.get("y", "")))
                    for p in self.children(element, "Point")
                )
        except ValueError:
            self.issue(
                Severity.ERROR,
                "malformed-points",
                f"cannot parse {_local(element.tag)} points {points!r}",
                owner_id,
            )
            return None
        return polygon or None

    def text_equiv(self, element: ET.Element) -> str | None:
        """Text of the primary TextEquiv: the lowest @index, else the first one."""
        equivs = list(self.children(element, "TextEquiv"))
        indexed = [e for e in equivs if (e.get("index") or "").lstrip("-").isdigit()]
        if indexed:
            equivs = [min(indexed, key=lambda e: int(e.get("index", "0")))]
        for equiv in equivs[:1]:
            unicode = self.child(equiv, "Unicode")
            if unicode is not None:
                return unicode.text or ""
        return None

    def read(self) -> PageGT | None:
        try:
            root = ET.parse(self.path).getroot()
        except ET.ParseError as error:
            line, column = error.position
            self.issue(Severity.ERROR, "xml-parse-error", f"line {line}, column {column}: {error}")
            return None
        except OSError as error:
            self.issue(Severity.ERROR, "gt-unreadable", str(error))
            return None

        namespace = root.tag[1:].split("}")[0] if root.tag.startswith("{") else ""
        version = PAGE_NAMESPACE_RE.match(namespace)
        if _local(root.tag) != "PcGts" or version is None:
            self.issue(
                Severity.ERROR,
                "not-page-xml",
                f"root element {root.tag!r} is not a PAGE PcGts element",
            )
            return None
        page = self.child(root, "Page")
        if page is None:
            self.issue(Severity.ERROR, "missing-page", "PcGts has no Page element")
            return None
        try:
            width, height = int(page.get("imageWidth", "")), int(page.get("imageHeight", ""))
        except ValueError:
            self.issue(
                Severity.ERROR,
                "missing-page-size",
                "Page needs integer imageWidth and imageHeight",
            )
            return None

        self.check_duplicate_ids(page)
        regions = self.collect_regions(page, parent_id=None)
        ordered, source = self.order_regions(page, regions)
        reference = tuple(self.reference_lines(ordered))
        if not reference:
            self.issue(Severity.WARNING, "no-reference-text", "the page has no reference text")
        self.check_bounds(ordered, width, height)
        return PageGT(
            schema_version=version.group(1),
            image_filename=page.get("imageFilename"),
            width=width,
            height=height,
            regions=tuple(ordered),
            reading_order=source,
            reference_lines=reference,
        )

    def check_duplicate_ids(self, page: ET.Element) -> None:
        counts = Counter(e.get("id") for e in page.iter() if e.get("id") is not None)
        for element_id, count in counts.items():
            if count > 1:
                self.issue(
                    Severity.WARNING,
                    "duplicate-id",
                    f"id is used by {count} elements",
                    element_id,
                )

    def collect_regions(self, element: ET.Element, parent_id: str | None) -> list[TextRegion]:
        """All TextRegions in document order, including nested ones and table cells."""
        regions: list[TextRegion] = []
        for child in element:
            if _local(child.tag) == "TextRegion":
                region = self.region(child, parent_id)
                regions.append(region)
                regions.extend(self.collect_regions(child, region.id))
            elif _local(child.tag) not in {"TextLine", "ReadingOrder"}:
                regions.extend(self.collect_regions(child, parent_id))
        return regions

    def region(self, element: ET.Element, parent_id: str | None) -> TextRegion:
        region_id = element.get("id", "")
        if not region_id:
            self.issue(Severity.WARNING, "missing-id", "TextRegion has no id")
        lines = [self.line(e, region_id) for e in self.children(element, "TextLine")]
        return TextRegion(
            id=region_id,
            type=element.get("type"),
            coords=self.polygon(self.child(element, "Coords"), region_id),
            text=self.text_equiv(element),
            lines=tuple(self.order_lines(element, lines, region_id)),
            parent_id=parent_id,
        )

    def line(self, element: ET.Element, region_id: str) -> TextLine:
        line_id = element.get("id", "")
        if not line_id:
            self.issue(
                Severity.WARNING, "missing-id", f"a TextLine in region {region_id} has no id"
            )
        return TextLine(
            id=line_id,
            text=self.text_equiv(element),
            coords=self.polygon(self.child(element, "Coords"), line_id),
            baseline=self.polygon(self.child(element, "Baseline"), line_id),
        )

    def order_lines(
        self, element: ET.Element, lines: list[TextLine], region_id: str
    ) -> list[TextLine]:
        """Order lines by TextLine/@index when every line has a distinct one."""
        indexes = [e.get("index") for e in self.children(element, "TextLine")]
        if all(i is None for i in indexes):
            return lines
        if all(i is not None and i.lstrip("-").isdigit() for i in indexes):
            numbers = [int(i) for i in indexes if i is not None]
            if len(set(numbers)) == len(numbers):
                return [line for _, line in sorted(zip(numbers, lines, strict=True))]
        self.issue(
            Severity.WARNING,
            "inconsistent-line-index",
            "TextLine @index is missing, duplicated or not an integer; using document order",
            region_id,
        )
        return lines

    def order_regions(
        self, page: ET.Element, regions: list[TextRegion]
    ) -> tuple[list[TextRegion], Literal["page", "document"]]:
        reading_order = self.child(page, "ReadingOrder")
        if reading_order is None:
            self.issue(
                Severity.INFO,
                "no-reading-order",
                "no ReadingOrder element; using document order of regions",
            )
            return regions, "document"

        by_id = {r.id: r for r in regions if r.id}
        all_ids = {e.get("id") for e in page.iter()}
        referenced: dict[str, None] = {}  # insertion-ordered set
        for ref in self.walk_group(reading_order):
            if ref in referenced:
                self.issue(
                    Severity.WARNING,
                    "duplicate-reading-order-ref",
                    "region is referenced more than once; keeping the first position",
                    ref,
                )
            elif ref in by_id:
                referenced[ref] = None
            elif ref not in all_ids:
                self.issue(
                    Severity.WARNING,
                    "unknown-reading-order-ref",
                    "ReadingOrder references a missing element",
                    ref,
                )
            # references to non-text regions (images, separators) are legitimate and ignored

        children: dict[str | None, list[TextRegion]] = {}
        for region in regions:
            children.setdefault(region.parent_id, []).append(region)
        ordered: list[TextRegion] = []
        placed: set[int] = set()  # object ids: regions without @id must not collide

        def place(region: TextRegion) -> None:
            ordered.append(region)
            placed.add(id(region))
            # nested regions without their own position follow their parent
            for child in children.get(region.id, []):
                if child.id not in referenced:
                    place(child)

        for ref in referenced:
            place(by_id[ref])
        for region in regions:
            if id(region) not in placed:
                if region.text or any(line.text for line in region.lines):
                    self.issue(
                        Severity.WARNING,
                        "region-not-in-reading-order",
                        "text region is missing from ReadingOrder; appended at the end",
                        region.id,
                    )
                place(region)
        return ordered, "page"

    def walk_group(self, element: ET.Element) -> Iterator[str]:
        """Region references of a ReadingOrder (sub)tree in reading order."""
        if _local(element.tag) in _GROUPS and element.get("regionRef"):
            yield element.get("regionRef", "")
        members = [c for c in element if _local(c.tag) in _GROUPS | _REFS]
        if _local(element.tag).startswith("Ordered"):
            keys = [m.get("index") for m in members]
            if all(k is not None and k.isdigit() for k in keys):
                members = [m for _, m in sorted(zip(map(int, keys), members, strict=True))]
            else:
                self.issue(
                    Severity.WARNING,
                    "inconsistent-reading-order-index",
                    "OrderedGroup members lack integer @index; using document order",
                    element.get("id"),
                )
        elif _local(element.tag).startswith("Unordered"):
            self.issue(
                Severity.INFO,
                "unordered-group",
                "UnorderedGroup has no defined order; using document order",
                element.get("id"),
            )
        for member in members:
            if _local(member.tag) in _REFS:
                if member.get("regionRef"):
                    yield member.get("regionRef", "")
            else:
                yield from self.walk_group(member)

    def reference_lines(self, regions: Iterable[TextRegion]) -> Iterator[ReferenceLine]:
        """Line text where any line has text; region text only as a fallback. Never both."""
        for region in regions:
            with_text = [line for line in region.lines if line.text and line.text.strip()]
            for line in region.lines:
                if not (line.text and line.text.strip()):
                    self.issue(
                        Severity.WARNING,
                        "line-without-text",
                        "TextLine has no text; skipped in the reference",
                        line.id or region.id,
                    )
            if with_text:
                if region.text and region.text.strip():
                    self.issue(
                        Severity.INFO,
                        "region-text-ignored",
                        "region-level TextEquiv ignored because its lines have text",
                        region.id,
                    )
                for line in with_text:
                    yield ReferenceLine(line.text or "", region.id, line.id, line.coords)
            elif region.text and region.text.strip():
                yield ReferenceLine(region.text, region.id, None, region.coords)

    def check_bounds(self, regions: Iterable[TextRegion], width: int, height: int) -> None:
        outside, first = 0, None
        for region in regions:
            shapes = [(region.id, region.coords)]
            shapes += [(line.id, s) for line in region.lines for s in (line.coords, line.baseline)]
            for owner, polygon in shapes:
                for x, y in polygon or ():
                    if not (0 <= x <= width and 0 <= y <= height):
                        outside += 1
                        first = first or owner
        if outside:
            self.issue(
                Severity.INFO,
                "points-out-of-bounds",
                f"{outside} point(s) outside the {width}x{height} image; kept as annotated",
                first,
            )


def read_page_xml(path: Path, sample_id: str | None = None) -> tuple[PageGT | None, list[Issue]]:
    """Parse a PAGE XML file.

    Returns None if the file cannot be read as PAGE. Other errors (e.g. malformed points) are
    reported in the issues next to a partially read page.
    """
    reader = _Reader(Path(path), sample_id)
    page = reader.read()
    return page, reader.issues
