"""Standardized OCR predictions: JSON Schema check, cross-field validation and standardization.

The format is documented in docs/PREDICTIONS.md (ADR 0006). A prediction is engine output and
therefore untrusted: structure errors make it invalid, while recoverable problems are reported
as warnings and fixed in the standardized result (out-of-bounds points are clipped, and page
text that disagrees with the line/region text is rebuilt from the structure). A prediction with
any error is never usable as a successful OCR result; the runner (#7) records that failure.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from ocrbench.issues import Issue, Severity

PREDICTION_FORMAT = "ocrbench.prediction"
PREDICTION_VERSION = 1
# Recommended region types; other values are allowed and reported as info.
REGION_TYPES = (
    "text",
    "heading",
    "list-item",
    "caption",
    "footnote",
    "page-header",
    "page-footer",
    "page-number",
    "table",
    "formula",
    "figure",
    "other",
)
LINE_SEPARATOR = "\n"
REGION_SEPARATOR = "\n\n"

Point = tuple[float, float]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class Line:
    text: str
    id: str | None = None
    type: str | None = None
    polygon: Polygon | None = None
    baseline: Polygon | None = None
    confidence: float | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class Region:
    id: str
    type: str | None = None
    native_type: str | None = None
    polygon: Polygon | None = None
    text: str | None = None  # None when the region has lines
    lines: tuple[Line, ...] = ()
    confidence: float | None = None
    extra: dict[str, Any] | None = None

    @property
    def content(self) -> str | None:
        """The region's single text source: its lines when present, else its own text."""
        if self.lines:
            return LINE_SEPARATOR.join(line.text for line in self.lines)
        return self.text


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    text: str  # standardized page text; agrees with the structure text
    regions: tuple[Region, ...] = ()  # in reading order
    explicit_reading_order: bool = False
    image_size: tuple[int, int] | None = None
    conversion: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None


def structure_text(regions: tuple[Region, ...]) -> str | None:
    """Join region texts in order; None when no region carries text."""
    parts = [region.content for region in regions if region.content is not None]
    return REGION_SEPARATOR.join(parts) if parts else None


def same_text(a: str, b: str) -> bool:
    """Texts agree when they differ only in the amount or kind of whitespace."""
    return a.split() == b.split()


@cache
def schema() -> dict[str, Any]:
    source = (
        resources.files("ocrbench") / "schemas" / f"prediction-v{PREDICTION_VERSION}.schema.json"
    )
    return json.loads(source.read_text(encoding="utf-8"))


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not valid JSON")


def load_prediction(
    path: Path, *, sample_id: str | None = None, image_size: tuple[int, int] | None = None
) -> tuple[Prediction | None, list[Issue]]:
    """Read and validate a prediction JSON file; see validate_prediction."""
    path = Path(path)
    try:
        data = json.loads(path.read_bytes().decode("utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError) as error:
        return None, [
            Issue(Severity.ERROR, "prediction-unreadable", str(error), sample_id, path.name)
        ]
    except ValueError as error:  # includes json.JSONDecodeError
        return None, [
            Issue(Severity.ERROR, "prediction-json-error", str(error), sample_id, path.name)
        ]
    return validate_prediction(data, sample_id=sample_id, image_size=image_size, location=path.name)


def validate_prediction(
    data: Any,
    *,
    sample_id: str | None = None,
    image_size: tuple[int, int] | None = None,
    location: str | None = None,
) -> tuple[Prediction | None, list[Issue]]:
    """Validate parsed prediction JSON and return its standardized form.

    ``sample_id`` and ``image_size`` are the expected sample and the actual size of its image,
    when known. The prediction is None if there is any error.
    """
    return _Validator(sample_id, location or "prediction").run(data, image_size)


def prediction_to_json(prediction: Prediction) -> dict[str, Any]:
    """Serialize a standardized prediction; validate_prediction accepts the result unchanged."""

    def points(polygon: Polygon) -> list[list[float]]:
        return [[x, y] for x, y in polygon]

    def compact(item: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in item.items() if value is not None}

    data: dict[str, Any] = {
        "format": PREDICTION_FORMAT,
        "version": PREDICTION_VERSION,
        "sample_id": prediction.sample_id,
    }
    if prediction.image_size:
        data["image"] = {"width": prediction.image_size[0], "height": prediction.image_size[1]}
    data["text"] = prediction.text
    if prediction.regions:
        data["regions"] = [
            compact(
                {
                    "id": region.id,
                    "type": region.type,
                    "native_type": region.native_type,
                    "polygon": points(region.polygon) if region.polygon else None,
                    "text": region.text,
                    "lines": [
                        compact(
                            {
                                "id": line.id,
                                "type": line.type,
                                "text": line.text,
                                "polygon": points(line.polygon) if line.polygon else None,
                                "baseline": points(line.baseline) if line.baseline else None,
                                "confidence": line.confidence,
                                "extra": line.extra,
                            }
                        )
                        for line in region.lines
                    ]
                    or None,
                    "confidence": region.confidence,
                    "extra": region.extra,
                }
            )
            for region in prediction.regions
        ]
    if prediction.explicit_reading_order:
        data["reading_order"] = [region.id for region in prediction.regions]
    if prediction.conversion:
        data["conversion"] = prediction.conversion
    if prediction.extra is not None:
        data["extra"] = prediction.extra
    return data


def _pointer(path: Any) -> str:
    return "/" + "/".join(str(part) for part in path)


def _strings(value: Any, path: tuple) -> Any:
    """Yield (path, string) for every string in parsed JSON, keys included."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield (*path, key), key
            yield from _strings(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, (*path, index))


def _area(polygon: Polygon) -> float:
    pairs = zip(polygon, polygon[1:] + polygon[:1], strict=True)
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pairs)) / 2


@dataclass
class _Geometry:
    """Clips points to the image and counts what had to be fixed."""

    width: int
    height: int
    out_of_bounds: int = 0
    clipped: list[str] = field(default_factory=list)
    degenerate: list[str] = field(default_factory=list)

    def clip(self, raw: list[list[float]], where: str, *, area: bool) -> Polygon:
        polygon = []
        outside = 0
        for x, y in raw:
            cx, cy = min(max(x, 0), self.width), min(max(y, 0), self.height)
            outside += (cx, cy) != (x, y)
            polygon.append((cx, cy))
        if outside:
            self.out_of_bounds += outside
            self.clipped.append(where)
        if area and _area(tuple(polygon)) == 0:
            self.degenerate.append(where)
        return tuple(polygon)


class _Validator:
    def __init__(self, sample_id: str | None, location: str) -> None:
        self.sample_id = sample_id
        self.location = location
        self.issues: list[Issue] = []

    def issue(self, severity: Severity, code: str, message: str, path: str = "") -> None:
        where = f"{self.location}#{path}" if path else self.location
        self.issues.append(Issue(severity, code, message, self.sample_id, where))

    def errors(self) -> bool:
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    def run(self, data: Any, actual_size: tuple[int, int] | None):
        validator = Draft202012Validator(schema())
        for error in sorted(validator.iter_errors(data), key=lambda e: list(map(str, e.path))):
            self.issue(Severity.ERROR, "schema-violation", error.message, _pointer(error.path))
        if self.errors():
            return None, self.issues
        for path, value in _strings(data, ()):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                self.issue(
                    Severity.ERROR, "invalid-unicode", "string is not valid UTF-8", _pointer(path)
                )
        if self.sample_id is not None and data["sample_id"] != self.sample_id:
            self.issue(
                Severity.ERROR,
                "sample-id-mismatch",
                f"prediction is for sample {data['sample_id']!r}, expected {self.sample_id!r}",
                "/sample_id",
            )
        self.sample_id = self.sample_id or data["sample_id"]
        size = self.image_size(data, actual_size)
        regions = self.regions(data.get("regions", []), size)
        ordered, explicit = self.reading_order(data, regions)
        if self.errors():
            return None, self.issues

        text = data["text"]
        expected = structure_text(ordered)
        if expected is not None and not same_text(text, expected):
            self.issue(
                Severity.WARNING,
                "text-mismatch",
                "page text differs from the line/region text; it is rebuilt from the structure",
                "/text",
            )
            text = expected
        if not text.strip() and expected is None:
            self.issue(Severity.INFO, "empty-prediction", "the prediction contains no text")
        prediction = Prediction(
            sample_id=data["sample_id"],
            text=text,
            regions=ordered,
            explicit_reading_order=explicit,
            image_size=size,
            conversion=data.get("conversion"),
            extra=data.get("extra"),
        )
        return prediction, self.issues

    def image_size(self, data: dict, actual: tuple[int, int] | None) -> tuple[int, int] | None:
        image = data.get("image")
        declared = (image["width"], image["height"]) if image else None
        if declared and actual and declared != tuple(actual):
            self.issue(
                Severity.ERROR,
                "image-size-mismatch",
                f"prediction declares {declared[0]}x{declared[1]} but the image is "
                f"{actual[0]}x{actual[1]}; positions must refer to the original image",
                "/image",
            )
        has_geometry = any(
            "polygon" in region
            or any("polygon" in line or "baseline" in line for line in region.get("lines", []))
            for region in data.get("regions", [])
        )
        if has_geometry and declared is None:
            self.issue(
                Severity.ERROR,
                "missing-image-size",
                "'image' with width and height is required when positions are given",
            )
        return declared

    def regions(self, raw: list[dict], size: tuple[int, int] | None) -> list[Region]:
        geometry = _Geometry(*size) if size else None
        regions = []
        region_ids: set[str] = set()
        line_ids: set[str] = set()
        for r_index, item in enumerate(raw):
            where = f"/regions/{r_index}"
            if item["id"] in region_ids:
                self.issue(Severity.ERROR, "duplicate-id", f"region id {item['id']!r}", where)
            region_ids.add(item["id"])
            region_type = item.get("type")
            if region_type is not None and region_type not in REGION_TYPES:
                self.issue(
                    Severity.INFO, "unknown-region-type", f"region type {region_type!r}", where
                )
            if item.get("lines") and "text" in item:
                self.issue(
                    Severity.WARNING,
                    "region-text-ignored",
                    "region has lines; its own text is not used",
                    f"{where}/text",
                )
            lines = []
            for l_index, line in enumerate(item.get("lines", [])):
                line_where = f"{where}/lines/{l_index}"
                if "id" in line:
                    if line["id"] in line_ids:
                        self.issue(
                            Severity.ERROR, "duplicate-id", f"line id {line['id']!r}", line_where
                        )
                    line_ids.add(line["id"])
                lines.append(
                    Line(
                        text=line["text"],
                        id=line.get("id"),
                        type=line.get("type"),
                        polygon=self.geometry(geometry, line, "polygon", line_where),
                        baseline=self.geometry(geometry, line, "baseline", line_where),
                        confidence=line.get("confidence"),
                        extra=line.get("extra"),
                    )
                )
            regions.append(
                Region(
                    id=item["id"],
                    type=region_type,
                    native_type=item.get("native_type"),
                    polygon=self.geometry(geometry, item, "polygon", where),
                    text=None if lines else item.get("text"),
                    lines=tuple(lines),
                    confidence=item.get("confidence"),
                    extra=item.get("extra"),
                )
            )
        if geometry and geometry.clipped:
            self.issue(
                Severity.WARNING,
                "points-out-of-bounds",
                f"{geometry.out_of_bounds} point(s) in {len(geometry.clipped)} element(s) lie "
                f"outside the {size[0]}x{size[1]} image and were clipped",
                geometry.clipped[0],
            )
        if geometry and geometry.degenerate:
            self.issue(
                Severity.WARNING,
                "degenerate-polygon",
                f"{len(geometry.degenerate)} polygon(s) have zero area",
                geometry.degenerate[0],
            )
        return regions

    def geometry(self, geometry: _Geometry | None, item: dict, key: str, where: str):
        raw = item.get(key)
        if raw is None or geometry is None:  # a missing image size is already an error
            return None
        if not all(math.isfinite(v) for point in raw for v in point):
            self.issue(Severity.ERROR, "invalid-coordinate", "non-finite coordinate", where)
            return None
        return geometry.clip(raw, f"{where}/{key}", area=key == "polygon")

    def reading_order(self, data: dict, regions: list[Region]) -> tuple[tuple[Region, ...], bool]:
        if "reading_order" not in data:
            return tuple(regions), False
        by_id = {region.id: region for region in regions}
        ordered: list[Region] = []
        seen: set[str] = set()
        for index, ref in enumerate(data["reading_order"]):
            where = f"/reading_order/{index}"
            if ref not in by_id:
                self.issue(Severity.ERROR, "unknown-reading-order-ref", f"no region {ref!r}", where)
            elif ref in seen:
                self.issue(
                    Severity.ERROR, "duplicate-reading-order-ref", f"region {ref!r} repeats", where
                )
            else:
                seen.add(ref)
                ordered.append(by_id[ref])
        missing = [region for region in regions if region.id not in seen]
        if missing:
            self.issue(
                Severity.WARNING,
                "region-not-in-reading-order",
                f"{len(missing)} region(s) are not in reading_order and are appended in array "
                f"order: {', '.join(region.id for region in missing[:5])}"
                + (", …" if len(missing) > 5 else ""),
                "/reading_order",
            )
        return tuple(ordered + missing), True
