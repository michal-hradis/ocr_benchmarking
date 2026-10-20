"""Versioned JSON dataset manifests and dataset validation.

A manifest pairs each sample image with its PAGE XML ground truth. The format is documented in
docs/DATASETS.md. Ground-truth paths are host-side only: the runner (#7) must never pass a
sample's ``gt`` path or its content to an engine container (ADR 0002).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ocrbench.issues import Issue, Severity
from ocrbench.pagexml import PageGT, read_page_xml

MANIFEST_FORMAT = "ocrbench.dataset-manifest"
MANIFEST_VERSION = 1
UNITS = ("page", "paragraph")
SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_KEYS = {"format", "version", "name", "description", "samples"}
_SAMPLE_KEYS = {
    "id",
    "image",
    "gt",
    "unit",
    "doc_type",
    "language",
    "year",
    "split",
    "image_sha256",
    "gt_sha256",
}
_EXIF_ORIENTATION = 0x0112


@dataclass(frozen=True)
class Sample:
    id: str
    image: Path  # resolved against the manifest directory
    gt: Path  # host-side only; never given to an engine container
    unit: str = "page"
    doc_type: str | None = None
    language: tuple[str, ...] = ()
    year: int | None = None
    split: str | None = None
    image_sha256: str | None = None
    gt_sha256: str | None = None


@dataclass(frozen=True)
class Manifest:
    path: Path
    name: str
    description: str | None
    samples: tuple[Sample, ...]


@dataclass(frozen=True)
class SampleSummary:
    sample_id: str
    unit: str
    width: int
    height: int
    regions: int
    lines: int
    reference_lines: int
    reference_chars: int


@dataclass
class DatasetReport:
    manifest: Manifest | None
    issues: list[Issue] = field(default_factory=list)
    summaries: list[SampleSummary] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity is Severity.ERROR for issue in self.issues)

    def count(self, severity: Severity) -> int:
        return sum(issue.severity is severity for issue in self.issues)


def _error(code: str, message: str, sample_id: str | None = None, location: str | None = None):
    return Issue(Severity.ERROR, code, message, sample_id, location)


def _optional_str(entry: dict[str, Any], key: str, errors: list[Issue], sid: str | None):
    value = entry.get(key)
    if value is not None and not (isinstance(value, str) and value):
        errors.append(_error("invalid-field", f"{key!r} must be a non-empty string", sid))
        return None
    return value


def _parse_sample(entry: Any, position: int, base: Path, errors: list[Issue]) -> Sample | None:
    where = f"samples[{position}]"
    if not isinstance(entry, dict):
        errors.append(_error("invalid-sample", "sample must be a JSON object", location=where))
        return None
    sid = entry.get("id")
    if not isinstance(sid, str) or not SAMPLE_ID_RE.match(sid):
        errors.append(
            _error(
                "invalid-sample-id",
                f"id {sid!r} must match {SAMPLE_ID_RE.pattern}",
                location=where,
            )
        )
        return None
    before = len(errors)
    for key in sorted(set(entry) - _SAMPLE_KEYS):
        errors.append(_error("unknown-field", f"unknown sample field {key!r}", sid))
    paths = {}
    for key in ("image", "gt"):
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            errors.append(_error("invalid-field", f"{key!r} must be a non-empty path", sid))
        else:
            paths[key] = (base / value).resolve()
    unit = entry.get("unit", "page")
    if unit not in UNITS:
        errors.append(_error("invalid-field", f"unit must be one of {UNITS}, not {unit!r}", sid))
    language = entry.get("language", [])
    if not (isinstance(language, list) and all(isinstance(x, str) and x for x in language)):
        errors.append(_error("invalid-field", "'language' must be a list of strings", sid))
        language = []
    year = entry.get("year")
    if year is not None and (isinstance(year, bool) or not isinstance(year, int)):
        errors.append(_error("invalid-field", "'year' must be an integer", sid))
    hashes = {}
    for key in ("image_sha256", "gt_sha256"):
        value = entry.get(key)
        if value is not None and not (isinstance(value, str) and SHA256_RE.match(value)):
            errors.append(_error("invalid-field", f"{key!r} must be 64 lowercase hex digits", sid))
        hashes[key] = value
    doc_type = _optional_str(entry, "doc_type", errors, sid)
    split = _optional_str(entry, "split", errors, sid)
    if len(errors) > before:
        return None
    return Sample(
        id=sid,
        image=paths["image"],
        gt=paths["gt"],
        unit=unit,
        doc_type=doc_type,
        language=tuple(language),
        year=year,
        split=split,
        image_sha256=hashes["image_sha256"],
        gt_sha256=hashes["gt_sha256"],
    )


def load_manifest(path: Path) -> tuple[Manifest | None, list[Issue]]:
    """Parse and check a manifest's structure. File contents are checked by validate_dataset."""
    path = Path(path)
    location = path.name
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        return None, [_error("manifest-unreadable", str(error), location=location)]
    except json.JSONDecodeError as error:
        return None, [
            _error(
                "manifest-json-error",
                f"line {error.lineno}, column {error.colno}: {error.msg}",
                location=location,
            )
        ]
    if not isinstance(data, dict):
        return None, [_error("invalid-manifest", "manifest must be a JSON object", None, location)]
    errors: list[Issue] = []
    for key in sorted(set(data) - _TOP_KEYS):
        errors.append(_error("unknown-field", f"unknown manifest field {key!r}", None, location))
    if data.get("format") != MANIFEST_FORMAT:
        errors.append(
            _error("invalid-manifest", f"'format' must be {MANIFEST_FORMAT!r}", None, location)
        )
    if data.get("version") != MANIFEST_VERSION:
        errors.append(
            _error(
                "unsupported-version",
                f"manifest version {data.get('version')!r} is not supported "
                f"(expected {MANIFEST_VERSION})",
                None,
                location,
            )
        )
    name = data.get("name")
    if not isinstance(name, str) or not name:
        errors.append(
            _error("invalid-manifest", "'name' must be a non-empty string", None, location)
        )
    description = data.get("description")
    if description is not None and not isinstance(description, str):
        errors.append(_error("invalid-manifest", "'description' must be a string", None, location))
    entries = data.get("samples")
    if not isinstance(entries, list) or not entries:
        errors.append(
            _error("invalid-manifest", "'samples' must be a non-empty list", None, location)
        )
        entries = []
    if errors:
        return None, errors

    base = path.resolve().parent
    samples = [_parse_sample(entry, i, base, errors) for i, entry in enumerate(entries)]
    seen_ids: set[str] = set()
    seen_images: dict[Path, str] = {}
    for sample in samples:
        if sample is None:
            continue
        if sample.id in seen_ids:
            errors.append(
                _error("duplicate-sample-id", "sample id is used more than once", sample.id)
            )
        seen_ids.add(sample.id)
        if sample.image in seen_images:
            errors.append(
                _error(
                    "duplicate-image",
                    f"image is also used by sample {seen_images[sample.image]!r}",
                    sample.id,
                )
            )
        seen_images.setdefault(sample.image, sample.id)
    if errors:
        return None, errors
    manifest = Manifest(path, name, description, tuple(s for s in samples if s is not None))
    return manifest, []


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_size(sample: Sample, issues: list[Issue]) -> tuple[int, int] | None:
    try:
        with Image.open(sample.image) as image:  # reads the header only
            orientation = image.getexif().get(_EXIF_ORIENTATION, 1)
            size = image.size
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        issues.append(_error("image-unreadable", str(error), sample.id, sample.image.name))
        return None
    if orientation != 1:
        issues.append(
            Issue(
                Severity.WARNING,
                "exif-orientation",
                f"EXIF orientation is {orientation}; engines may rotate the image while "
                "PAGE coordinates refer to the stored pixels",
                sample.id,
                sample.image.name,
            )
        )
    return size


def validate_sample(sample: Sample) -> tuple[PageGT | None, list[Issue]]:
    """Check one sample's files, image size and ground truth; the page is None on any error."""
    issues: list[Issue] = []
    for key, path, expected in (
        ("image", sample.image, sample.image_sha256),
        ("gt", sample.gt, sample.gt_sha256),
    ):
        if not path.is_file():
            issues.append(_error("missing-file", f"{key} file not found: {path}", sample.id))
        elif expected and sha256_file(path) != expected:
            issues.append(
                _error("checksum-mismatch", f"{key}_sha256 does not match", sample.id, path.name)
            )
    if issues:
        return None, issues

    size = _image_size(sample, issues)
    page, gt_issues = read_page_xml(sample.gt, sample.id)
    issues += gt_issues
    if page is None or size is None:
        return None, issues
    if size != (page.width, page.height):
        issues.append(
            _error(
                "image-size-mismatch",
                f"image is {size[0]}x{size[1]} but PAGE declares {page.width}x{page.height}",
                sample.id,
                sample.gt.name,
            )
        )
        return None, issues
    if any(issue.severity is Severity.ERROR for issue in issues):
        return None, issues
    if page.image_filename and Path(page.image_filename).name != sample.image.name:
        issues.append(
            Issue(
                Severity.INFO,
                "image-filename-differs",
                f"PAGE imageFilename {page.image_filename!r} differs from {sample.image.name!r}",
                sample.id,
                sample.gt.name,
            )
        )
    return page, issues


def validate_dataset(path: Path) -> DatasetReport:
    """Validate a manifest and every sample it lists; collect all findings."""
    manifest, issues = load_manifest(path)
    report = DatasetReport(manifest, issues)
    if manifest is None:
        return report
    for sample in manifest.samples:
        page, sample_issues = validate_sample(sample)
        report.issues += sample_issues
        if page is not None:
            report.summaries.append(
                SampleSummary(
                    sample_id=sample.id,
                    unit=sample.unit,
                    width=page.width,
                    height=page.height,
                    regions=len(page.regions),
                    lines=sum(len(region.lines) for region in page.regions),
                    reference_lines=len(page.reference_lines),
                    reference_chars=sum(len(line.text) for line in page.reference_lines),
                )
            )
    return report
