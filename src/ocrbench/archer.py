"""Import ARCHER OCR reference XML as an ocrbench dataset (#135).

An ARCHER reference file is a ``<dataset>`` of ``<task>`` elements. Each task names one page
image and annotates it with labelled ``<block>`` elements (percentage-based, optionally rotated
boxes and a transcription) and a directed ``<relation from to>`` graph. The importer pairs each
task with its source image, validates it and writes, per sample:

- an **evaluation reference** in PAGE XML, read by the unchanged ``ocrbench.pagexml`` reader;
- a lossless **sidecar** JSON with the original block IDs, labels, raw box attributes, raw text
  and relations, plus a copy of the original reference file.

The scored text is a deterministic, versioned **text view** of the blocks (``TEXT_VIEW``); the
rules are documented in docs/DATASETS.md. This is evaluation on an ARCHER dataset with OCRbench
metrics, not the official ARCHER block-reconstruction scorer. The reference never reaches an OCR
engine: the manifest only pairs images with host-side GT (ADR 0002).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from ocrbench.issues import Issue, Severity
from ocrbench.manifest import MANIFEST_FORMAT, MANIFEST_VERSION, SAMPLE_ID_RE, sha256_file

IMPORT_FORMAT = "ocrbench.archer-import"
SIDECAR_FORMAT = "ocrbench.archer-reference"
IMPORTER_VERSION = 1
TEXT_VIEW = "archer-text-v1"
PAGE_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"

# Labels whose text is transcription, and how each label is represented in PAGE XML.
TEXT_LABELS = ("Text", "Table", "Handwritten_annotations")
PAGE_REGION = {
    "Text": ("TextRegion", "paragraph"),
    "Table": ("TextRegion", "other"),
    "Handwritten_annotations": ("TextRegion", "other"),
    "Image": ("ImageRegion", None),
    "Noise": ("NoiseRegion", None),
}
# Structural span tags removed from the scored text; other parenthesized text such as "(E)" in
# a UN document symbol is ordinary text and stays.
SPAN_TAG_RE = re.compile(r"\((?:/?PR|/?HW)\)")
ILLEGIBLE = "(YYY)"
TABLE_ROW_RE = re.compile(r"^\s*\|(.*)\|\s*$")
TABLE_RULE_RE = re.compile(r"^[\s|:\-]*-[\s|:\-]*$")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")

_KNOWN = {
    "dataset": {"techno", "language"},
    "task": {"id"},
    "image": {"original_width", "original_height"},
    "block": {"id", "label"},
    "bbox": {"x", "y", "width", "height", "rotation"},
    "relation": {"from", "to"},
}


class ArcherImportError(Exception):
    """The import found errors; nothing was written."""

    def __init__(self, issues: list[Issue]) -> None:
        super().__init__(f"{sum(i.severity is Severity.ERROR for i in issues)} error(s)")
        self.issues = issues


@dataclass(frozen=True)
class TextView:
    text: str  # scored text, line breaks kept; empty when nothing is transcribed
    illegible: int  # (YYY) markers removed


def text_view(label: str, raw: str | None) -> TextView:
    """The scored text of one block under ``TEXT_VIEW`` (docs/DATASETS.md)."""
    if label not in TEXT_LABELS or raw is None:
        return TextView("", 0)
    text = SPAN_TAG_RE.sub("", raw)
    illegible = text.count(ILLEGIBLE)
    text = text.replace(ILLEGIBLE, "")
    if label == "Table":
        text = "\n".join(_table_line(line) for line in text.split("\n"))
    return TextView(text if text.strip() else "", illegible)


def _table_line(line: str) -> str:
    """A pipe row becomes its non-empty cells joined by spaces; a ``|---|`` rule is dropped."""
    row = TABLE_ROW_RE.match(line)
    if row is None:
        return line
    if TABLE_RULE_RE.match(line):
        return ""
    cells = re.split(r"(?<!\\)\|", row.group(1))
    return " ".join(c.strip().replace("\\|", "|") for c in cells if c.strip())


def has_geometry(bbox: dict[str, float]) -> bool:
    """False for ARCHER's empty box (zero width or height), which carries no position."""
    return bbox["width"] > 0 and bbox["height"] > 0


def rotated_box(bbox: dict[str, float], width: int, height: int) -> list[tuple[int, int]]:
    """Pixel corners of a percentage box rotated clockwise (degrees) about its top-left corner."""
    x0, y0 = bbox["x"] * width / 100, bbox["y"] * height / 100
    w, h = bbox["width"] * width / 100, bbox["height"] * height / 100
    angle = math.radians(bbox["rotation"])
    cos, sin = math.cos(angle), math.sin(angle)
    corners = ((0.0, 0.0), (w, 0.0), (w, h), (0.0, h))
    return [
        (round(x0 + dx * cos - dy * sin), round(y0 + dx * sin + dy * cos)) for dx, dy in corners
    ]


@dataclass
class _Block:
    id: str
    label: str
    bbox_raw: dict[str, str]
    bbox: dict[str, float]
    text: str | None
    view: TextView = field(default_factory=lambda: TextView("", 0))


@dataclass
class _Task:
    source: Path
    index: int  # position of the task in its reference file
    id: str
    dataset_attrib: dict[str, str]
    meta: list[tuple[str, str]]
    image_name: str | None
    source_name: str | None
    width: int
    height: int
    blocks: list[_Block]
    relations: list[tuple[str, str]]
    image: Path | None = None


class _Importer:
    def __init__(self, src: Path, ref: Path) -> None:
        self.src, self.ref = src, ref
        self.issues: list[Issue] = []

    def issue(self, severity: Severity, code: str, message: str, sample: str | None, where: str):
        self.issues.append(Issue(severity, code, message, sample, where))

    def errors(self) -> int:
        return sum(i.severity is Severity.ERROR for i in self.issues)

    # Reading -----------------------------------------------------------------------------

    def read_all(self) -> list[_Task]:
        for directory, what in ((self.src, "image"), (self.ref, "reference")):
            if not directory.is_dir():
                self.issue(
                    Severity.ERROR, "missing-dir", f"no {what} directory", None, str(directory)
                )
        if self.errors():
            return []
        files = sorted(self.ref.glob("*.xml"))
        if not files:
            self.issue(
                Severity.ERROR, "no-reference", "no *.xml reference files", None, str(self.ref)
            )
        tasks = [task for path in files for task in self.read_file(path)]
        self.check_task_ids(tasks)
        self.pair_images(tasks)
        return tasks

    def read_file(self, path: Path) -> list[_Task]:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            line, column = error.position
            self.issue(
                Severity.ERROR,
                "xml-parse-error",
                f"line {line}, column {column}: {error}",
                None,
                path.name,
            )
            return []
        if root.tag != "dataset":
            self.issue(
                Severity.ERROR,
                "not-archer-xml",
                f"root element {root.tag!r} is not <dataset>",
                None,
                path.name,
            )
            return []
        self.check_attributes(root, path.name, None)
        tasks = []
        for index, element in enumerate(root):
            if element.tag != "task":
                self.unknown(element, path.name, None)
                continue
            task = self.read_task(path, index, root, element)
            if task is not None:
                tasks.append(task)
        if not any(element.tag == "task" for element in root):
            self.issue(Severity.ERROR, "no-task", "no <task> elements", None, path.name)
        return tasks

    def read_task(self, path: Path, index: int, root: ET.Element, task: ET.Element) -> _Task | None:
        task_id = task.get("id")
        where = f"{path.name}#task[{index}]"
        if not task_id:
            self.issue(Severity.ERROR, "missing-task-id", "task has no id", None, where)
            return None
        where = f"{path.name}#{task_id}"
        if not SAMPLE_ID_RE.match(task_id):
            self.issue(
                Severity.ERROR,
                "invalid-task-id",
                "task id is not a valid sample ID ([A-Za-z0-9][A-Za-z0-9._-]*, at most 128)",
                task_id,
                where,
            )
        self.check_attributes(task, where, task_id)
        meta_element, data, annotations = (task.find(n) for n in ("meta", "data", "annotations"))
        for child in task:
            if child.tag not in {"meta", "data", "annotations"}:
                self.unknown(child, where, task_id)
        meta = [(c.tag, c.text or "") for c in meta_element] if meta_element is not None else []
        source_name = meta_element.findtext("source_name") if meta_element is not None else None
        image = data.find("image") if data is not None else None
        if image is None or not (image.text or "").strip():
            self.issue(
                Severity.ERROR, "missing-image-ref", "task has no data/image", task_id, where
            )
            return None
        self.check_attributes(image, where, task_id)
        width, height = (
            self.dimension(image, name, task_id, where)
            for name in ("original_width", "original_height")
        )
        blocks, relations = self.read_annotations(annotations, task_id, where)
        if width is None or height is None:
            return None
        return _Task(
            source=path,
            index=index,
            id=task_id,
            dataset_attrib=dict(root.attrib),
            meta=meta,
            image_name=(image.text or "").strip(),
            source_name=source_name.strip() if source_name else None,
            width=width,
            height=height,
            blocks=blocks,
            relations=relations,
        )

    def dimension(self, image: ET.Element, name: str, task_id: str, where: str) -> int | None:
        value = image.get(name)
        try:
            number = int(value or "")
        except ValueError:
            number = 0
        if number <= 0:
            self.issue(
                Severity.ERROR,
                "invalid-dimensions",
                f"data/image/@{name} must be a positive integer, got {value!r}",
                task_id,
                where,
            )
            return None
        return number

    def read_annotations(
        self, annotations: ET.Element | None, task_id: str, where: str
    ) -> tuple[list[_Block], list[tuple[str, str]]]:
        blocks: list[_Block] = []
        relations: list[tuple[str, str]] = []
        if annotations is None:
            self.issue(
                Severity.WARNING, "no-annotations", "task has no <annotations>", task_id, where
            )
            return blocks, relations
        for element in annotations:
            if element.tag == "block":
                block = self.read_block(element, task_id, where)
                if block is not None:
                    blocks.append(block)
            elif element.tag == "relation":
                self.check_attributes(element, where, task_id)
                source, target = element.get("from"), element.get("to")
                if not source or not target:
                    self.issue(
                        Severity.ERROR,
                        "malformed-relation",
                        "relation needs from and to",
                        task_id,
                        where,
                    )
                    continue
                relations.append((source, target))
            else:
                self.unknown(element, where, task_id)
        ids = [b.id for b in blocks]
        for block_id in sorted({i for i in ids if ids.count(i) > 1}):
            self.issue(
                Severity.ERROR,
                "duplicate-block-id",
                f"{ids.count(block_id)} blocks share this id",
                task_id,
                f"{where}/{block_id}",
            )
        known = set(ids)
        seen: set[tuple[str, str]] = set()
        for source, target in relations:
            for end in (source, target):
                if end not in known:
                    self.issue(
                        Severity.ERROR,
                        "unknown-relation-block",
                        f"relation {source} -> {target} names no block {end!r}",
                        task_id,
                        where,
                    )
            if source == target:
                self.issue(
                    Severity.WARNING,
                    "self-relation",
                    f"relation {source} -> itself",
                    task_id,
                    where,
                )
            if (source, target) in seen:
                self.issue(
                    Severity.WARNING,
                    "duplicate-relation",
                    f"relation {source} -> {target} repeats",
                    task_id,
                    where,
                )
            seen.add((source, target))
        regions = [_region_id(i) for i in ids]
        for region_id in sorted(
            {r for r in regions if regions.count(r) > 1}
            - {_region_id(i) for i in ids if ids.count(i) > 1}
        ):
            self.issue(
                Severity.ERROR,
                "region-id-collision",
                f"block ids map to the same PAGE id {region_id}",
                task_id,
                where,
            )
        if not any(b.view.text for b in blocks):
            self.issue(
                Severity.WARNING, "no-reference-text", "task has no scored text", task_id, where
            )
        return blocks, relations

    def read_block(self, element: ET.Element, task_id: str, where: str) -> _Block | None:
        self.check_attributes(element, where, task_id)
        block_id, label = element.get("id"), element.get("label")
        if not block_id:
            self.issue(Severity.ERROR, "missing-block-id", "block has no id", task_id, where)
            return None
        here = f"{where}/{block_id}"
        if label not in PAGE_REGION:
            self.issue(
                Severity.ERROR,
                "unknown-label",
                f"label {label!r} is not one of {', '.join(PAGE_REGION)}",
                task_id,
                here,
            )
            return None
        bbox_element = element.find("bbox")
        bbox_raw = dict(bbox_element.attrib) if bbox_element is not None else {}
        bbox: dict[str, float] = {}
        for name in ("x", "y", "width", "height", "rotation"):
            try:
                bbox[name] = float(bbox_raw.get(name, "0" if name == "rotation" else ""))
            except ValueError:
                bbox[name] = math.nan
        if bbox_element is None or not all(math.isfinite(v) for v in bbox.values()):
            self.issue(
                Severity.ERROR,
                "malformed-bbox",
                f"block needs a numeric bbox, got {bbox_raw}",
                task_id,
                here,
            )
            return None
        self.check_attributes(bbox_element, here, task_id)
        if bbox["width"] < 0 or bbox["height"] < 0:
            self.issue(
                Severity.ERROR,
                "malformed-bbox",
                "bbox width and height must not be negative",
                task_id,
                here,
            )
            return None
        if not has_geometry(bbox):
            # ARCHER writes x=y=width=height=0 for a block without a box; its text still counts
            degenerate = bbox["x"] or bbox["y"] or bbox["width"] or bbox["height"]
            self.issue(
                Severity.WARNING if degenerate else Severity.INFO,
                "no-geometry",
                f"{label} block has an empty bbox {bbox_raw}; no coordinates are written",
                task_id,
                here,
            )
        elif (
            not (0 <= bbox["x"] <= 100 and 0 <= bbox["y"] <= 100)
            or bbox["x"] + bbox["width"] > 100.5
            or bbox["y"] + bbox["height"] > 100.5
        ):
            self.issue(
                Severity.INFO,
                "bbox-out-of-bounds",
                "bbox extends outside the image; kept",
                task_id,
                here,
            )
        for child in element:
            if child.tag not in {"bbox", "text"}:
                self.unknown(child, here, task_id)
        text_element = element.find("text")
        text = None if text_element is None else (text_element.text or "")
        block = _Block(block_id, label, bbox_raw, bbox, text, text_view(label, text))
        if text is not None and label not in TEXT_LABELS and text.strip():
            self.issue(
                Severity.WARNING,
                "text-not-scored",
                f"{label} block has text; the text view does not score it",
                task_id,
                here,
            )
        elif label in {"Text", "Table"} and not block.view.text:
            self.issue(
                Severity.INFO, "empty-text", f"{label} block has no scored text", task_id, here
            )
        if block.view.illegible:
            self.issue(
                Severity.INFO,
                "illegible-removed",
                f"{block.view.illegible} (YYY) marker(s) removed from the scored text",
                task_id,
                here,
            )
        return block

    def check_attributes(self, element: ET.Element, where: str, task_id: str | None) -> None:
        extra = sorted(set(element.attrib) - _KNOWN.get(element.tag, set()))
        if extra:
            self.issue(
                Severity.WARNING,
                "unknown-attribute",
                f"<{element.tag}> attribute(s) {', '.join(extra)} kept only in the original copy",
                task_id,
                where,
            )

    def unknown(self, element: ET.Element, where: str, task_id: str | None) -> None:
        self.issue(
            Severity.WARNING,
            "unknown-element",
            f"<{element.tag}> is not interpreted; kept only in the original copy",
            task_id,
            where,
        )

    # Pairing ------------------------------------------------------------------------------

    def check_task_ids(self, tasks: list[_Task]) -> None:
        by_id: dict[str, list[_Task]] = {}
        for task in tasks:
            by_id.setdefault(task.id, []).append(task)
        for task_id, same in by_id.items():
            if len(same) > 1:
                places = ", ".join(f"{t.source.name}#{t.index}" for t in same)
                self.issue(
                    Severity.ERROR,
                    "duplicate-task-id",
                    f"task id used by {places}",
                    task_id,
                    same[0].source.name,
                )

    def pair_images(self, tasks: list[_Task]) -> None:
        by_stem: dict[str, list[Path]] = {}
        for path in sorted(self.src.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                by_stem.setdefault(path.stem, []).append(path)
        used: dict[Path, str] = {}
        for task in tasks:
            where = f"{task.source.name}#{task.id}"
            names = [n for n in (task.image_name, task.source_name) if n]
            if task.source_name and task.image_name != task.source_name:
                self.issue(
                    Severity.WARNING,
                    "source-name-differs",
                    f"meta/source_name {task.source_name!r} differs from data/image "
                    f"{task.image_name!r}",
                    task.id,
                    where,
                )
            exact = sorted({self.src / n for n in names if (self.src / n).is_file()})
            if len(exact) > 1:
                self.issue(
                    Severity.ERROR,
                    "ambiguous-image",
                    "data/image and meta/source_name name different files: "
                    f"{[p.name for p in exact]}",
                    task.id,
                    where,
                )
                continue
            if not exact:
                stems = {Path(n).stem for n in names} | {task.id}
                near = sorted(p.name for s in stems for p in by_stem.get(s, []))
                hint = (
                    f"; same stem, other suffix: {near} (rename or fix the reference)"
                    if near
                    else ""
                )
                self.issue(
                    Severity.ERROR,
                    "missing-image",
                    f"no image {names[0]!r} in {self.src}{hint}",
                    task.id,
                    where,
                )
                continue
            image = exact[0]
            if image in used:
                self.issue(
                    Severity.ERROR,
                    "image-reused",
                    f"{image.name} is also used by task {used[image]}",
                    task.id,
                    where,
                )
                continue
            used[image] = task.id
            try:
                with Image.open(image) as opened:
                    size = opened.size
            except (OSError, UnidentifiedImageError) as error:
                self.issue(Severity.ERROR, "image-unreadable", str(error), task.id, image.name)
                continue
            if size != (task.width, task.height):
                self.issue(
                    Severity.ERROR,
                    "image-size-mismatch",
                    f"image is {size[0]}x{size[1]}, reference declares {task.width}x{task.height}",
                    task.id,
                    where,
                )
                continue
            task.image = image


# Writing ----------------------------------------------------------------------------------


def _region_id(block_id: str) -> str:
    """A PAGE (NCName) id for an ARCHER block id, which may start with a digit or ``-``."""
    return "b_" + re.sub(r"[^A-Za-z0-9._-]", "_", block_id)


def _page_xml(task: _Task, ref_sha256: str) -> bytes:
    ET.register_namespace("", PAGE_NAMESPACE)
    ns = f"{{{PAGE_NAMESPACE}}}"
    root = ET.Element(f"{ns}PcGts")
    metadata = ET.SubElement(root, f"{ns}Metadata")
    ET.SubElement(metadata, f"{ns}Creator").text = f"ocrbench ARCHER import v{IMPORTER_VERSION}"
    stamp = dict(task.meta).get("annotated_at") or "1970-01-01T00:00:00Z"
    ET.SubElement(metadata, f"{ns}Created").text = stamp
    ET.SubElement(metadata, f"{ns}LastChange").text = stamp
    ET.SubElement(metadata, f"{ns}Comments").text = (
        f"Evaluation reference generated from ARCHER task {task.id} in {task.source.name} "
        f"(sha256 {ref_sha256}) with text view {TEXT_VIEW}. Regions follow the annotation "
        "order; the original blocks and relations are in the archer/ sidecar."
    )
    page = ET.SubElement(
        root,
        f"{ns}Page",
        imageFilename=task.image_name or "",
        imageWidth=str(task.width),
        imageHeight=str(task.height),
    )
    if task.relations:
        relations = ET.SubElement(page, f"{ns}Relations")
        for source, target in task.relations:
            relation = ET.SubElement(
                relations, f"{ns}Relation", type="link", custom="archer-relation"
            )
            ET.SubElement(relation, f"{ns}SourceRegionRef", regionRef=_region_id(source))
            ET.SubElement(relation, f"{ns}TargetRegionRef", regionRef=_region_id(target))
    for block in task.blocks:
        tag, kind = PAGE_REGION[block.label]
        attrib = {
            "id": _region_id(block.id),
            "custom": f"archer {{id:{block.id}; label:{block.label};}}",
        }
        if kind:
            attrib["type"] = kind
        if block.bbox["rotation"]:
            # PAGE: clockwise rotation that corrects the skew, in (-180, 180]
            attrib["orientation"] = f"{-((block.bbox['rotation'] + 180) % 360 - 180):g}"
        region = ET.SubElement(page, f"{ns}{tag}", attrib)
        if has_geometry(block.bbox):  # never invent coordinates for a block without a box
            corners = rotated_box(block.bbox, task.width, task.height)
            ET.SubElement(region, f"{ns}Coords", points=" ".join(f"{x},{y}" for x, y in corners))
        if tag == "TextRegion" and block.view.text:
            equiv = ET.SubElement(region, f"{ns}TextEquiv")
            ET.SubElement(equiv, f"{ns}Unicode").text = block.view.text
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def _sidecar(task: _Task, ref_file: str, ref_sha256: str, page_file: str) -> dict[str, Any]:
    return {
        "format": SIDECAR_FORMAT,
        "version": IMPORTER_VERSION,
        "text_view": TEXT_VIEW,
        "source": {"file": ref_file, "sha256": ref_sha256, "task_index": task.index},
        "dataset": task.dataset_attrib,
        "task": {"id": task.id, "meta": [list(item) for item in task.meta]},
        "image": {
            "name": task.image_name,
            "source_name": task.source_name,
            "original_width": task.width,
            "original_height": task.height,
        },
        "page_xml": page_file,
        "blocks": [
            {
                "id": b.id,
                "label": b.label,
                "bbox": b.bbox_raw,
                "text": b.text,
                "scored_text": b.view.text,
                "illegible_removed": b.view.illegible,
                "page_region_id": _region_id(b.id),
                "polygon": [list(p) for p in rotated_box(b.bbox, task.width, task.height)]
                if has_geometry(b.bbox)
                else None,
            }
            for b in task.blocks
        ],
        "relations": [{"from": s, "to": t} for s, t in task.relations],
    }


def _dump(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _relative(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def import_archer(
    src: Path, ref: Path, output: Path, name: str | None = None, force: bool = False
) -> tuple[Path, list[Issue]]:
    """Validate ARCHER references in ``ref`` against images in ``src`` and write the dataset.

    Writes ``output/manifest.json``, ``page/<id>.xml``, ``archer/<id>.json``,
    ``archer/ref/<file>`` and ``import.json``, deterministically (no timestamps). Raises
    ``ArcherImportError`` without writing anything if any error is found.
    """
    importer = _Importer(Path(src), Path(ref))
    tasks = importer.read_all()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if not (output / "import.json").is_file():
            importer.issue(
                Severity.ERROR,
                "output-not-empty",
                "output directory exists and was not written by the ARCHER importer",
                None,
                str(output),
            )
        elif not force:
            importer.issue(
                Severity.ERROR,
                "output-exists",
                "output exists; use --force to replace it",
                None,
                str(output),
            )
    if importer.errors():
        raise ArcherImportError(importer.issues)

    languages = sorted({t.dataset_attrib.get("language", "") for t in tasks} - {""})
    name = name or "archer-" + ("-".join(languages).lower() or "x") + "-" + TEXT_VIEW.removeprefix(
        "archer-"
    )
    if output.exists():
        shutil.rmtree(output)
    (output / "page").mkdir(parents=True)
    (output / "archer" / "ref").mkdir(parents=True)
    ref_hashes: dict[str, str] = {}
    for path in sorted({t.source for t in tasks}):
        shutil.copyfile(path, output / "archer" / "ref" / path.name)
        ref_hashes[path.name] = sha256_file(path)
    samples = []
    for task in sorted(tasks, key=lambda t: t.id):
        assert task.image is not None
        page_file = f"page/{task.id}.xml"
        (output / page_file).write_bytes(_page_xml(task, ref_hashes[task.source.name]))
        sidecar = _sidecar(
            task, f"archer/ref/{task.source.name}", ref_hashes[task.source.name], page_file
        )
        (output / "archer" / f"{task.id}.json").write_bytes(_dump(sidecar))
        sample: dict[str, Any] = {
            "id": task.id,
            "image": _relative(task.image, output),
            "gt": page_file,
            "unit": "page",
        }
        language = task.dataset_attrib.get("language")
        if language:
            sample["language"] = [language]
        published = dict(task.meta).get("published_at", "")
        if re.match(r"^\d{4}-", published):
            sample["year"] = int(published[:4])
        sample["image_sha256"] = sha256_file(task.image)
        sample["gt_sha256"] = hashlib.sha256((output / page_file).read_bytes()).hexdigest()
        samples.append(sample)
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": MANIFEST_VERSION,
        "name": name,
        "description": (
            f"ARCHER references imported by ocrbench ARCHER import v{IMPORTER_VERSION}, text view "
            f"{TEXT_VIEW}; OCRbench metrics, not official ARCHER scores. See import.json."
        ),
        "samples": samples,
    }
    (output / "manifest.json").write_bytes(_dump(manifest))
    record = {
        "format": IMPORT_FORMAT,
        "version": IMPORTER_VERSION,
        "text_view": TEXT_VIEW,
        "references": [{"file": f"archer/ref/{n}", "sha256": h} for n, h in ref_hashes.items()],
        "samples": len(samples),
        "findings": [
            {
                "severity": i.severity.value,
                "code": i.code,
                "sample": i.sample_id,
                "where": i.location,
                "message": i.message,
            }
            for i in importer.issues
        ],
    }
    (output / "import.json").write_bytes(_dump(record))
    return output / "manifest.json", importer.issues
