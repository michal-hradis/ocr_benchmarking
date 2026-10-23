"""OCR-D pipeline with Calamari recognition; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json and run a fixed
OCR-D pipeline on every sample image in job order, in this process: ocrd-cis-ocropy-binarize,
ocrd-tesserocr-segment (regions, lines and reading order) and ocrd-calamari-recognize with one of
the pinned Calamari model ensembles (parameter model). Each sample gets its own temporary METS
workspace holding only its image; the processors are set up once and reused. It writes
$OCRBENCH_OUTPUT/<id>/ with the final PAGE XML (page.xml), the standardized prediction.json built
from it and, last, result.json. Regions, lines, line confidences and the reading order are the
pipeline's, in original-image pixels.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

MODELS_DIR = Path(os.environ.get("OCRBENCH_OCRD_CALAMARI_MODELS", "/opt/ocrbench/models"))
# SHA-256 of the model archives and of every file unpacked from them (relative to MODELS_DIR). They
# sit next to the adapter and cannot be overridden: they define the official ensembles.
CHECKSUMS = Path(__file__).with_name("models.sha256")
FILE_CHECKSUMS = Path(__file__).with_name("models.files.sha256")
WORK_DIR = Path(os.environ.get("OCRBENCH_OCRD_CALAMARI_WORK", "/tmp/ocrd-calamari"))
# Parameter model: a calamari_models 2.2 ensemble, the directory of its *.ckpt.json files.
MODELS = ("gt4histocr", "fraktur_19th_century", "antiqua_historical")
DEFAULT_PARAMS: dict[str, Any] = {"model": "gt4histocr"}
# File groups of the workspace: the image, then the output of each processor in STEPS.
GROUPS = ("OCR-D-IMG", "OCR-D-BIN", "OCR-D-SEG", "OCR-D-OCR")
STEPS = ("binarize", "segment", "recognize")
PAGE_ID = "PHYS_0001"
# Resolution assumed by the binarization for every image, instead of the image's DPI metadata: a
# declared DPI below the real one shrinks ocropy's background window and hollows out thick strokes
# (72 DPI JPEGs). 300 is ocropy's reference, also used for images without metadata.
BINARIZE_DPI = 300
# Distributions whose versions are recorded with every sample.
PACKAGES = (
    "ocrd",
    "ocrd-cis",
    "ocrd-tesserocr",
    "ocrd-calamari",
    "calamari-ocr",
    "tensorflow",
    "tesserocr",
)
# PAGE ReadingOrder elements (groups may nest and may themselves reference a region).
GROUPS_RO = {"OrderedGroup", "UnorderedGroup", "OrderedGroupIndexed", "UnorderedGroupIndexed"}
REFS = {"RegionRef", "RegionRefIndexed"}
# PAGE TextRegion type -> standardized region type (docs/PREDICTIONS.md#region-types). Tesseract
# writes paragraph, heading, caption and floating (pull-out text); table cells have no type.
REGION_TYPES = {"paragraph": "text", "heading": "heading", "caption": "caption"}


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {
        **DEFAULT_PARAMS,
        **{key: value for key, value in params.items() if value is not None},
    }
    if merged["model"] not in MODELS:
        raise AdapterError(f"model must be one of {', '.join(MODELS)}: {merged['model']!r}")
    return merged


def _checksums(path: Path) -> dict[str, str]:
    """``sha256sum`` output: file name -> SHA-256."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return {name: digest for digest, name in (line.split(maxsplit=1) for line in lines)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_info(name: str) -> dict[str, Any]:
    """The model directory, its checkpoints and the SHA-256 of the archive it came from.

    The directory must hold exactly the files unpacked from that archive, each with the SHA-256 in
    models.files.sha256, so the recorded archive describes the checkpoints actually loaded, also
    from a directory set by OCRBENCH_OCRD_CALAMARI_MODELS.
    """
    directory = MODELS_DIR / name
    archive = _checksums(CHECKSUMS).get(f"{name}.tar.gz")
    expected = {
        path: digest
        for path, digest in _checksums(FILE_CHECKSUMS).items()
        if path.startswith(f"{name}/")
    }
    if archive is None or not expected:
        raise AdapterError(f"no checksums for Calamari model {name}")
    if not directory.is_dir():
        raise AdapterError(f"no Calamari model directory {directory}")
    found = {p.relative_to(MODELS_DIR).as_posix() for p in directory.rglob("*") if not p.is_dir()}
    problems = [f"missing {path}" for path in sorted(expected.keys() - found)]
    problems += [f"unexpected {path}" for path in sorted(found - expected.keys())]
    problems += [
        f"modified {path}"
        for path in sorted(expected.keys() & found)
        if _sha256(MODELS_DIR / path) != expected[path]
    ]
    if problems:
        raise AdapterError(
            f"Calamari model {name} in {directory} differs from {FILE_CHECKSUMS.name}: "
            + ", ".join(problems)
        )
    return {
        "name": name,
        "path": str(directory),
        "checkpoints": sorted(p.name for p in directory.glob("*.ckpt.json")),
        "archive_sha256": archive,
        "files_verified": len(expected),
    }


def environment() -> dict[str, Any]:
    """Versions of the pipeline's packages, Tesseract and the ocrd_calamari source commit."""
    info: dict[str, Any] = {}
    for name in PACKAGES:
        try:
            info[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info[name] = None
    info["ocrd_calamari_commit"] = os.environ.get("OCRBENCH_OCRD_CALAMARI_COMMIT")
    try:
        import tesserocr

        info["tesseract"] = tesserocr.tesseract_version().split()[1]
    except Exception as error:
        raise AdapterError(f"cannot load Tesseract: {type(error).__name__}: {error}") from error
    info["device"] = "cpu"
    return info


def load_pipeline(params: dict[str, Any]) -> list[tuple[type, dict[str, Any]]]:
    """The OCR-D processor classes of STEPS with their parameters (others: OCR-D defaults).

    OCR-D's own logging is set up first: its processors depend on it.
    """
    try:
        from ocrd_utils import initLogging

        initLogging()
        from ocrd_calamari.recognize import CalamariRecognize
        from ocrd_cis.ocropy.binarize import OcropyBinarize
        from ocrd_tesserocr.segment import TesserocrSegment
    except Exception as error:
        raise AdapterError(f"cannot import OCR-D: {type(error).__name__}: {error}") from error
    checkpoints = str(MODELS_DIR / params["model"])
    return [
        (OcropyBinarize, {"dpi": BINARIZE_DPI}),
        (TesserocrSegment, {}),
        (CalamariRecognize, {"checkpoint_dir": checkpoints, "device": -1}),
    ]


def run_pipeline(pipeline: list, image: Path, workdir: Path) -> tuple[str, dict[str, float]]:
    """The final PAGE XML for one image, and the seconds each step took.

    The image is copied into a new METS workspace in ``workdir`` as the only page. The processor
    instances are cached by OCR-D across samples (``instance_caching``), so each model is loaded
    once per container.
    """
    from ocrd import Resolver, run_processor
    from ocrd_utils import EXT_TO_MIME

    mimetype = EXT_TO_MIME.get(image.suffix.lower())
    if mimetype is None or not mimetype.startswith("image/"):
        raise AdapterError(f"unsupported image type {image.suffix!r}")
    workspace = Resolver().workspace_from_nothing(directory=str(workdir))
    local = image.name
    shutil.copyfile(image, workdir / local)
    workspace.add_file(
        GROUPS[0], file_id="IMG", mimetype=mimetype, page_id=PAGE_ID, local_filename=local
    )
    workspace.save_mets()
    seconds = {}
    for step, (processor, parameter), source, target in zip(
        STEPS, pipeline, GROUPS[:-1], GROUPS[1:], strict=True
    ):
        start = time.monotonic()
        run_processor(
            processor,
            workspace=workspace,
            input_file_grp=source,
            output_file_grp=target,
            parameter=parameter,
            instance_caching=True,
        )
        seconds[step] = round(time.monotonic() - start, 3)
    files = list(workspace.find_files(file_grp=GROUPS[-1], page_id=PAGE_ID))
    if len(files) != 1:
        raise AdapterError(f"expected one PAGE file in {GROUPS[-1]}, found {len(files)}")
    return (workdir / files[0].local_filename).read_text(encoding="utf-8"), seconds


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _points(element: ET.Element | None) -> list[list[float]] | None:
    text = element.get("points", "") if element is not None else ""
    try:
        points = [[float(v) for v in point.split(",")] for point in text.split()]
    except ValueError:
        return None
    if len(points) >= 3 and all(len(point) == 2 for point in points):
        return points
    return None


def _reading_order(element: ET.Element) -> Iterator[str]:
    """Region references of a PAGE ReadingOrder (sub)tree in reading order.

    The members of an ordered group follow their integer ``@index`` (document order when any is
    missing), nested groups are read in place, and an unordered group keeps document order, as
    in the benchmark's PAGE reader (src/ocrbench/pagexml.py). Tesseract's tables are a group
    that references the TableRegion and, inside it, its cells.
    """
    if _local(element) in GROUPS_RO and element.get("regionRef"):
        yield element.get("regionRef")
    members = [m for m in element if _local(m) in GROUPS_RO | REFS]
    if _local(element).startswith("Ordered"):
        keys = [m.get("index") for m in members]
        if all(k is not None and k.isdigit() for k in keys):
            members = [m for _, m in sorted(zip(map(int, keys), members, strict=True))]
    for member in members:
        if _local(member) in REFS:
            yield member.get("regionRef", "")
        else:
            yield from _reading_order(member)


def region_type(page_type: str | None) -> str:
    return "text" if page_type is None else REGION_TYPES.get(page_type, "other")


def page_xml_to_prediction(xml: str, sample_id: str) -> dict[str, Any]:
    """Standardized prediction (docs/PREDICTIONS.md) from the pipeline's PAGE XML (``page.xml``).

    All TextRegions, including table cells nested in a TableRegion, in the order of the PAGE
    ReadingOrder (by ``@index``); regions it does not reference follow in document order. Lines
    keep the file order within their region. Lines with non-blank text keep their polygon and
    the line confidence; regions without such lines stay only in the native output. The image
    size is the one OCR-D read from the image.
    """
    root = ET.fromstring(xml)
    ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    page = root.find(f"{ns}Page")
    if page is None:
        raise AdapterError("PAGE XML without a Page element")
    try:
        size = int(page.get("imageWidth", "")), int(page.get("imageHeight", ""))
    except ValueError as error:
        raise AdapterError("PAGE XML without a valid image size") from error
    parents = {child: parent for parent in page.iter() for child in parent}
    native_regions = list(page.iter(f"{ns}TextRegion"))
    reading_order = page.find(f"{ns}ReadingOrder")
    order = list(_reading_order(reading_order)) if reading_order is not None else []
    rank: dict[str, int] = {}
    for ref in order:
        rank.setdefault(ref, len(rank))
    native_regions.sort(key=lambda r: rank.get(r.get("id"), len(rank)))  # stable: file order
    regions: list[dict[str, Any]] = []
    count = 0
    for native in native_regions:
        lines = []
        for native_line in native.findall(f"{ns}TextLine"):
            equiv = native_line.find(f"{ns}TextEquiv")
            text = equiv.findtext(f"{ns}Unicode") if equiv is not None else None
            if not text or not text.strip():
                continue
            count += 1
            entry: dict[str, Any] = {"id": f"l{count}", "text": text}
            if polygon := _points(native_line.find(f"{ns}Coords")):
                entry["polygon"] = polygon
            if equiv.get("conf") is not None:
                entry["confidence"] = float(equiv.get("conf"))
            entry["extra"] = {"ocrd_id": native_line.get("id")}
            lines.append(entry)
        if not lines:
            continue
        region: dict[str, Any] = {"id": f"r{len(regions) + 1}"}
        page_type = native.get("type")
        region["type"] = region_type(page_type)
        if page_type is not None:
            region["native_type"] = page_type
        if polygon := _points(native.find(f"{ns}Coords")):
            region["polygon"] = polygon
        region["lines"] = lines
        region["extra"] = {"ocrd_id": native.get("id")}
        parent = parents.get(native)
        if parent is not None and parent is not page:
            region["extra"]["parent"] = {"type": _local(parent), "ocrd_id": parent.get("id")}
        regions.append(region)
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": size[0], "height": size[1]},
        "text": "\n\n".join("\n".join(e["text"] for e in r["lines"]) for r in regions),
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    return prediction


def process(pipeline: list, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Run the pipeline on one sample, write its outputs and return ``result.json``."""
    workdir = WORK_DIR / sample["id"]
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    try:
        xml, seconds = run_pipeline(pipeline, source / sample["image"], workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    (out / "page.xml").write_text(xml, encoding="utf-8")
    prediction = page_xml_to_prediction(xml, sample["id"])
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    regions = prediction.get("regions", [])
    extra = {
        "regions": len(regions),
        "lines": sum(len(region["lines"]) for region in regions),
        "step_seconds": seconds,
    }
    return {"status": "ok", "primary": "prediction.json", "extra": extra}


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    pipeline, params, info, model, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment()
        model = model_info(params["model"])
        pipeline = load_pipeline(params)
    except AdapterError as error:
        init_error = str(error)
    models = {"calamari": model, "job_model": job.get("model")}
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(pipeline, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # a processor failed on this page: fail it, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=models, params=params)
        write_result(out, result)
    print(f"ocrd-calamari adapter processed {len(job['samples'])} samples", file=sys.stderr)


# Calamari predicts in a subprocess started with multiprocessing's spawn method, which imports
# this module again: everything must stay behind this guard.
if __name__ == "__main__":
    main()
