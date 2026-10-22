"""PERO OCR offline page pipeline adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, check the
user-supplied PERO model bundles in the /cache volume against bundles.sha256, build one pero-ocr
PageParser from each bundle's unmodified config.ini (layout analysis, then line cropping and OCR)
and run both on every sample image in job order. It writes $OCRBENCH_OUTPUT/<id>/ with PERO's own
PAGE XML (page.xml) and ALTO (alto.xml), the standardized prediction.json built from page.xml and,
last, result.json. Regions, lines, baselines and the region order are PERO's, in image pixels.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path
from typing import Any

# The bundles are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_PERO_CACHE", "/cache"))
BUNDLES_DIR = "models/pero"  # relative to the cache: <cache>/models/pero/<bundle>/config.ini
CHECKSUMS = Path(os.environ.get("OCRBENCH_PERO_CHECKSUMS", "/opt/ocrbench/bundles.sha256"))
CONFIG = "config.ini"  # the entry point of every bundle
# Engine parameters (job.json "params"), README.md. The default bundles form the universal preset.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "layout_bundle": "layout_model",
    "ocr_bundle": "universal.transformer.2025-03-19",
}
# The pipeline stages ([PAGE_PARSER] RUN_* switches) each bundle must enable, and must not.
STAGES = {
    "layout_bundle": ({"RUN_LAYOUT_PARSER"}, {"RUN_LINE_CROPPER", "RUN_OCR", "RUN_DECODER"}),
    "ocr_bundle": ({"RUN_LINE_CROPPER", "RUN_OCR"}, {"RUN_LAYOUT_PARSER", "RUN_DECODER"}),
}
# PAGE ReadingOrder elements (groups may nest and may themselves reference a region).
GROUPS = {"OrderedGroup", "UnorderedGroup", "OrderedGroupIndexed", "UnorderedGroupIndexed"}
REFS = {"RegionRef", "RegionRefIndexed"}
# PAGE TextRegion type -> standardized region type (docs/PREDICTIONS.md#region-types); PERO's
# layout CNN gives no type, which is "text". Any other type becomes "other".
REGION_TYPES = {
    "paragraph": "text",
    "heading": "heading",
    "caption": "caption",
    "header": "page-header",
    "footer": "page-footer",
    "page-number": "page-number",
    "footnote": "footnote",
    "footnote-continued": "footnote",
    "list-label": "list-item",
}


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
    device = merged["device"]
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+|cpu", device):
        raise AdapterError(f"device must be 'cuda:<n>' or 'cpu': {device!r}")
    for key in STAGES:
        name = merged[key]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise AdapterError(f"{key} must be a bundle directory name: {name!r}")
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict[str, dict[str, str]]:
    """bundles.sha256 as {bundle: {file: sha256}}; each line is ``<sha256>  <bundle>/<file>``."""
    bundles: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            bundle, file = name.split("/", 1)
            bundles.setdefault(bundle, {})[file] = digest
    return bundles


def verify_bundle(name: str, checksums: dict[str, dict[str, str]]) -> tuple[Path, dict[str, str]]:
    """The bundle directory and its files with checksums; every registered file must match."""
    if name not in checksums:
        known = ", ".join(sorted(checksums))
        raise AdapterError(f"bundle {name!r} is not registered in bundles.sha256 ({known})")
    directory = CACHE / BUNDLES_DIR / name
    for file, digest in checksums[name].items():
        path = directory / file
        if not path.is_file():
            raise AdapterError(f"bundle {name}: missing file {path} (run make pero-bundles)")
        if sha256(path) != digest:
            raise AdapterError(f"bundle {name}: {file} differs from bundles.sha256")
    return directory, dict(checksums[name])


def read_config(directory: Path, role: str) -> configparser.ConfigParser:
    """The bundle's config.ini as PERO reads it, with its stages checked for ``role``."""
    config = configparser.ConfigParser()
    try:
        config.read_string((directory / CONFIG).read_text(encoding="utf-8"))
    except configparser.Error as error:
        raise AdapterError(f"{directory.name}/{CONFIG} is not a valid INI file: {error}") from error
    if not config.has_section("PAGE_PARSER"):
        raise AdapterError(f"{directory.name}/{CONFIG} has no [PAGE_PARSER] section")
    parser = config["PAGE_PARSER"]
    required, forbidden = STAGES[role]
    try:
        enabled = {key for key in (*required, *forbidden) if parser.getboolean(key, False)}
    except ValueError as error:
        raise AdapterError(f"{directory.name}/{CONFIG}: {error}") from error
    if missing := sorted(required - enabled):
        raise AdapterError(f"{role} {directory.name}: {', '.join(missing)} must be yes")
    if extra := sorted(enabled & forbidden):
        what = "decoding with a language model is not supported" if "RUN_DECODER" in extra else ""
        raise AdapterError(
            f"{role} {directory.name}: {', '.join(extra)} must be no {what}".rstrip()
        )
    return config


def referenced_files(config: configparser.ConfigParser, directory: Path, device: str) -> list[str]:
    """Bundle files that the configuration loads, relative to the bundle directory.

    These are the layout models (MODEL_PATH), the OCR definitions (OCR_JSON) and the OCR
    checkpoints they name. PERO's transformer OCR loads ``<checkpoint>.cpu`` on the CPU.
    """
    files = []
    for name in config.sections():
        section = config[name]
        if "MODEL_PATH" in section:
            files.append(section["MODEL_PATH"])
        if "OCR_JSON" in section:
            files.append(section["OCR_JSON"])
            definition = _inside(directory, section["OCR_JSON"])
            try:
                checkpoint = json.loads(definition.read_text(encoding="utf-8"))["checkpoint"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise AdapterError(f"{directory.name}: unreadable OCR_JSON: {error}") from error
            checkpoint = str(Path(section["OCR_JSON"]).parent / checkpoint)
            transformer = section.get("METHOD") == "pytorch_ocr-transformer"
            on_cpu = device == "cpu" or section.getboolean("USE_CPU", False)
            files.append(checkpoint + ".cpu" if transformer and on_cpu else checkpoint)
    return files


def _inside(directory: Path, reference: str) -> Path:
    path = (directory / reference).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise AdapterError(f"{directory.name}: {reference} is outside the bundle")
    return path


def check_references(config, directory: Path, files: dict[str, str], device: str) -> None:
    """Every file the configuration loads exists in the bundle and is checksummed."""
    for reference in referenced_files(config, directory, device):
        path = _inside(directory, reference)
        relative = path.relative_to(directory.resolve()).as_posix()
        if relative not in files:
            hint = (
                " (the CPU needs a separate .cpu checkpoint)" if relative.endswith(".cpu") else ""
            )
            raise AdapterError(
                f"{directory.name}: {reference} is not a registered bundle file{hint}"
            )


def environment(device: str) -> dict[str, Any]:
    """Versions and GPU diagnostics; raises AdapterError when the requested GPU is unusable."""
    try:
        import pero_ocr  # noqa: F401
        import torch
    except Exception as error:
        raise AdapterError(f"cannot import pero-ocr/PyTorch: {error}") from error
    info: dict[str, Any] = {
        "pero_ocr": metadata.version("pero-ocr"),
        "pero_ocr_commit": os.environ.get("OCRBENCH_PERO_COMMIT"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
    }
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        count = torch.cuda.device_count()
        if index >= count:
            raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
        info["gpu"] = torch.cuda.get_device_name(index)
    return info


def load_parsers(params: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """The layout and OCR PageParsers, and the bundles with their files and checksums."""
    import torch
    from pero_ocr.document_ocr.page_parser import PageParser

    if not CHECKSUMS.is_file():
        raise AdapterError(f"{CHECKSUMS} does not exist")
    checksums = read_checksums(CHECKSUMS)
    device = torch.device(params["device"])
    parsers, bundles = [], {}
    for role in STAGES:
        directory, files = verify_bundle(params[role], checksums)
        config = read_config(directory, role)
        check_references(config, directory, files, params["device"])
        # Relative paths in the INI resolve against the bundle directory, as in PERO's
        # parse_folder.py; the INI itself is used as supplied.
        parsers.append(PageParser(config, device=device, config_path=str(directory)))
        bundles[role] = {"name": params[role], "path": str(directory), "files": files}
    return parsers, bundles


def read_image(path: Path) -> Any:
    """The image as PERO's parse_folder.py reads it (8-bit BGR), without EXIF rotation."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise AdapterError(f"cannot decode image {path.name}")
    return image


def _points(element: ET.Element | None, minimum: int) -> list[list[float]] | None:
    text = element.get("points", "") if element is not None else ""
    try:
        points = [[float(v) for v in point.split(",")] for point in text.split()]
    except ValueError:
        return None
    if len(points) >= minimum and all(len(point) == 2 for point in points):
        return points
    return None


def _reading_order(element: ET.Element) -> Iterator[str]:
    """Region references of a PAGE ReadingOrder (sub)tree in reading order.

    The members of an ordered group follow their integer ``@index`` (document order when any is
    missing), nested groups are read in place, and an unordered group keeps document order, as
    in the benchmark's PAGE reader (src/ocrbench/pagexml.py).
    """
    local = element.tag.rsplit("}", 1)[-1]
    if local in GROUPS and element.get("regionRef"):
        yield element.get("regionRef")
    members = [m for m in element if m.tag.rsplit("}", 1)[-1] in GROUPS | REFS]
    if local.startswith("Ordered"):
        keys = [m.get("index") for m in members]
        if all(k is not None and k.isdigit() for k in keys):
            members = [m for _, m in sorted(zip(map(int, keys), members, strict=True))]
    for member in members:
        if member.tag.rsplit("}", 1)[-1] in REFS:
            yield member.get("regionRef", "")
        else:
            yield from _reading_order(member)


def region_type(page_type: str | None) -> str:
    return "text" if page_type is None else REGION_TYPES.get(page_type, "other")


def page_xml_to_prediction(xml: str, sample_id: str, size: tuple[int, int]) -> dict[str, Any]:
    """Standardized prediction (docs/PREDICTIONS.md) from PERO's PAGE XML (``page.xml``).

    Regions and their lines keep PERO's order, which is its reading order: an explicit PAGE
    ReadingOrder (by ``@index``) when one is written, otherwise the order of the regions in the
    file (PERO's region sorter reorders them). Lines with non-blank text keep their polygon,
    baseline and confidence; regions without such lines stay only in the native outputs.
    """
    root = ET.fromstring(xml)
    ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    page = root.find(f"{ns}Page")
    if page is None:
        raise AdapterError("PAGE XML without a Page element")
    native_regions = page.findall(f".//{ns}TextRegion")
    reading_order = page.find(f"{ns}ReadingOrder")
    order = list(_reading_order(reading_order)) if reading_order is not None else []
    if order:
        rank = {ref: i for i, ref in enumerate(order)}
        native_regions.sort(key=lambda r: rank.get(r.get("id"), len(rank)))
    regions: list[dict[str, Any]] = []
    count = 0
    for native in native_regions:
        lines = []
        for native_line in native.findall(f"{ns}TextLine"):
            text = native_line.findtext(f"{ns}TextEquiv/{ns}Unicode") or ""
            if not text.strip():
                continue
            count += 1
            entry: dict[str, Any] = {"id": f"l{count}", "text": text}
            if polygon := _points(native_line.find(f"{ns}Coords"), 3):
                entry["polygon"] = polygon
            if baseline := _points(native_line.find(f"{ns}Baseline"), 2):
                entry["baseline"] = baseline
            conf = native_line.find(f"{ns}TextEquiv")
            if conf is not None and conf.get("conf") is not None:
                entry["confidence"] = float(conf.get("conf"))
            entry["extra"] = {"pero_id": native_line.get("id")}
            lines.append(entry)
        if not lines:
            continue
        region: dict[str, Any] = {"id": f"r{len(regions) + 1}"}
        page_type = native.get("type")
        region["type"] = region_type(page_type)
        if page_type is not None:
            region["native_type"] = page_type
        if polygon := _points(native.find(f"{ns}Coords"), 3):
            region["polygon"] = polygon
        region["lines"] = lines
        region["extra"] = {"pero_id": native.get("id")}
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


def process(parsers: list, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Parse one sample with both PageParsers, write its outputs and return ``result.json``."""
    import torch
    from pero_ocr.core.layout import PageLayout

    name = Path(sample["image"]).name
    image = read_image(source / sample["image"])
    height, width = image.shape[:2]
    device = params["device"]
    if device.startswith("cuda:"):
        torch.cuda.reset_peak_memory_stats(device)
    # PERO breaks ties between lines at the same height with random jitter; a fixed seed per
    # sample makes the line order reproducible.
    random.seed(0)
    layout = PageLayout(id=name, page_size=(height, width))
    for parser in parsers:
        layout = parser.process_page(image, layout)
    xml = layout.to_pagexml_string()
    (out / "page.xml").write_text(xml, encoding="utf-8")
    extra: dict[str, Any] = {}
    try:  # ALTO is an additional native output; the prediction comes from page.xml
        alto = layout.to_altoxml_string(word_splitters=None)
        (out / "alto.xml").write_text(alto, encoding="utf-8")
    except Exception as error:
        extra["alto_error"] = f"{type(error).__name__}: {error}"
    prediction = page_xml_to_prediction(xml, sample["id"], (width, height))
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    regions = prediction.get("regions", [])
    extra |= {
        "pero_regions": len(layout.regions),
        "pero_lines": sum(len(region.lines) for region in layout.regions),
        "regions": len(regions),
        "lines": sum(len(region["lines"]) for region in regions),
    }
    if device.startswith("cuda:"):
        extra["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) >> 20
    return {"status": "ok", "primary": "prediction.json", "extra": extra}


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    parsers, params, info, bundles, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        parsers, bundles = load_parsers(params)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load PERO OCR: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {"bundles": bundles, "job_model": job.get("model")}
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(parsers, params, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=model_info, params=params)
        write_result(out, result)
    print(f"pero adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
