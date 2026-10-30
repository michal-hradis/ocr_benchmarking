"""Docling adapter (engines/docling): conversion, image input, parameters, engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and
``docling`` modules (tests/fixtures/docling/fake). The fake returns committed native outputs:
real Docling 2.145.0 results of preset ``docling`` (docling.json and docling_layout.json) for
synthetic pages rendered by tests/synthetic.py: ``page`` (PAGE), ``paragraph`` (PARAGRAPH, a
crop that Docling's layout takes for a picture with text inside), ``columns`` (COLUMNS),
``newspaper`` (NEWSPAPER), ``table`` (TABLE_PAGE: a table, a picture and a caption) and
``blank``. The opt-in GPU test builds the engine image and runs real Docling through
``ocrbench run``, ``evaluate`` and ``report``.
"""

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.issues import Severity
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import (
    COLUMNS,
    NEWSPAPER,
    PAGE,
    PARAGRAPH,
    TABLE_PAGE,
    page_xml,
    render,
    render_table_figure,
    smoke_dataset_with_columns,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "docling"
FIXTURES = Path(__file__).parent / "fixtures" / "docling"
ADAPTER_PATH = ENGINE_DIR / "docling_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
PRESETS = {
    name: tomllib.loads((ENGINE_DIR / "presets" / f"{name}.toml").read_text(encoding="utf-8"))
    for name in ("rapidocr-regions", "rapidocr-fullpage")
}
SIZES = {
    "page": (900, 420),
    "paragraph": (520, 120),
    "columns": COLUMNS[2],
    "newspaper": NEWSPAPER[2],
    "table": TABLE_PAGE[-1],
    "blank": (300, 200),
}
NATIVE = {
    name: tuple(
        json.loads((FIXTURES / name / f).read_text(encoding="utf-8"))
        for f in ("docling.json", "docling_layout.json")
    )
    for name in SIZES
}

_spec = importlib.util.spec_from_file_location("docling_engine", ADAPTER_PATH)
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


def convert(name, sample_id="s1"):
    document, layout = NATIVE[name]
    return dl.to_prediction(document, layout, sample_id, SIZES[name])


def errors(prediction, size):
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=size)
    assert standardized is not None, [i.format() for i in issues]
    return [i for i in issues if i.severity is not Severity.INFO]


def clusters_by_id(name):
    return {c["id"]: c for c in dl.flatten(NATIVE[name][1]["pages"][0]["clusters"])}


# Conversion of the committed native results -----------------------------------------------------


@pytest.mark.parametrize("name", ["page", "paragraph", "columns", "newspaper", "table"])
def test_items_become_regions_in_document_order_with_their_ocr_cells(name):
    prediction, counts = convert(name)
    assert errors(prediction, SIZES[name]) == []
    document = NATIVE[name][0]
    items = dl.walk(document)
    assert [r["extra"]["self_ref"] for r in prediction["regions"]] == [i["self_ref"] for i in items]
    assert prediction["reading_order"] == [r["id"] for r in prediction["regions"]]
    clusters = clusters_by_id(name)
    for region, item in zip(prediction["regions"], items, strict=True):
        assert region["native_type"] == item["label"]
        assert region["type"] == dl.REGION_TYPES[item["label"]]
        cluster = clusters[region["extra"]["cluster"]]
        texts = [c["text"] for c in cluster["cells"]]
        if item["label"] != "picture":  # a picture's cells are its child items' lines
            assert [line["text"] for line in region.get("lines", [])] == texts
    assert prediction["text"] == "\n\n".join(
        "\n".join(line["text"] for line in r["lines"])
        for r in prediction["regions"]
        if "lines" in r
    )
    assert counts["cells_outside_regions"] == 0
    assert counts["regions"] == len(items)


def test_page_text_and_positions_match_the_rendered_page(tmp_path):
    prediction, _ = convert("page")
    rendered = render(tmp_path / "page.png", PAGE, SIZES["page"])
    assert prediction["text"] == "\n\n".join("\n".join(t for t, _ in r) for r in rendered)
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    boxes = [box for region in rendered for _, box in region]
    for line, (x0, y0, x1, y1) in zip(lines, boxes, strict=True):
        xs, ys = [x for x, _ in line["polygon"]], [y for _, y in line["polygon"]]
        # input-image pixels, top-left origin: the OCR box around the rendered glyphs
        assert [min(xs), max(xs)] == pytest.approx([x0, x1], abs=6)
        assert [min(ys), max(ys)] == pytest.approx([y0, y1], abs=10)


def test_region_boxes_are_provenance_boxes_in_input_pixels():
    """Docling's points (bottom-left origin) times width / page width, flipped to the top."""
    prediction, _ = convert("table")
    document = NATIVE["table"][0]
    page = document["pages"]["1"]["size"]
    scale = SIZES["table"][0] / page["width"]
    assert scale == pytest.approx(3.0)  # 72 / 216 DPI: one point is three input pixels
    items = {i["self_ref"]: i for i in dl.walk(document)}
    for region in prediction["regions"]:
        box = items[region["extra"]["self_ref"]]["prov"][0]["bbox"]
        assert box["coord_origin"] == "BOTTOMLEFT"
        left, top = box["l"] * scale, (page["height"] - box["t"]) * scale
        right, bottom = box["r"] * scale, (page["height"] - box["b"]) * scale
        expected = [left, top, right, top, right, bottom, left, bottom]
        assert sum(region["polygon"], []) == pytest.approx(expected, abs=1e-3)


def test_table_and_picture_regions(tmp_path):
    prediction, _ = convert("table")
    regions = {r["native_type"]: r for r in prediction["regions"]}
    assert regions["table"]["type"] == "table"
    cells = [line["text"] for line in regions["table"]["lines"]]
    rows = TABLE_PAGE[2]
    assert cells[:4] == rows[0] + rows[1][:1]  # cells read row by row
    assert len(cells) == 9
    assert cells.count("135") == 1
    pictures = [r for r in prediction["regions"] if r["native_type"] == "picture"]
    assert len(pictures) == 2
    assert all("lines" not in r for r in pictures)
    assert regions["caption"]["lines"][0]["text"] == TABLE_PAGE[3]
    assert errors(prediction, SIZES["table"]) == []


def test_text_inside_a_picture_is_read_once():
    """The picture's cluster holds the cells of its child clusters; each goes to its child item."""
    prediction, counts = convert("paragraph")
    picture, *words = prediction["regions"]
    assert picture["native_type"] == "picture"
    assert "lines" not in picture
    cluster = clusters_by_id("paragraph")[picture["extra"]["cluster"]]
    assert len(cluster["cells"]) == len(words) == 8
    assert [w["lines"][0]["text"] for w in words] == " ".join(PARAGRAPH[0]).split()
    assert counts == {"regions": 9, "lines": 8, "cells": 8, "cells_outside_regions": 0}


def test_docling_reading_order_is_kept_even_where_it_reads_across_columns():
    """Docling reads the newspaper's paragraphs row by row; the adapter keeps that order."""
    prediction, _ = convert("newspaper")
    firsts = [r["lines"][0]["text"] for r in prediction["regions"][1:4]]
    assert firsts == [column[0][0] for column in NEWSPAPER[1]]


def test_a_blank_page_is_an_empty_prediction():
    prediction, counts = convert("blank")
    assert prediction == {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": "s1",
        "image": {"width": 300, "height": 200},
        "text": "",
    }
    assert counts == {"regions": 0, "lines": 0, "cells": 0, "cells_outside_regions": 0}


# Conversion rules on minimal hand-made results ---------------------------------------------------


def cell(index, text, box, origin="TOPLEFT"):
    left, top, right, bottom = box
    rect = {"r_x0": left, "r_y0": bottom, "r_x1": right, "r_y1": bottom}
    rect |= {"r_x2": right, "r_y2": top, "r_x3": left, "r_y3": top, "coord_origin": origin}
    return {"index": index, "text": text, "rect": rect, "confidence": 0.95, "from_ocr": True}


def cluster(cid, label, box, cells, children=()):
    left, top, right, bottom = box
    bbox = {"l": left, "t": top, "r": right, "b": bottom, "coord_origin": "TOPLEFT"}
    return {
        "id": cid,
        "label": label,
        "bbox": bbox,
        "confidence": 0.9,
        "cells": cells,
        "children": list(children),
    }


def item(ref, label, boxes, text="", children=(), page_no=1, height=100.0):
    """A DoclingDocument item with provenance boxes ``(l, t, r, b)`` given top-left."""
    prov = [
        {
            "page_no": page_no,
            "charspan": [0, 0],
            "bbox": {
                "l": left,
                "t": height - top,
                "r": right,
                "b": height - bottom,
                "coord_origin": "BOTTOMLEFT",
            },
        }
        for left, top, right, bottom in boxes
    ]
    return {
        "self_ref": ref,
        "label": label,
        "prov": prov,
        "text": text,
        "orig": text,
        "children": [{"$ref": c} for c in children],
    }


def native(items, clusters, size=(200.0, 100.0)):
    """A one-page DoclingDocument JSON with ``items`` under the body, and its layout."""
    document = {
        "body": {"self_ref": "#/body", "children": []},
        "furniture": {"self_ref": "#/furniture", "children": []},
        "texts": [],
        "pictures": [],
        "tables": [],
        "groups": [],
        "pages": {"1": {"page_no": 1, "size": {"width": size[0], "height": size[1]}}},
    }
    for entry in items:
        kind, number = entry["self_ref"].lstrip("#/").split("/")
        document[kind].append(entry)
        assert int(number) == len(document[kind]) - 1
    children = {c["$ref"] for entry in items for c in entry["children"]}
    document["body"]["children"] = [
        {"$ref": entry["self_ref"]} for entry in items if entry["self_ref"] not in children
    ]
    layout = {
        "pages": [{"page_no": 1, "size": document["pages"]["1"]["size"], "clusters": clusters}]
    }
    return document, layout


def test_raw_ocr_cells_are_kept_instead_of_doclings_joined_text():
    """Docling's item text removes the hyphen at the line end and replaces typographic quotes."""
    cells = [
        cell(0, "The algo-", (10, 10, 90, 20)),
        cell(1, "rithm „works“ ’here’", (10, 22, 90, 32)),
    ]
    document, layout = native(
        [item("#/texts/0", "text", [(10, 10, 90, 32)], "The algorithm „works\" 'here'")],
        [cluster(0, "text", (10, 10, 90, 32), cells)],
    )
    prediction, _ = dl.to_prediction(document, layout, "s1", (400, 200))
    assert prediction["text"] == "The algo-\nrithm „works“ ’here’"
    assert errors(prediction, (400, 200)) == []


def test_points_become_pixels_on_each_axis():
    document, layout = native(
        [item("#/texts/0", "text", [(10, 20, 50, 30)])],
        [cluster(0, "text", (10, 20, 50, 30), [cell(4, "word", (12, 21, 48, 29), "TOPLEFT")])],
    )
    prediction, _ = dl.to_prediction(document, layout, "s1", (400, 300))  # x 2, y 3
    region = prediction["regions"][0]
    assert region["polygon"] == [[20.0, 60.0], [100.0, 60.0], [100.0, 90.0], [20.0, 90.0]]
    assert region["lines"] == [
        {
            "id": "l4",
            "text": "word",
            "confidence": 0.95,
            "polygon": [[24.0, 87.0], [96.0, 87.0], [96.0, 63.0], [24.0, 63.0]],
        }
    ]
    # The same cell given with a bottom-left origin lands on the same pixels.
    layout["pages"][0]["clusters"][0]["cells"] = [cell(4, "word", (12, 79, 48, 71), "BOTTOMLEFT")]
    flipped, _ = dl.to_prediction(document, layout, "s1", (400, 300))
    assert flipped["regions"][0]["lines"][0]["polygon"] == region["lines"][0]["polygon"]


def test_an_item_merged_from_two_clusters_becomes_two_regions():
    """Docling merges a paragraph continued in the next column into one item with two boxes."""
    left = cluster(0, "text", (10, 10, 90, 40), [cell(0, "first half", (10, 10, 90, 20))])
    right = cluster(1, "text", (110, 10, 190, 40), [cell(1, "second half", (110, 10, 190, 20))])
    document, layout = native(
        [item("#/texts/0", "text", [(10, 10, 90, 40), (110, 10, 190, 40)], "first half second")],
        [right, left],
    )
    prediction, _ = dl.to_prediction(document, layout, "s1", (200, 100))
    assert [
        (r["id"], r["extra"]["prov"], r["extra"]["cluster"]) for r in prediction["regions"]
    ] == [
        ("r1", 0, 0),
        ("r2", 1, 1),
    ]
    assert prediction["text"] == "first half\n\nsecond half"


def test_identical_boxes_prefer_the_cluster_with_the_items_label():
    box = (10, 10, 90, 40)
    picture = cluster(0, "picture", box, [cell(0, "x", box)])
    text = cluster(1, "text", box, [cell(1, "y", box)])
    document, layout = native([item("#/texts/0", "text", [box])], [picture, text])
    prediction, counts = dl.to_prediction(document, layout, "s1", (200, 100))
    assert prediction["text"] == "y"
    assert counts["cells_outside_regions"] == 1


def test_blank_cells_add_no_lines():
    cells = [cell(0, "  ", (10, 10, 50, 20)), cell(1, "text", (10, 22, 50, 32))]
    document, layout = native(
        [item("#/texts/0", "text", [(10, 10, 50, 32)])],
        [cluster(0, "text", (10, 10, 50, 32), cells)],
    )
    prediction, counts = dl.to_prediction(document, layout, "s1", (200, 100))
    assert [line["id"] for line in prediction["regions"][0]["lines"]] == ["l1"]
    assert counts == {"regions": 1, "lines": 1, "cells": 1, "cells_outside_regions": 0}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("unmatched", "no layout cluster matches #/texts/0 box 0"),
        ("page", "#/texts/0 refers to page 2"),
        ("two-pages", "expected one page, got 2"),
        ("shared-cell", "OCR cell 0 is in two regions"),
        ("origin", "unknown coord_origin 'CENTER'"),
        ("cycle", "#/texts/0 appears twice in the document tree"),
        ("layout-page", "the layout clusters belong to another page"),
    ],
)
def test_inconsistent_native_results_are_rejected(change, message):
    shared = cell(0, "shared", (10, 10, 50, 20))
    document, layout = native(
        [
            item("#/texts/0", "text", [(10, 10, 50, 20)]),
            item("#/texts/1", "text", [(60, 10, 90, 20)]),
        ],
        [
            cluster(0, "text", (10, 10, 50, 20), [shared]),
            cluster(1, "text", (60, 10, 90, 20), [copy.deepcopy(shared)]),
        ],
    )
    if change == "unmatched":
        layout["pages"][0]["clusters"][0]["bbox"]["l"] = 11
    elif change == "page":
        document["texts"][0]["prov"][0]["page_no"] = 2
    elif change == "two-pages":
        document["pages"]["2"] = document["pages"]["1"]
    elif change == "origin":
        layout["pages"][0]["clusters"][0]["bbox"]["coord_origin"] = "CENTER"
    elif change == "cycle":
        document["body"]["children"].append({"$ref": "#/texts/0"})
    elif change == "layout-page":
        layout["pages"][0]["size"] = {"width": 1.0, "height": 1.0}
    with pytest.raises(dl.AdapterError, match=message):
        dl.to_prediction(document, layout, "s1", (200, 100))


def test_check_page_size_accepts_only_the_input_image_at_the_nominal_dpi():
    document = {"pages": {"1": {"page_no": 1, "size": {"width": 300.0, "height": 140.0}}}}
    dl.check_page_size(document, (900, 420), 216.0)
    with pytest.raises(dl.AdapterError, match="expected 216.000 x 100.800 pt for 900 x 420 px"):
        dl.check_page_size(document, (900, 420), 300.0)
    turned = {"pages": {"1": {"page_no": 1, "size": {"width": 140.0, "height": 300.0}}}}
    with pytest.raises(dl.AdapterError, match="Docling page is 140.000 x 300.000 pt"):
        dl.check_page_size(turned, (900, 420), 216.0)


# Image input -------------------------------------------------------------------------------------


def test_image_stream_keeps_the_stored_pixels_and_declares_the_nominal_dpi(tmp_path):
    """EXIF orientation is not applied and the file's own DPI is replaced, losslessly."""
    image = Image.effect_noise((64, 40), 50).convert("RGB")
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display
    path = tmp_path / "photo.jpg"
    image.save(path, quality=95, dpi=(300, 300), exif=exif)
    stream, size = dl.image_stream(path, 216.0)
    assert size == (64, 40)
    with Image.open(stream) as tiff:
        assert (tiff.format, tiff.size) == ("TIFF", (64, 40))
        assert tiff.info["dpi"] == (216.0, 216.0)
        assert 0x0112 not in tiff.getexif()
        with Image.open(path) as original:
            assert tiff.tobytes() == original.tobytes()


def test_undecodable_image_is_an_adapter_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(dl.AdapterError, match="cannot decode image broken.png"):
        dl.image_stream(path, 216.0)


# Parameters, preset and models -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"psm": 3}, "unknown parameters: psm"),
        ({"device": "gpu:0"}, "device must be 'cuda:<n>' or 'cpu'"),
        ({"ocr_engine": "tesseract"}, "ocr_engine must be one of"),
        ({"ocr_lang": "latin"}, r"ocr_lang must be one of \[\['latin'\]\]"),
        ({"ocr_lang": ["latin", "en"]}, "ocr_lang must be one of"),
        ({"ocr_lang": ["iso:cs"]}, "the recognizers with pinned models"),
        ({"ocr_mode": "pdf_aware_layout_regions"}, "ocr_mode must be one of"),
        ({"ocr_scale": 0}, "ocr_scale must be a number with 0 < value <= 10"),
        ({"ocr_scale": float("nan")}, "ocr_scale must be a number"),
        ({"table_structure": 1}, "table_structure must be true or false"),
        ({"document_timeout": -1}, "document_timeout must be a positive number or null"),
        ({"num_threads": 0}, "num_threads must be a positive integer"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(dl.AdapterError, match=message):
        dl.parse_params(params)


def test_preset_is_the_adapter_default_written_out():
    assert dl.parse_params(CONFIG["params"]) == dl.DEFAULT_PARAMS
    assert set(CONFIG["params"]) == set(dl.DEFAULT_PARAMS) - {"document_timeout", "num_threads"}
    assert dl.nominal_dpi(dl.DEFAULT_PARAMS) == 216.0
    assert CONFIG["model"] == dl.MODEL
    assert CONFIG["image"].startswith("ocrbench/engine-docling:2.145.0-")
    requirements = (ENGINE_DIR / "requirements.in").read_text(encoding="utf-8")
    assert "docling-slim[models-local,feat-ocr-rapidocr-onnx,convert-core]==2.145.0" in requirements


def test_rapidocr_presets_differ_from_docling_only_in_name_and_ocr_mode():
    assert sorted(p.name for p in (ENGINE_DIR / "presets").iterdir()) == [
        f"{name}.toml" for name in sorted(PRESETS)
    ]
    modes = {name: (p["name"], p["params"]["ocr_mode"]) for name, p in PRESETS.items()}
    assert modes == {
        "rapidocr-regions": ("docling-rapidocr-regions", "layout_regions"),
        "rapidocr-fullpage": ("docling-rapidocr-fullpage", "full_page"),
    }
    for preset in PRESETS.values():
        assert dl.parse_params(preset["params"])["ocr_mode"] == preset["params"]["ocr_mode"]
        params = dict(preset["params"], ocr_mode=CONFIG["params"]["ocr_mode"])
        assert {**preset, "name": CONFIG["name"], "params": params} == CONFIG


def rapidocr_config(cache, **changes):
    """A RapidOCR reader configuration as Docling 2.145.0 builds it, with ``changes``."""
    cfg = {
        "Global": dict.fromkeys(dl.RAPIDOCR_SETTINGS["Global"], 1),
        "Det": dict.fromkeys(dl.RAPIDOCR_SETTINGS["Det"], 0.5),
        "Cls": dict.fromkeys(dl.RAPIDOCR_SETTINGS["Cls"], 6),
        "Rec": dict(rec_img_shape=(3, 48, 320), rec_batch_num=6, rec_keys_path=None),
    }
    for section, name in dl.RAPIDOCR_MODELS.items():
        cfg[section]["model_path"] = cache / dl.MODELS_DIR / name
    for key, value in changes.items():
        section, key = key.split(".")
        cfg[section][key] = value
    return cfg


def test_rapidocr_settings_are_recorded_for_the_pinned_models_only(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE", tmp_path)
    names = (ENGINE_DIR / "models.sha256").read_text(encoding="utf-8").split()[1::2]
    assert set(dl.RAPIDOCR_MODELS.values()) == {n for n in names if n.startswith("RapidOcr/")}
    settings = dl.rapidocr_settings(rapidocr_config(tmp_path))
    assert settings["Rec"] == {
        "rec_img_shape": [3, 48, 320],
        "rec_batch_num": 6,
        "rec_keys_path": None,
        "model_path": str(tmp_path / "models/docling/RapidOcr/latin_PP-OCRv5_rec_mobile.onnx"),
    }
    assert set(settings["Global"]) == set(dl.RAPIDOCR_SETTINGS["Global"])
    json.dumps(settings)
    other = tmp_path / "rapidocr" / "models" / "latin_PP-OCRv6_rec_small.onnx"
    for changes in (
        {"Rec.model_path": other},  # RapidOCR's own model directory, not the pinned file
        {"Det.model_path": None},
        {"Rec.rec_keys_path": "latin_dict.txt"},  # a character list that is not pinned
    ):
        with pytest.raises(dl.AdapterError, match="RapidOCR does not run the pinned models"):
            dl.rapidocr_settings(rapidocr_config(tmp_path, **changes))


def load_fetch_model(monkeypatch):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    return fetch_model


def test_every_pinned_model_has_a_pinned_source(monkeypatch):
    fetch_model = load_fetch_model(monkeypatch)
    names = (ENGINE_DIR / "models.sha256").read_text(encoding="utf-8").split()[1::2]
    assert {name.split("/", 1)[0] for name in names} == set(fetch_model.SOURCES)
    for name in names:
        url = fetch_model.url(name)
        assert url.endswith("/" + name.rsplit("/", 1)[1])
        assert "/resolve/v3.10.0/" in url or len(url.split("/resolve/")[1].split("/")[0]) == 40
    assert fetch_model.MODELS_DIR == dl.MODELS_DIR


def model_cache(root, corrupt=None):
    """A cache with fake model files, and checksums that match them."""
    cache = root / "cache"
    lines = []
    for name in ("RapidOcr/rec.onnx", "docling-project--docling-layout-heron/model.safetensors"):
        path = cache / dl.MODELS_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    if corrupt:
        (cache / dl.MODELS_DIR / corrupt).write_text("changed", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"OCRBENCH_DOCLING_CACHE": str(cache), "OCRBENCH_DOCLING_CHECKSUMS": str(checksums)}


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    fetch_model = load_fetch_model(monkeypatch)
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_DOCLING_CHECKSUMS"]))
    monkeypatch.setitem(fetch_model.RAPIDOCR_PATHS, "rec.onnx", "PP-OCRv5/rec")

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url == f"{fetch_model.SOURCES['RapidOcr']}/PP-OCRv5/rec/rec.onnx"
        Path(path).write_text("wrong", encoding="utf-8")

    models = tmp_path / "cache" / dl.MODELS_DIR / "RapidOcr"
    (models / "rec.onnx").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="RapidOcr/rec.onnx: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert list(models.iterdir()) == []


# Engine protocol through the runner, with the fake modules ---------------------------------------


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    samples = []
    heights = {
        width: SIZES[name][1]
        for name, width in (("newspaper", 1500), ("table", 1200), ("paragraph", 520))
    }
    for sid, width in widths.items():
        size = (width, heights.get(width, 420))
        Image.new("L", size, 255).save(data / f"{sid}.png")
        page_xml(data / f"{sid}.xml", [], size)
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def run_adapter(tmp_path, widths, params=None, env=None, model=None):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"])
    config["model"] = model or CONFIG["model"]
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {"PYTHONPATH": str(FIXTURES / "fake"), **model_cache(tmp_path), **(env or {})}
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, widths),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in widths}


def test_adapter_follows_the_engine_protocol(tmp_path):
    widths = {
        "news": 1500,
        "blank": 300,
        "boom": 666,
        "fail": 777,
        "part": 888,
        "turned": 999,
        "orphan": 910,
        "table": 1200,
    }
    run, records = run_adapter(tmp_path, widths)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "news": ("success", None),
        "blank": ("success", None),
        "boom": ("failure", "engine-error"),
        "fail": ("failure", "engine-error"),
        "part": ("partial", None),
        "turned": ("failure", "engine-error"),
        "orphan": ("success", None),
        "table": ("success", None),  # the adapter keeps going after failed samples
    }
    assert records["boom"]["message"] == "RuntimeError: CUDA out of memory (fake)"
    assert records["fail"]["message"] == "Docling: pipeline: fake failure"
    assert records["part"]["message"] == "timeout: fake timeout"
    assert "Docling page is 140.000 x 333.000 pt" in records["turned"]["message"]
    # Text Docling recognized outside every item is reported, not silently dropped or invented.
    assert records["orphan"]["message"] == (
        "1 of 1 non-blank OCR cells are in no DoclingDocument item and are left out"
    )
    assert records["orphan"]["engine_extra"]["counts"]["cells_outside_regions"] == 1
    assert (run.path / "samples" / "orphan" / "attempt-1" / "prediction.txt").read_text() == ""
    assert records["news"]["message"] is None
    news = run.path / "samples" / "news" / "attempt-1"
    outputs = {o["path"] for o in records["news"]["outputs"]}
    assert outputs == {
        "docling.json",
        "docling.md",
        "docling_layout.json",
        "prediction.json",
        "result.json",
    }
    document, layout = NATIVE["newspaper"]
    assert json.loads((news / "output" / "docling.json").read_text()) == document
    assert json.loads((news / "output" / "docling_layout.json").read_text()) == layout
    expected, counts = convert("newspaper", "news")
    assert (news / "prediction.txt").read_text(encoding="utf-8") == expected["text"]
    extra = records["news"]["engine_extra"]
    assert extra["counts"] == counts
    assert (extra["conversion_status"], extra["errors"]) == ("success", [])
    assert extra["warnings"] == ["docling.fake: fake warning for news.tif"]
    assert (extra["nominal_dpi"], extra["gpu_peak_memory_mb"]) == (216.0, 512)
    environment = extra["environment"]
    assert environment["docling-slim"] == "2.145.0+fake"
    assert environment["rapidocr"] == "3.10.0+fake"
    assert (environment["gpu"], environment["gpu_memory_mb"]) == ("Fake GPU", 24 << 10)
    assert extra["models"]["job_model"] == CONFIG["model"]
    assert set(extra["models"]["files"]) == {
        "RapidOcr/rec.onnx",
        "docling-project--docling-layout-heron/model.safetensors",
    }
    assert extra["params"] == dl.DEFAULT_PARAMS
    options = extra["pipeline_options"]
    assert options["artifacts_path"].endswith("/cache/models/docling")
    assert (options["do_ocr"], options["do_table_structure"]) == (True, True)
    assert options["enable_remote_services"] is False
    assert options["accelerator_options"] == {"device": "cuda:0", "num_threads": 4}
    ocr = options["ocr_options"]
    assert (ocr["lang"], ocr["backend"], ocr["mode"], ocr["scale"]) == (
        ["latin"],
        "onnxruntime",
        "default",
        3.0,
    )
    assert not any(ocr["rapidocr_params"].values())  # RapidOCR on the CPU
    crops = extra["ocr_crops"]  # the default mode: RapidOCR reads the layout clusters
    assert crops["count"] == len(layout["pages"][0]["clusters"])
    assert (crops["shrunk"], crops["min_resize"]) == (0, 1.0)
    rapidocr = extra["rapidocr"]
    assert (rapidocr["Global"]["max_side_len"], rapidocr["Det"]["limit_side_len"]) == (2000, 736)
    assert rapidocr["Rec"]["model_path"].endswith(
        "/models/docling/RapidOcr/latin_PP-OCRv5_rec_mobile.onnx"
    )
    part = run.path / "samples" / "part" / "attempt-1"
    assert (part / "prediction.txt").read_text(encoding="utf-8") == ""
    assert (run.path / "samples" / "blank" / "attempt-1" / "prediction.txt").read_text() == ""

    assert main(["evaluate", str(run.path)]) == 0
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_ocr_mode_reaches_docling(tmp_path, name):
    _, records = run_adapter(tmp_path, {"page": 900}, PRESETS[name]["params"])
    assert records["page"]["status"] == "success"
    extra = records["page"]["engine_extra"]
    assert extra["pipeline_options"]["ocr_options"]["mode"] == PRESETS[name]["params"]["ocr_mode"]
    crops = extra["ocr_crops"]["crops"]
    if name == "rapidocr-fullpage":
        assert crops == [{"box": [0.0, 0.0, 900.0, 420.0], "resize": 1.0}]
    else:
        assert len(crops) == len(NATIVE["page"][1]["pages"][0]["clusters"]) > 1


def test_ocr_crops_are_recorded_in_input_pixels_with_rapidocrs_resize():
    def rect(origin, *box):
        return SimpleNamespace(
            coord_origin=SimpleNamespace(value=origin), **dict(zip("ltrb", box, strict=True))
        )

    page = [rect("TOPLEFT", 0, 0, 1398, 2064)]  # a 4194 x 6192 newspaper read in full
    model = SimpleNamespace(get_ocr_rects=lambda page: page)
    crops = dl.OcrCrops(model, 2000)
    assert model.get_ocr_rects(page) is page  # Docling still gets its rects
    model.get_ocr_rects([rect("BOTTOMLEFT", 100, 2064, 400, 1064), rect("TOPLEFT", 0, 0, 0, 0)])
    summary = crops.summary(3.0, {"width": 1398.0, "height": 2064.0}, (4194, 6192))
    assert summary == {
        "count": 3,
        "shrunk": 2,
        "min_resize": round(2000 / 6192, 4),
        "crops": [
            {"box": [0.0, 0.0, 4194.0, 6192.0], "resize": round(2000 / 6192, 4)},
            {"box": [300.0, 0.0, 1200.0, 3000.0], "resize": round(2000 / 3000, 4)},
            {"box": [0.0, 0.0, 0.0, 0.0], "resize": 1.0},
        ],
    }
    crops.boxes.clear()
    assert crops.summary(3.0, {"width": 1.0, "height": 1.0}, (3, 3))["count"] == 0


def test_cpu_device_reaches_docling(tmp_path):
    params = dict(CONFIG["params"], device="cpu")
    _, records = run_adapter(tmp_path, {"page": 900}, params, {"FAKE_TORCH_GPUS": "0"})
    assert records["page"]["status"] == "success"
    extra = records["page"]["engine_extra"]
    assert extra["pipeline_options"]["accelerator_options"]["device"] == "cpu"
    assert "gpu" not in extra["environment"]
    assert "gpu_peak_memory_mb" not in extra


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "no-cache", "does not exist; download the models with make docling-model"),
        ({}, "corrupt", "not matching models.sha256: ['RapidOcr/rec.onnx']"),
        ({}, "model", "model must be 'docling-2.145.0@layout-heron-8f39ad3c,"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "no-cache":
        env = {"OCRBENCH_DOCLING_CACHE": str(tmp_path / "nowhere")}
    elif env == "corrupt":
        env = model_cache(tmp_path / "corrupt", "RapidOcr/rec.onnx")
    model = None
    if env == "model":  # a copied configuration that names other models than the pinned ones
        env, model = {}, CONFIG["model"].replace("heron-8f39ad3c", "egret-large-0123abcd")
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params, env, model)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def docling_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    for line in (ENGINE_DIR / "models.sha256").read_text(encoding="utf-8").splitlines():
        if not (cache / dl.MODELS_DIR / line.split()[1]).is_file():
            pytest.fail(f"models missing in {cache}: run make docling-model first")
    subprocess.run(
        ["docker", "build", "-q", "-t", CONFIG["image"], str(ENGINE_DIR)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=True,
    )
    return cache


def smoke_dataset(root):
    """The synthetic pages, the table page and page-1 as a JPEG turned by EXIF, at 300 DPI."""
    manifest = smoke_dataset_with_columns(root)
    samples = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
    *table, size = TABLE_PAGE
    page_xml(root / "table-1.xml", render_table_figure(root / "table-1.png", *table, size), size)
    exif = Image.Exif()
    exif[0x0112] = 6  # a viewer shows it turned; the pixels and the GT are page-1's
    with Image.open(root / "page-1.png") as page:
        page.convert("RGB").save(root / "exif-1.jpg", quality=95, dpi=(300, 300), exif=exif)
    (root / "exif-1.xml").write_text((root / "page-1.xml").read_text())
    samples += [
        {"id": "table-1", "image": "table-1.png", "gt": "table-1.xml", "unit": "page"},
        {"id": "exif-1", "image": "exif-1.jpg", "gt": "exif-1.xml", "unit": "page"},
    ]
    return write_manifest(root, samples)


@pytest.mark.docker
@pytest.mark.gpu
def test_gpu_smoke_benchmark(tmp_path, docling_image):
    cache = docling_image
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    args += ["--cache", str(cache)]
    configs = {"docling": engine} | {
        name: str(ENGINE_DIR / "presets" / f"{name}.toml") for name in sorted(PRESETS)
    }
    paths = []
    for run_id, config in configs.items():
        mode = tomllib.loads(Path(config).read_text(encoding="utf-8"))["params"]["ocr_mode"]
        assert main([*args[:3], config, *args[4:], "--run-id", run_id]) == 0
        run = load_run(tmp_path / "runs" / run_id)
        paths.append(str(run.path))
        records = {s.id: latest_record(run, s.id) for s in run.samples}
        assert {sid: r["status"] for sid, r in records.items()} == dict.fromkeys(
            ("page-1", "para-1", "blank-1", "columns-1", "news-1", "table-1", "exif-1"), "success"
        )
        extra = records["page-1"]["engine_extra"]
        assert extra["pipeline_options"]["ocr_options"]["mode"] == mode
        assert extra["rapidocr"]["Global"]["max_side_len"] == 2000
        crops = extra["ocr_crops"]  # the whole page, or the (merged) layout clusters
        assert crops["count"] == 1 if mode == "full_page" else crops["count"] >= 1
        assert crops["shrunk"] == 0  # the synthetic pages are smaller than RapidOCR's limit
        environment = extra["environment"]
        assert environment["docling-slim"] == "2.145.0"
        assert environment["rapidocr"] == "3.10.0"
        assert environment["torch"] == "2.14.1+cu130"
        assert environment["gpu"]
        assert extra["models"]["files"] == dict(
            reversed(line.split())
            for line in (ENGINE_DIR / "models.sha256").read_text().splitlines()
        )
        assert json.loads((run.path / "run.json").read_text())["host"]["gpus"]
        for sid in ("columns-1", "news-1", "table-1", "exif-1"):
            prediction = json.loads(
                (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
            )
            assert prediction["reading_order"]
            assert all("polygon" in region for region in prediction["regions"])
            assert not records[sid]["findings"], sid  # positions fit the input image
        blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
        assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure
        page = (run.path / "samples" / "page-1" / "attempt-1" / "prediction.json").read_text()
        exif = (run.path / "samples" / "exif-1" / "attempt-1" / "prediction.json").read_text()
        lines = [sum(json.loads(p)["regions"][1]["lines"][0]["polygon"], []) for p in (page, exif)]
        assert lines[1] == pytest.approx(lines[0], abs=4)  # the stored pixels, not the EXIF view

        # Scoring and re-scoring read only stored outputs: no container runs.
        for _ in range(2):
            assert main(["evaluate", str(run.path)]) == 0
        scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
        rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
        flexible = {s["sample_id"]: s["flexible_chars"]["rate"] for s in scores["samples"]}
        for sid in ("page-1", "para-1", "columns-1", "exif-1", "table-1"):
            assert rates[sid] < 0.1, sid
        assert flexible["news-1"] < 0.1  # Docling reads its paragraphs across the columns

    report = tmp_path / "report"
    assert main(["report", *paths, "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # A GPU that is not visible is an explicit engine failure for every sample.
    bad = tmp_path / "bad.toml"
    bad.write_text(Path(engine).read_text().replace('"cuda:0"', '"cuda:7"'), encoding="utf-8")
    args[3] = str(bad)
    assert main([*args, "--run-id", "bad-gpu"]) == 0
    run = load_run(tmp_path / "runs" / "bad-gpu")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("cuda:7" in r["message"] for r in records)
