"""Kraken adapter (engines/kraken): result mapping, parameters and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and ``kraken``
modules (tests/fixtures/kraken/fake) that return a committed result
(tests/fixtures/kraken/page.json and page.xml, real Kraken 7.1.1 output for a synthetic 900x420
page). The opt-in GPU test builds the engine image and runs real Kraken through ``ocrbench run``,
``evaluate`` and ``report``.
"""

import importlib.util
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.issues import Severity
from ocrbench.pagexml import read_page_xml
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "kraken"
FIXTURES = Path(__file__).parent / "fixtures" / "kraken"
NATIVE = json.loads((FIXTURES / "page.json").read_text(encoding="utf-8"))
ADAPTER_PATH = ENGINE_DIR / "kraken_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("kraken_engine", ADAPTER_PATH)
kr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kr)

FIXTURE_LINES = [
    "The Daily Chronicle",
    "The quick brown fox jumps over",
    "the lazy dog near the river.",
    "A second paragraph and more text",
    "with numbers 1923 and 2026.",
]
BOX = [[0, 0], [10, 0], [10, 5], [0, 5]]


def errors(prediction):
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=(900, 420))
    assert standardized is not None, [i.format() for i in issues]
    assert standardized.text == prediction["text"]  # page text agrees with the structure
    return [i for i in issues if i.severity is not Severity.INFO]


def line(lid, text, region=None, kind="default", boundary=BOX):
    return {
        "id": lid,
        "text": text,
        "regions": [region] if region else [],
        "tags": {"type": [{"type": kind}]},
        "baseline": [[0, 4], [10, 4]],
        "boundary": boundary,
    }


def native(lines, regions=None, line_orders=None):
    segmentation = {"type": "baselines", "lines": lines, "regions": regions or {}}
    segmentation["line_orders"] = line_orders or []
    return {"segmentation": segmentation, "confidences": [[0.5, 1.0]] * len(lines)}


def test_result_keeps_kraken_regions_lines_and_geometry():
    prediction = kr.result_to_prediction(NATIVE, "s1", (900, 420))
    assert errors(prediction) == []
    assert prediction["image"] == {"width": 900, "height": 420}
    assert prediction["text"] == "\n".join(FIXTURE_LINES)
    [kraken_region] = NATIVE["segmentation"]["regions"]["text"]
    [region] = prediction["regions"]
    assert region["id"] == "r1"
    assert (region["type"], region["native_type"]) == ("text", "text")
    assert region["polygon"] == kraken_region["boundary"]  # input-image pixels, as is
    assert region["extra"] == {"kraken_id": kraken_region["id"]}
    assert prediction["reading_order"] == ["r1"]
    lines = region["lines"]
    native_lines = NATIVE["segmentation"]["lines"]
    assert [line["text"] for line in lines] == FIXTURE_LINES
    assert [line["id"] for line in lines] == ["l1", "l2", "l3", "l4", "l5"]
    assert [line["polygon"] for line in lines] == [n["boundary"] for n in native_lines]
    assert [line["baseline"] for line in lines] == [n["baseline"] for n in native_lines]
    assert [line["extra"]["kraken_id"] for line in lines] == [n["id"] for n in native_lines]
    means = [round(sum(c) / len(c), 4) for c in NATIVE["confidences"]]
    assert [line["confidence"] for line in lines] == means
    assert all("type" not in line for line in lines)  # Kraken's "default" line type is dropped


def test_native_page_xml_has_the_same_lines_as_the_prediction():
    page, issues = read_page_xml(FIXTURES / "page.xml")
    assert page is not None, [i.format() for i in issues]
    assert all(i.severity is Severity.INFO for i in issues)
    assert (page.width, page.height, page.image_filename) == (900, 420, "page-1.png")
    prediction = kr.result_to_prediction(NATIVE, "s1", (900, 420))
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    assert [line.text for line in page.reference_lines] == [line["text"] for line in lines]
    assert [line.line_id for line in page.reference_lines] == [
        line["extra"]["kraken_id"] for line in lines
    ]


def test_lines_are_grouped_by_region_in_reading_order():
    regions = {
        "Title": [{"id": "t", "boundary": [[0, 0], [9, 0], [9, 9]]}],
        "MainZone:column": [{"id": "m", "boundary": [[0, 0], [1, 1]]}],  # too few points
        "Stamp": [{"id": "s", "boundary": BOX}],  # no lines: only in the native outputs
    }
    lines = [
        line("a", "orphan 1"),
        line("b", "orphan 2", region="unknown-region"),
        line("c", "title", region="t"),
        line("d", " ", region="m"),  # blank: dropped
        line("e", "body 1", region="m", kind="handwritten"),
        line("f", "title 2", region="t"),
        line("g", "body 2", region="m", boundary=[[0, 0], [1, 1]]),
        line("h", "orphan 3"),
    ]
    prediction = kr.result_to_prediction(native(lines, regions), "s1", (900, 420))
    assert errors(prediction) == []
    got = [
        (r["id"], r["type"], r.get("native_type"), [line["text"] for line in r["lines"]])
        for r in prediction["regions"]
    ]
    assert got == [
        ("r1", "text", None, ["orphan 1", "orphan 2"]),
        ("r2", "heading", "Title", ["title", "title 2"]),  # interleaved lines join their region
        ("r3", "text", "MainZone:column", ["body 1", "body 2"]),
        ("r4", "text", None, ["orphan 3"]),
    ]
    assert prediction["reading_order"] == ["r1", "r2", "r3", "r4"]
    assert (
        prediction["text"] == "orphan 1\norphan 2\n\ntitle\ntitle 2\n\nbody 1\nbody 2\n\norphan 3"
    )
    r1, r2, r3, _ = prediction["regions"]
    assert r1["extra"] == {"grouping": "unassigned"}
    assert "polygon" not in r1
    assert "polygon" in r2
    assert "polygon" not in r3
    assert [line["id"] for line in r3["lines"]] == ["l5", "l6"]
    assert r3["lines"][0]["type"] == "handwritten"
    assert "polygon" not in r3["lines"][1]
    assert r3["lines"][0]["confidence"] == 0.75


def test_a_reading_order_model_order_is_used():
    lines = [line("a", "second"), line("b", "first")]
    prediction = kr.result_to_prediction(native(lines, line_orders=[[1, 0]]), "s1", (900, 420))
    assert prediction["text"] == "first\nsecond"


@pytest.mark.parametrize(
    ("kraken_type", "expected"),
    [
        ("text", "text"),
        ("Title", "heading"),
        ("MainZone", "text"),
        ("NumberingZone:page", "page-number"),
        ("GraphicZone", "figure"),
        ("Commentary", "other"),
    ],
)
def test_region_types(kraken_type, expected):
    assert kr.region_type(kraken_type) == expected


def test_empty_and_inconsistent_results():
    empty = kr.result_to_prediction(
        native([], {"text": [{"id": "x", "boundary": BOX}]}), "s1", (9, 9)
    )
    assert empty["text"] == ""
    assert "regions" not in empty
    assert "reading_order" not in empty
    blank = kr.result_to_prediction(native([line("a", "  ")]), "s1", (900, 420))
    assert "regions" not in blank
    broken = native([line("a", "x")])
    broken["confidences"] = []
    with pytest.raises(kr.AdapterError, match="differ in length"):
        kr.result_to_prediction(broken, "s1", (900, 420))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lang": "en"}, "unknown parameters: lang"),
        ({"device": "gpu:0"}, "device"),
        ({"device": "cuda"}, "device"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": 2.0}, "batch_size"),
        ({"segmentation_min_height": -1}, "segmentation_min_height"),
        ({"segmentation_min_height": True}, "segmentation_min_height"),
        ({"recognition_model": "other.mlmodel"}, "recognition_model"),
        ({"recognition_model": "/opt/ocrbench/models/x.mlmodel"}, "recognition_model"),
        ({"recognition_model": "/cache/../etc/passwd"}, "recognition_model"),
        ({"segmentation_model": "blla.mlmodel"}, "segmentation_model"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(kr.AdapterError, match=message):
        kr.parse_params(params)


def test_params_defaults_and_model_paths():
    params = kr.parse_params({"batch_size": None})
    assert params == kr.DEFAULT_PARAMS
    bundled = {"recognition_model": "McCATMuS_nfd_nofix_V1.mlmodel"}
    paths = kr.model_paths(kr.parse_params(bundled | {"segmentation_model": "/cache/s.mlmodel"}))
    assert paths["recognition"] == kr.MODELS_DIR / "McCATMuS_nfd_nofix_V1.mlmodel"
    own = {"recognition_model": "/cache/kraken/r.mlmodel", "segmentation_model": "/cache/s.mlmodel"}
    paths = kr.model_paths(kr.parse_params(own | {"device": "cpu"}))
    assert paths == {
        "segmentation": Path("/cache/s.mlmodel"),
        "recognition": Path(own["recognition_model"]),
    }


@dataclass
class _Line:
    baseline: list
    boundary: list


@dataclass
class _Region:
    boundary: list


@dataclass
class _Segmentation:
    lines: list
    regions: dict


def test_short_images_are_segmented_on_a_canvas_and_moved_back():
    image = Image.new("RGB", (520, 120), (200, 190, 170))
    canvas, offset = kr.segmentation_canvas(image, 1200)
    assert (canvas.size, offset) == ((520, 1200), 540)
    assert canvas.getpixel((5, 5)) == (200, 190, 170)  # the paper colour of the image's edges
    tall, offset = kr.segmentation_canvas(image, 120)
    assert (tall, offset) == (image, 0)
    found = _Segmentation(
        lines=[_Line([[0, 600], [9, 600]], [[0, 530], [9, 590], [9, 700]])],
        regions={"text": [_Region([[0, 540], [9, 660], [5, 600]])]},
    )
    moved = kr.shift_segmentation(found, 540, 120)
    assert moved.lines[0] == _Line([[0, 60], [9, 60]], [[0, 0], [9, 50], [9, 120]])  # clipped
    assert moved.regions["text"][0].boundary == [[0, 0], [9, 120], [5, 60]]
    assert kr.shift_segmentation(found, 0, 120) is found


def test_undecodable_image_is_an_adapter_error(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(kr.AdapterError, match="cannot decode image x.png"):
        kr.read_image(tmp_path / "x.png")
    Image.new("P", (4, 3)).save(tmp_path / "p.png")
    assert kr.read_image(tmp_path / "p.png").mode == "RGB"


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    samples = []
    for sid, width in widths.items():
        if width is None:
            (data / f"{sid}.png").write_bytes(b"not an image")
        else:
            Image.new("L", (width, 420), 255).save(data / f"{sid}.png")
        (data / f"{sid}.xml").write_text("<PcGts/>", encoding="utf-8")
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def run_adapter(tmp_path, widths, params=None, env=None):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"])
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {
        "PYTHONPATH": str(FIXTURES / "fake"),
        "OCRBENCH_KRAKEN_MODELS": str(FIXTURES / "fake" / "models"),
        **(env or {}),
    }
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, widths),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in widths}


def test_adapter_follows_the_engine_protocol(tmp_path):
    widths = {"page": 900, "blank": 300, "boom": 666, "page2": 900}
    run, records = run_adapter(tmp_path, widths)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # nothing segmented: a genuine empty result
        "boom": ("failure", "engine-error"),
        "page2": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: fake segmentation error" in records["boom"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    outputs = {o["path"] for o in records["page"]["outputs"]}
    assert outputs == {"kraken.json", "kraken.xml", "prediction.json", "result.json"}
    # Segmented on a 1200-pixel canvas, moved back: the committed coordinates come out unchanged.
    assert json.loads((page / "output" / "kraken.json").read_text()) == NATIVE
    xml = (page / "output" / "kraken.xml").read_text(encoding="utf-8")
    assert xml.startswith('<PcGts size="900x420" name="page.png">The Daily Chronicle|')
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n".join(FIXTURE_LINES)
    extra = records["page"]["engine_extra"]
    assert extra["canvas_offset"] == 390
    assert (extra["kraken_lines"], extra["lines"], extra["regions"]) == (5, 5, 1)
    assert extra["environment"]["kraken"] == "7.1.1+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    files = extra["models"]["files"]
    assert files["recognition"]["path"].endswith("/catmus-print-fondue-large.mlmodel")
    assert len(files["segmentation"]["sha256"]) == 64
    assert extra["models"]["job_model"] == CONFIG["model"]
    assert extra["params"]["device"] == "cuda:0"
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""


def test_cpu_device_needs_no_gpu(tmp_path):
    params = CONFIG["params"] | {"device": "cpu"}
    _, records = run_adapter(tmp_path, {"page": 900}, params=params, env={"FAKE_TORCH_GPUS": "0"})
    assert records["page"]["status"] == "success"
    assert "gpu" not in records["page"]["engine_extra"]["environment"]


def models_dir(tmp_path, content):
    directory = tmp_path / "models"
    directory.mkdir()
    if content is not None:
        (directory / "catmus-print-fondue-large.mlmodel").write_text(content, encoding="utf-8")
    return {"OCRBENCH_KRAKEN_MODELS": str(directory)}


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, None, "recognition model /"),  # missing file
        ({}, "segmentation", "unsupported model: No recognition model"),
        ({}, "bbox", "unsupported model: the recognition model was trained on bounding-box"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if not isinstance(env, dict):
        env = models_dir(tmp_path, env)
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params=params, env=env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def kraken_image():
    subprocess.run(
        ["docker", "build", "-q", "-t", CONFIG["image"], str(ENGINE_DIR)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=True,
    )
    return CONFIG["image"]


@pytest.mark.docker
@pytest.mark.gpu
def test_gpu_smoke_benchmark(tmp_path, kraken_image):
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--run-id", "kraken"]) == 0
    run = load_run(tmp_path / "runs" / "kraken")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    environment = records["page-1"]["engine_extra"]["environment"]
    assert environment["kraken"] == "7.1.1"
    assert environment["torch"] == "2.14.1+cu130"
    assert environment["gpu"]
    files = records["page-1"]["engine_extra"]["models"]["files"]
    checksums = (ENGINE_DIR / "models.sha256").read_text().split()
    assert files["recognition"]["sha256"] == checksums[0]
    sample = run.path / "samples" / "page-1" / "attempt-1"
    prediction = json.loads((sample / "prediction.json").read_text())
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    assert len(lines) >= 5
    assert all(line["polygon"] and line["baseline"] for line in lines)
    page, _ = read_page_xml(sample / "output" / "kraken.xml")
    assert [line.text for line in page.reference_lines] == [line["text"] for line in lines]
    blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
    assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure

    # Scoring and re-scoring read only stored outputs: no container runs.
    for _ in range(2):
        assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
    rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < 0.1
    assert rates["para-1"] < 0.1  # the paragraph crop is segmented on a page-sized canvas
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # A GPU that is not visible is an explicit engine failure for every sample.
    bad = tmp_path / "bad.toml"
    bad.write_text(
        (ENGINE_DIR / "engine.toml").read_text().replace('"cuda:0"', '"cuda:7"'), encoding="utf-8"
    )
    args[3] = str(bad)
    assert main([*args, "--run-id", "bad-gpu"]) == 0
    run = load_run(tmp_path / "runs" / "bad-gpu")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("cuda:7" in r["message"] for r in records)
