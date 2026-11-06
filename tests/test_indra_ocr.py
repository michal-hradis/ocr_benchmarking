"""Indra OCR adapter (engines/indra-ocr): page mapping, coordinates, parameters, models, protocol.

Offline tests import the adapter and run it as a host process with fake ``indra_ocr`` and
``cv2`` modules (tests/fixtures/indra-ocr/fake). The fake returns committed native pages: real
Indra OCR 1.1.2 output of the engine image (indra.json) for synthetic pages rendered by
tests/synthetic.py (page.json for PAGE, columns.json for COLUMNS, newspaper.json for NEWSPAPER,
blank.json for a blank page, turned.json for PAGE turned a quarter turn counter-clockwise and
line.json for one line). The opt-in Docker test builds the engine image and runs real Indra OCR
on the CPU through ``ocrbench run``, ``evaluate`` and ``report``.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.issues import Severity
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import (
    COLUMNS,
    NEWSPAPER,
    PAGE,
    page_xml,
    render,
    render_columns,
    smoke_dataset_with_columns,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "indra-ocr"
FIXTURES = Path(__file__).parent / "fixtures" / "indra-ocr"
ADAPTER_PATH = ENGINE_DIR / "indra_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = dict(
    reversed(line.split()) for line in (ENGINE_DIR / "models.sha256").read_text().splitlines()
)
NATIVE = {
    name: json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("page", "columns", "newspaper", "blank", "turned", "line")
}
SIZES = {
    "page": (900, 420),
    "columns": COLUMNS[2],
    "newspaper": NEWSPAPER[2],
    "blank": (300, 200),
    "turned": (420, 900),
    "line": (520, 60),
}

_spec = importlib.util.spec_from_file_location("indra_engine", ADAPTER_PATH)
ie = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ie)


def rendered(name, directory):
    """The PAGE regions of a fixture's synthetic page, rendered as for the fixture."""
    if name == "page":
        return render(directory / "page.png", PAGE, SIZES["page"])
    heading, columns, size, font_size = {"columns": COLUMNS, "newspaper": NEWSPAPER}[name]
    return render_columns(directory / f"{name}.png", heading, columns, size, font_size)


def errors(prediction, size):
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=size)
    assert standardized is not None, [i.format() for i in issues]
    return [i for i in issues if i.severity is not Severity.INFO]


def native(blocks, lines, rotation=0, source="ocr"):
    """A minimal indra.json: blocks ``(label, bbox, text)``, lines ``(text, box)``."""
    return {
        "file": "s1.png",
        "page": 0,
        "source": source,
        "rotation": rotation,
        "skew": 0.0,
        "vertical": False,
        "note": "order=model",
        "lines": [{"box": box, "text": text, "conf": 0.9} for text, box in lines],
        "markdown": "",
        "blocks": [{"label": label, "bbox": bbox, "text": text} for label, bbox, text in blocks],
        "timings": {"total_ms": 1.0},
    }


def region_texts(prediction):
    return [[line["text"] for line in r.get("lines", [])] for r in prediction["regions"]]


@pytest.mark.parametrize("name", ["page", "columns", "newspaper"])
def test_blocks_become_regions_in_reading_order_with_their_lines(tmp_path, name):
    prediction, counts = ie.page_to_prediction(NATIVE[name], "s1", SIZES[name])
    assert errors(prediction, SIZES[name]) == []
    expected = [[text for text, _ in region] for region in rendered(name, tmp_path)]
    assert region_texts(prediction) == expected  # Indra reads these pages exactly
    assert prediction["reading_order"] == [r["id"] for r in prediction["regions"]]
    assert [r["native_type"] for r in prediction["regions"]] == [
        b["label"] for b in NATIVE[name]["blocks"]
    ]
    assert prediction["regions"][0]["type"] == "heading"
    assert {r["type"] for r in prediction["regions"][1:]} == {"text"}
    lines = [line for r in prediction["regions"] for line in r["lines"]]
    assert [line["id"] for line in lines] == [f"l{i + 1}" for i in range(len(lines))]
    assert [line["confidence"] for line in lines] == [x["conf"] for x in NATIVE[name]["lines"]]
    assert prediction["text"] == "\n\n".join("\n".join(region) for region in expected)
    assert counts == {
        "lines": len(lines),
        "layout_blocks": len(expected),
        "regions": len(expected),
        "rotation": 0,
    }


def contains(outer, inner, slack):
    return (
        outer[0] - slack <= inner[0]
        and outer[1] - slack <= inner[1]
        and inner[2] <= outer[2] + slack
        and inner[3] <= outer[3] + slack
    )


def bbox(polygon):
    xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


@pytest.mark.parametrize("name", ["page", "newspaper"])
def test_line_and_region_polygons_cover_the_rendered_text_in_image_pixels(tmp_path, name):
    prediction, _ = ie.page_to_prediction(NATIVE[name], "s1", SIZES[name])
    for region, truth in zip(prediction["regions"], rendered(name, tmp_path), strict=True):
        for line, (_, box) in zip(region["lines"], truth, strict=True):
            assert contains(bbox(line["polygon"]), box, slack=3), (line, box)
            assert contains(bbox(region["polygon"]), bbox(line["polygon"]), slack=0.1)


def test_a_turned_page_maps_its_boxes_back_to_the_original_image(tmp_path):
    """Indra reads the page turned upright; positions refer to the image as stored."""
    assert NATIVE["turned"]["rotation"] == 90
    prediction, counts = ie.page_to_prediction(NATIVE["turned"], "s1", SIZES["turned"])
    assert errors(prediction, SIZES["turned"]) == []
    assert counts["rotation"] == 90
    truth = render(tmp_path / "upright.png", PAGE, SIZES["page"])
    assert region_texts(prediction) == [[text for text, _ in region] for region in truth]
    width = SIZES["page"][0]  # of the upright page: (x, y) is at (y, width - x) once turned
    for region, lines in zip(prediction["regions"], truth, strict=True):
        for line, (_, (x0, y0, x1, y1)) in zip(region["lines"], lines, strict=True):
            assert contains(bbox(line["polygon"]), [y0, width - x1, y1, width - x0], slack=3)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_original_polygon_undoes_indras_quarter_turns(rotation):
    """A box on the upright page lands on the same pixels of the turned input image."""
    upright = Image.new("L", (60, 40), 255)
    box = (7, 5, 23, 12)
    ImageDraw.Draw(upright).rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=0)
    image = upright.rotate(rotation, expand=True)  # the content turned counter-clockwise
    polygon = ie.original_polygon(list(box), rotation, image.size)
    inverted = image.point(lambda v: 255 - v)
    assert bbox(polygon) == list(inverted.getbbox())
    assert len(polygon) == 4


def test_page_without_layout_blocks_keeps_its_lines_without_a_region_position():
    prediction, counts = ie.page_to_prediction(NATIVE["line"], "s1", SIZES["line"])
    assert NATIVE["line"]["blocks"] == []
    assert errors(prediction, SIZES["line"]) == []
    (region,) = prediction["regions"]
    assert "polygon" not in region
    assert "type" not in region
    assert region["extra"] == {"grouping": "no-layout"}
    assert [line["text"] for line in region["lines"]] == ["A single line of text"]
    assert counts["regions"] == 1


def test_a_blank_page_is_an_empty_prediction():
    prediction, counts = ie.page_to_prediction(NATIVE["blank"], "s1", SIZES["blank"])
    assert prediction["text"] == ""
    assert "regions" not in prediction
    assert errors(prediction, SIZES["blank"]) == []
    assert counts == {"lines": 0, "layout_blocks": 0, "regions": 0, "rotation": 0}


def test_region_types_pictures_and_blank_lines():
    lines = [("Title", [1, 1, 50, 9]), (" ", [1, 20, 50, 29]), ("x", [1, 30, 9, 39])]
    page = native(
        [
            ("Section-Header", [1, 1, 50, 9], "Title"),
            ("Picture", [60, 1, 90, 40], ""),
            ("Table-Of-Contents", [1, 20, 50, 39], " \nx"),
        ],
        lines,
    )
    prediction, _ = ie.page_to_prediction(page, "s1", (100, 50))
    assert errors(prediction, (100, 50)) == []
    heading, figure, other = prediction["regions"]
    assert (heading["type"], figure["type"], other["type"]) == ("heading", "figure", "other")
    assert "lines" not in figure
    assert "text" not in figure
    assert figure["polygon"] == [[60.0, 1.0], [90.0, 1.0], [90.0, 40.0], [60.0, 40.0]]
    # A blank line is left out, but still belongs to its block: the next line keeps its ID.
    assert [(line["id"], line["text"]) for line in other["lines"]] == [("l3", "x")]
    assert set(ie.REGION_TYPES) >= {"Text", "Section-Header", "Table", "Picture", "List-Group"}


@pytest.mark.parametrize(
    ("page", "message"),
    [
        (
            native([("Text", [0, 0, 9, 9], "a\nb")], [("a", [0, 0, 9, 4]), ("c", [0, 5, 9, 9])]),
            "block 1 does not match the next 2 lines",
        ),
        (
            native([("Text", [0, 0, 9, 9], "a\nb")], [("a", [0, 0, 9, 4])]),
            "block 1 does not match the next 2 lines",
        ),
        (
            native([("Text", [0, 0, 9, 4], "a")], [("a", [0, 0, 9, 4]), ("b", [0, 5, 9, 9])]),
            "the layout blocks hold 1 of 2 lines",
        ),
        (native([], [], source="text"), "Indra page source is 'text'"),
        (native([], [], rotation=45), "unexpected page rotation 45"),
    ],
)
def test_inconsistent_pages_are_rejected(page, message):
    with pytest.raises(ie.AdapterError, match=re.escape(message)):
        ie.page_to_prediction(page, "s1", (20, 20))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"math": True}, "unknown parameters: math"),
        ({"threads": 0}, "threads must be an integer of at least 1: 0"),
        ({"threads": "4"}, "threads must be an integer of at least 1: '4'"),
        ({"threads": True}, "threads must be an integer of at least 1: True"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(ie.AdapterError, match=re.escape(message)):
        ie.parse_params(params)


def test_params_defaults_and_cpu_allocation():
    assert ie.parse_params({}) == ie.DEFAULT_PARAMS == {"threads": 4}
    assert ie.parse_params(CONFIG["params"]) == ie.DEFAULT_PARAMS
    threads = CONFIG["params"]["threads"]
    assert CONFIG["resources"]["cpus"] == threads
    assert CONFIG["env"]["OMP_NUM_THREADS"] == str(threads)
    assert "gpus" not in CONFIG["resources"]


def test_image_downloads_exactly_the_checksummed_default_models():
    dockerfile = (ENGINE_DIR / "Dockerfile").read_text(encoding="utf-8")
    commit = re.search(r"ARG INDRA_MODEL_COMMIT=([0-9a-f]{40})", dockerfile).group(1)
    assert commit == ie.MODEL_COMMIT
    assert CONFIG["model"] == ie.MODEL_ID == f"datalab-to/indra-ocr@{ie.MODEL_TAG}:{commit}"
    assert "sha256sum --check --strict /opt/ocrbench/models.sha256" in dockerfile
    assert "ORT_DISABLE_TELEMETRY=1" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    # Indra's default model set (indra_ocr.models: *.onnx, meta.json, lm*.npz) and its license,
    # without the optional math models and the unused safetensors copy of the recognizer.
    assert set(CHECKSUMS) == {
        "LICENSE",
        "meta.json",
        *(f"{name}.onnx" for name in ("det", "rec", "order", "orient", "script")),
        *(f"rec_{group}.onnx" for group in ("cjk", "rtl", "world")),
        "lm.npz",
        "lm_cjk.npz",
        "lm_world.npz",
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in CHECKSUMS.values())
    requirements = (ENGINE_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "indra-ocr==1.1.2 \\" in requirements
    assert "--hash=sha256:" in requirements


def model_dir(root, corrupt=None, missing=None):
    """Fake model files, and checksums that match them."""
    models = root / "models"
    models.mkdir(parents=True)
    lines = []
    for name in CHECKSUMS:
        path = models / name
        path.write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    if corrupt:
        (models / corrupt).write_text("changed", encoding="utf-8")
    if missing:
        (models / missing).unlink()
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"OCRBENCH_INDRA_MODELS": str(models), "OCRBENCH_INDRA_CHECKSUMS": str(checksums)}


def make_dataset(root, sizes):
    data = root / "data"
    data.mkdir()
    samples = []
    for sid, size in sizes.items():
        Image.new("L", size, 255).save(data / f"{sid}.png")
        page_xml(data / f"{sid}.xml", [], size)
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def run_adapter(tmp_path, sizes, params=None, env=None, model=CONFIG["model"]):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"], model=model)
    if model is None:
        del config["model"]
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {
        "PYTHONPATH": str(FIXTURES / "fake"),
        **model_dir(tmp_path / "models"),
        **(env or {}),
    }
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, sizes),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in sizes}


def test_adapter_follows_the_engine_protocol(tmp_path):
    log = tmp_path / "load.json"
    sizes = {
        "news": SIZES["newspaper"],
        "blank": SIZES["blank"],
        "boom": (666, 100),
        "empty": (500, 100),
        "turned": SIZES["turned"],
        "page": SIZES["page"],
    }
    run, records = run_adapter(tmp_path, sizes, env={"FAKE_INDRA_LOG": str(log)})
    assert {sid: (r["status"], r["reason"]) for sid, r in records.items()} == {
        "news": ("success", None),
        "blank": ("success", None),
        "boom": ("failure", "engine-error"),
        "empty": ("success", None),  # nothing detected: a genuine empty result
        "turned": ("success", None),
        "page": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: onnxruntime failed (fake)" in records["boom"]["message"]
    # Indra is loaded once, on the pinned models, in OCR mode without math, on the CPU.
    assert json.loads(log.read_text()) == {
        "path": str(tmp_path / "models" / "models"),
        "threads": 4,
        "mode": "ocr",
        "math": False,
        "furniture": False,
        "providers": ["CPUExecutionProvider"],
        "cv2_threads": [4],
    }
    news = run.path / "samples" / "news" / "attempt-1"
    assert {o["path"] for o in records["news"]["outputs"]} == {
        "indra.json",
        "prediction.json",
        "result.json",
    }
    assert json.loads((news / "output" / "indra.json").read_text()) == NATIVE["newspaper"] | {
        "file": "news.png"
    }
    expected, counts = ie.page_to_prediction(NATIVE["newspaper"], "news", SIZES["newspaper"])
    assert json.loads((news / "output" / "prediction.json").read_text()) == expected
    standardized, _ = validate_prediction(expected, sample_id="news")
    assert (news / "prediction.txt").read_text(encoding="utf-8") == standardized.text
    assert standardized.text.startswith("Evening Gazette\n\nRain fell over the city\n")
    extra = records["news"]["engine_extra"]
    assert {k: extra[k] for k in counts} == counts
    assert extra["indra_seconds"] >= 0
    assert extra["timings"] == NATIVE["newspaper"]["timings"]
    environment = extra["environment"]
    assert environment["indra-ocr"] == "1.1.2+fake"
    assert environment["onnxruntime"] == "1.31.1+fake"
    assert environment["threads"] == 4
    assert environment["affinity_cpus"] >= 1
    assert "cpu_max" in environment
    assert environment["load_seconds"] >= 0
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["repository"], models["tag"]) == ("datalab-to/indra-ocr", "v1.1.0")
    assert models["commit"] == "a1283c4b42a273c09480ad50e71461daff50fcc7"
    assert set(models["files"]) == set(CHECKSUMS)
    assert extra["params"] == ie.DEFAULT_PARAMS
    assert records["turned"]["engine_extra"]["rotation"] == 90
    for sid in ("blank", "empty"):
        path = run.path / "samples" / sid / "attempt-1" / "prediction.txt"
        assert path.read_text(encoding="utf-8") == ""

    assert main(["evaluate", str(run.path)]) == 0
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


def test_threads_reach_indra_and_opencv(tmp_path):
    log = tmp_path / "load.json"
    _, records = run_adapter(
        tmp_path, {"page": SIZES["page"]}, {"threads": 2}, {"FAKE_INDRA_LOG": str(log)}
    )
    assert records["page"]["status"] == "success"
    assert records["page"]["engine_extra"]["params"] == {"threads": 2}
    recorded = json.loads(log.read_text())
    assert (recorded["threads"], recorded["cv2_threads"]) == (2, [2])


@pytest.mark.parametrize(
    ("params", "models", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"corrupt": "rec.onnx"}, "not matching models.sha256: ['rec.onnx']"),
        ({}, {"missing": "det.onnx"}, "not matching models.sha256: ['det.onnx']"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, models, message):
    env = model_dir(tmp_path / "bad", **models)
    _, records = run_adapter(tmp_path, {"a": (900, 420), "b": (300, 200)}, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


@pytest.mark.parametrize(
    "model",
    [
        "datalab-to/indra-ocr@v1.1.0:0000000000000000000000000000000000000000",
        "datalab-to/indra-ocr@v1.2.0:a1283c4b42a273c09480ad50e71461daff50fcc7",
        "other-model@rev1",
        None,
    ],
)
def test_a_job_for_another_model_fails_every_sample(tmp_path, model):
    """The image runs only its pinned models; results are never attributed to another one."""
    log = tmp_path / "load.json"
    _, records = run_adapter(
        tmp_path, {"a": (900, 420), "b": (300, 200)}, env={"FAKE_INDRA_LOG": str(log)}, model=model
    )
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    expected = f"job model {model!r} is not the pinned model {ie.MODEL_ID!r}"
    assert all(r["message"] == expected for r in records.values())
    assert not log.exists()  # Indra is not even loaded


def test_load_errors_fail_every_sample_explicitly(tmp_path):
    # meta.json is not pinned in this checksum file, so only Indra's loading notices it is gone.
    env = model_dir(tmp_path / "partial", missing="meta.json")
    checksums = Path(env["OCRBENCH_INDRA_CHECKSUMS"])
    lines = checksums.read_text().splitlines()
    checksums.write_text("\n".join(x for x in lines if not x.endswith("meta.json")) + "\n")
    _, records = run_adapter(tmp_path, {"a": (900, 420)}, None, env)
    assert records["a"]["status"] == "failure"
    assert "cannot load Indra OCR: FileNotFoundError" in records["a"]["message"]


def test_undecodable_image_fails_the_sample(tmp_path):
    sys.path.insert(0, str(FIXTURES / "fake"))
    try:
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")
        with pytest.raises(ie.AdapterError, match="cannot decode image broken.png"):
            ie.read_image(path)
    finally:
        sys.path.remove(str(FIXTURES / "fake"))
        sys.modules.pop("cv2", None)


# Opt-in Docker smoke benchmark -------------------------------------------------------------------


@pytest.fixture(scope="module")
def indra_image():
    subprocess.run(["make", "-C", str(ROOT), "docker-indra-ocr"], check=True)


@pytest.mark.docker
def test_docker_smoke_benchmark(tmp_path, indra_image):
    manifest = smoke_dataset_with_columns(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--run-id", "indra"]) == 0
    run = load_run(tmp_path / "runs" / "indra")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == dict.fromkeys(
        ("page-1", "para-1", "blank-1", "columns-1", "news-1"), "success"
    )
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"]["indra-ocr"] == "1.1.2"
    assert extra["environment"]["onnxruntime"] == "1.31.1"
    assert extra["environment"]["cpu_max"] == "400000 100000"  # --cpus 4
    assert extra["models"]["files"] == CHECKSUMS
    for sid in ("columns-1", "news-1"):
        prediction = json.loads(
            (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
        )
        assert prediction["reading_order"]
        assert all("polygon" in line for r in prediction["regions"] for line in r["lines"])
    blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
    assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure
    log = next((run.path / "containers").glob("*/log.txt")).read_text(encoding="utf-8")
    assert "telemetry" not in log

    # Scoring and re-scoring read only stored outputs: no container runs.
    for _ in range(2):
        assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
    rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    for sid in ("page-1", "para-1", "columns-1", "news-1"):
        assert rates[sid] < 0.05, sid  # ordered CER: the columns are read one after another
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()
