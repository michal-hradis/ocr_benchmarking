"""PyLaia adapter (engines/pylaia): line crops, result mapping, parameters and the engine protocol.

Offline tests import the adapter and run it as a host process with the Kraken engine's fake
``torch`` and ``kraken`` modules (tests/fixtures/kraken/fake, which segment a 900x420 image into
the committed Kraken 7.1.1 lines of tests/fixtures/kraken/page.json) and a fake PyLaia recognizer
process (tests/fixtures/pylaia/fake_worker.py). The opt-in GPU test builds the engine image and
runs real Kraken segmentation and PyLaia recognition through ``ocrbench run``, ``evaluate`` and
``report``.
"""

import dataclasses
import importlib.util
import io
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.issues import Severity
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "pylaia"
KRAKEN_DIR = ROOT / "engines" / "kraken"
KRAKEN_FAKE = Path(__file__).parent / "fixtures" / "kraken" / "fake"
FAKE_WORKER = Path(__file__).parent / "fixtures" / "pylaia" / "fake_worker.py"
NATIVE = json.loads(
    (Path(__file__).parent / "fixtures" / "kraken" / "page.json").read_text(encoding="utf-8")
)
ADAPTER_PATH = ENGINE_DIR / "pylaia_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("kraken_engine", _load("kraken_engine", KRAKEN_DIR / "kraken_engine.py"))
pl = _load("pylaia_engine", ADAPTER_PATH)
worker = _load("pylaia_worker", ENGINE_DIR / "pylaia_worker.py")


def bbox(boundary):
    xs, ys = [x for x, _ in boundary], [y for _, y in boundary]
    return [min(xs), min(ys), max(xs), max(ys)]


def record(text, confidences=(0.5, 1.0), crop=(0, 0, 10, 5)):
    return {
        "kraken_id": "x",
        "crop": list(crop) if crop else None,
        "symbols": [],
        "text": text,
        "confidences": list(confidences),
    }


@pytest.mark.parametrize(
    ("boundary", "box"),
    [
        ([[10.4, 5.6], [20.2, 5.6], [20.2, 9.1], [10.4, 9.1]], [10, 5, 21, 10]),
        ([[-3, -2], [5, -2], [5, 4]], [0, 0, 5, 4]),  # clipped to the image
        ([[95, 45], [130, 45], [130, 70]], [95, 45, 100, 50]),
        ([[0, 0], [9, 0]], None),  # not a polygon
        (None, None),
        ([[5, 3], [9, 3], [9, 3]], None),  # zero height
        ([[120, 3], [130, 3], [130, 9]], None),  # outside the image
    ],
)
def test_crop_box_is_the_clipped_bounding_box(boundary, box):
    assert pl.crop_box(boundary, (100, 50)) == box


@pytest.mark.parametrize(
    ("symbols", "text"),
    [
        (["<space>", "Ř", "e", "<space>", "<space>", "č", "<space>"], "Ře  č"),
        (["a", "<unk>", "b"], "a<unk>b"),  # as pylaia-htr-decode-ctc prints it
        ([], ""),
        (["<space>"], ""),
    ],
)
def test_symbols_are_joined_as_pylaia_prints_them(symbols, text):
    assert worker.symbols_to_text(symbols) == text


def test_worker_reads_grayscale_requests():
    first, second = Image.new("L", (3, 2), 7), Image.new("L", (1, 4), 200)
    header = json.dumps({"lines": [[3, 2], [1, 4]]}).encode() + b"\n"
    stream = io.BytesIO(header + first.tobytes() + second.tobytes())
    images = worker.read_request(stream)
    assert [(i.mode, i.size, i.getpixel((0, 0))) for i in images] == [
        ("L", (3, 2), 7),
        ("L", (1, 4), 200),
    ]
    assert worker.read_request(stream) is None  # end of input
    with pytest.raises(EOFError):
        worker.read_request(io.BytesIO(header + first.tobytes()))


class FakeRecognizer:
    def __init__(self):
        self.crops = []

    def recognize(self, crops):
        self.crops = [(crop.mode, crop.size, crop.getpixel((0, 0))) for crop in crops]
        return [
            {"symbols": ["č"], "text": f"line {n}", "confidences": [0.9]} for n in range(len(crops))
        ]


def test_lines_are_cropped_from_the_original_image_in_kraken_order():
    image = Image.new("RGB", (100, 50), (255, 255, 255))
    image.paste((0, 0, 0), (10, 5, 21, 10))
    lines = [
        {"id": "a", "boundary": [[10.4, 5.6], [20.2, 5.6], [20.2, 9.1]]},
        {"id": "b", "boundary": [[1, 1], [2, 2]]},  # no polygon: not recognized
        {"id": "c", "boundary": [[50, 20], [60, 20], [60, 40]]},
    ]
    recognizer = FakeRecognizer()
    records = pl.recognize_lines(recognizer, image, lines)
    assert recognizer.crops == [("L", (11, 5), 0), ("L", (10, 20), 255)]
    assert records == [
        {
            "kraken_id": "a",
            "crop": [10, 5, 21, 10],
            "symbols": ["č"],
            "text": "line 0",
            "confidences": [0.9],
        },
        {"kraken_id": "b", "crop": None, "symbols": [], "text": "", "confidences": []},
        {
            "kraken_id": "c",
            "crop": [50, 20, 60, 40],
            "symbols": ["č"],
            "text": "line 1",
            "confidences": [0.9],
        },
    ]
    assert pl.recognize_lines(recognizer, image, []) == []


def test_prediction_keeps_kraken_geometry_and_order_with_pylaia_text():
    segmentation = NATIVE["segmentation"]
    texts = ["Žluťoučký kůň", "  ", "ſchöne Grüße", "x", ""]
    records = [record(text) for text in texts]
    records[3] = record("x", crop=None, confidences=())  # no crop: no confidence
    prediction = pl.result_to_prediction(segmentation, records, "s1", (900, 420))
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=(900, 420))
    assert standardized is not None, [i.format() for i in issues]
    assert all(i.severity is Severity.INFO for i in issues)
    assert prediction["text"] == "Žluťoučký kůň\nſchöne Grüße\nx"  # blank lines are dropped
    [region] = prediction["regions"]
    [kraken_region] = segmentation["regions"]["text"]
    assert region["polygon"] == kraken_region["boundary"]
    native = segmentation["lines"]
    expected = [native[0], native[2], native[3]]
    assert [line["polygon"] for line in region["lines"]] == [n["boundary"] for n in expected]
    assert [line["baseline"] for line in region["lines"]] == [n["baseline"] for n in expected]
    assert [line["extra"]["kraken_id"] for line in region["lines"]] == [n["id"] for n in expected]
    assert [line.get("confidence") for line in region["lines"]] == [0.75, 0.75, None]
    with pytest.raises(pl.AdapterError, match="differ in length"):
        pl.result_to_prediction(segmentation, records[:2], "s1", (900, 420))


@dataclasses.dataclass
class FakeSegmentation:
    lines: list
    regions: dict


class FakeSegmenter:
    def __init__(self, lines):
        self.lines = lines

    def predict(self, im, config):
        return FakeSegmentation(self.lines, NATIVE["segmentation"]["regions"])


@pytest.mark.parametrize(
    ("broken", "status", "message"),
    [
        (set(), "ok", None),
        (
            {1, 3},
            "partial",
            "2 of 5 segmented lines have no usable polygon and were not recognized",
        ),
    ],
)
def test_uncropped_lines_make_the_page_partial(tmp_path, broken, status, message):
    Image.new("L", (900, 420), 255).save(tmp_path / "page.png")
    lines = [
        line | {"boundary": [[5, 3], [9, 3]]} if n in broken else line
        for n, line in enumerate(NATIVE["segmentation"]["lines"])
    ]
    models = (FakeSegmenter(lines), None, FakeRecognizer())
    params = pl.DEFAULT_PARAMS | {"segmentation_min_height": 0}
    out = tmp_path / "out"
    out.mkdir()
    result = pl.process(models, params, {"id": "s", "image": "page.png"}, tmp_path, out)
    assert (result["status"], result.get("message")) == (status, message)
    assert result["extra"]["uncropped_lines"] == len(broken)
    # A page with no segmented lines is a genuine empty result, not a partial one.
    models = (FakeSegmenter([]), None, FakeRecognizer())
    result = pl.process(models, params, {"id": "s", "image": "page.png"}, tmp_path, out)
    assert (result["status"], result["extra"]["kraken_lines"]) == ("ok", 0)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lang": "ces"}, "unknown parameters: lang"),
        ({"use_language_model": True}, "unknown parameters: use_language_model"),
        ({"device": "gpu"}, "device"),
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": 1.5}, "batch_size"),
        ({"segmentation_min_height": False}, "segmentation_min_height"),
        ({"recognition_model": "iam"}, "recognition_model must be one of newseye-austrian"),
        ({"recognition_model": "/cache/model"}, "recognition_model"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(pl.AdapterError, match=message):
        pl.parse_params(params)


def test_params_defaults():
    assert pl.parse_params({"batch_size": None}) == pl.DEFAULT_PARAMS
    assert pl.DEFAULT_PARAMS["recognition_model"] == CONFIG["params"]["recognition_model"]


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    samples = []
    for sid, width in widths.items():
        Image.new("L", (width, 420), 255).save(data / f"{sid}.png")
        (data / f"{sid}.xml").write_text("<PcGts/>", encoding="utf-8")
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def models_dir(root, files=pl.MODEL_FILES):
    directory = root / "models" / "newseye-austrian"
    directory.mkdir(parents=True)
    for name in files:
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
    return directory.parent


def run_adapter(tmp_path, widths, params=None, env=None, models=None):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"])
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {
        "PYTHONPATH": f"{KRAKEN_FAKE}:{KRAKEN_DIR}",
        "OCRBENCH_PYLAIA_MODELS": str(models or models_dir(tmp_path)),
        "OCRBENCH_PYLAIA_PYTHON": sys.executable,
        "OCRBENCH_PYLAIA_WORKER": str(FAKE_WORKER),
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
    assert outputs == {"segmentation.json", "pylaia.json", "prediction.json", "result.json"}
    native = NATIVE["segmentation"]["lines"]
    # Segmented on a 1200-pixel canvas, moved back: Kraken's committed coordinates come out.
    segmentation = json.loads((page / "output" / "segmentation.json").read_text())
    assert [line["boundary"] for line in segmentation["lines"]] == [n["boundary"] for n in native]
    assert all(line["text"] is None for line in segmentation["lines"])  # unrecognized
    pylaia = json.loads((page / "output" / "pylaia.json").read_text(encoding="utf-8"))
    crops = [line["crop"] for line in pylaia["lines"]]
    assert crops == [bbox(n["boundary"]) for n in native]
    texts = [f"Řádek {x1 - x0}x{y1 - y0}" for x0, y0, x1, y1 in crops]
    assert [line["text"] for line in pylaia["lines"]] == texts
    assert pylaia["lines"][0]["symbols"][:3] == ["<space>", "Ř", "á"]
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n".join(texts)
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    lines = prediction["regions"][0]["lines"]
    assert [line["polygon"] for line in lines] == [n["boundary"] for n in native]
    assert {line["confidence"] for line in lines} == {0.5}
    extra = records["page"]["engine_extra"]
    assert extra["canvas_offset"] == 390
    assert (extra["kraken_lines"], extra["uncropped_lines"], extra["lines"]) == (5, 0, 5)
    environment = extra["environment"]
    assert (environment["kraken"], environment["pylaia"]) == ("7.1.1+fake", "1.1.2+fake")
    assert environment["gpu"] == "Fake GPU"
    files = extra["models"]["files"]
    assert set(files) == {
        "segmentation",
        "recognition/model",
        "recognition/weights.ckpt",
        "recognition/syms.txt",
    }
    assert files["segmentation"]["path"].endswith("/blla.mlmodel")
    assert all(len(f["sha256"]) == 64 for f in files.values())
    assert extra["models"]["job_model"] == CONFIG["model"]
    assert extra["params"] == pl.DEFAULT_PARAMS
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""


def test_recognizer_errors_fail_only_their_sample(tmp_path):
    widths = {"a": 900, "b": 900, "c": 900, "d": 900}
    env = {"FAKE_PYLAIA_ERROR_ON": "2", "FAKE_PYLAIA_CRASH_ON": "3"}
    _, records = run_adapter(tmp_path, widths, env=env)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "a": ("success", None),
        "b": ("failure", "engine-error"),
        "c": ("failure", "engine-error"),
        "d": ("success", None),  # the recognizer was restarted
    }
    assert records["b"]["message"] == "PyLaia: RuntimeError: CUDA out of memory (fake)"
    assert records["c"]["message"] == "the PyLaia process exited with code 3"


@pytest.mark.parametrize(
    ("params", "env", "files", "message"),
    [
        ({"psm": 3}, {}, pl.MODEL_FILES, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, pl.MODEL_FILES, "cuda:0 requested but 0 CUDA devices"),
        ({}, {}, ("model", "syms.txt"), "recognition/weights.ckpt model file /"),
        ({}, {"FAKE_PYLAIA_LOAD_ERROR": "cannot load PyLaia: boom"}, pl.MODEL_FILES, "boom"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, files, message):
    models = models_dir(tmp_path, files)
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params, env, models)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def pylaia_image():
    subprocess.run(
        ["make", "-C", str(ROOT), "docker-pylaia"],
        capture_output=True,
        text=True,
        timeout=3600,
        check=True,
    )
    return CONFIG["image"]


@pytest.mark.docker
@pytest.mark.gpu
def test_gpu_smoke_benchmark(tmp_path, pylaia_image):
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--run-id", "pylaia"]) == 0
    run = load_run(tmp_path / "runs" / "pylaia")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    environment = records["page-1"]["engine_extra"]["environment"]
    assert (environment["kraken"], environment["pylaia"]) == ("7.1.1", "1.1.2")
    assert environment["pylaia_torch"].startswith("1.13.1")
    assert environment["gpu"]
    assert environment["pylaia_gpu"]
    files = records["page-1"]["engine_extra"]["models"]["files"]
    checksums = dict(
        reversed(line.split()) for line in (ENGINE_DIR / "models.sha256").read_text().splitlines()
    )
    assert files["recognition/weights.ckpt"]["sha256"] == checksums["newseye-austrian/weights.ckpt"]
    sample = run.path / "samples" / "page-1" / "attempt-1"
    prediction = json.loads((sample / "prediction.json").read_text())
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    assert len(lines) >= 5
    assert all(line["polygon"] and line["baseline"] for line in lines)
    blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
    assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure

    # Scoring and re-scoring read only stored outputs: no container runs.
    for _ in range(2):
        assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
    rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < 0.25  # a Fraktur newspaper model on a synthetic Antiqua page
    assert rates["para-1"] < 0.25
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
