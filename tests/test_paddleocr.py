"""PaddleOCR adapter (engines/paddleocr): result mapping, parameters and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``paddle``,
``paddleocr`` and ``cv2`` modules (tests/fixtures/paddleocr/fake) that return a committed
result (tests/fixtures/paddleocr/page.json, real PaddleOCR 3.7.0 output for a synthetic 900x420
page). The opt-in GPU test builds the engine image and runs real PP-OCRv6 through
``ocrbench run``, ``evaluate`` and ``report``.
"""

import importlib.util
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
ENGINE_DIR = ROOT / "engines" / "paddleocr"
FIXTURES = Path(__file__).parent / "fixtures" / "paddleocr"
NATIVE = json.loads((FIXTURES / "page.json").read_text(encoding="utf-8"))
ADAPTER_PATH = ENGINE_DIR / "paddleocr_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("paddleocr_engine", ADAPTER_PATH)
pocr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pocr)

FIXTURE_LINES = [
    "The Daily Chronicle",
    "The quick brown fox jumps over",
    "the lazy dog near the river.",
    "A second paragraph & more text",
    "with <angle> brackets too.",
]


def native(texts, polys, scores):
    res = {"model_settings": {"use_doc_preprocessor": False}, "rec_texts": texts}
    return {"res": res | {"rec_polys": polys, "rec_scores": scores}}


def errors(prediction):
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=(900, 420))
    assert standardized is not None, [i.format() for i in issues]
    return [i for i in issues if i.severity is not Severity.INFO]


def test_result_lines_go_into_one_region_with_their_quadrilaterals():
    prediction = pocr.result_to_prediction(NATIVE, "s1", (900, 420))
    assert errors(prediction) == []
    assert prediction["image"] == {"width": 900, "height": 420}
    assert prediction["text"] == "\n".join(FIXTURE_LINES)
    [region] = prediction["regions"]
    assert region["id"] == "lines"
    assert region["type"] == "text"
    assert "polygon" not in region
    assert region["extra"] == {"grouping": "all-lines"}
    lines = region["lines"]
    assert [line["text"] for line in lines] == FIXTURE_LINES
    assert [line["id"] for line in lines] == ["l1", "l2", "l3", "l4", "l5"]
    res = NATIVE["res"]
    assert [line["polygon"] for line in lines] == res["rec_polys"]  # input-image pixels, as is
    assert [line["confidence"] for line in lines] == [round(s, 4) for s in res["rec_scores"]]
    assert all(0 < line["confidence"] <= 1 for line in lines)


def test_blank_lines_are_dropped_and_an_empty_page_has_no_regions():
    quad = [[0, 0], [10, 0], [10, 5], [0, 5]]
    prediction = pocr.result_to_prediction(
        native([" ", "ok"], [quad, quad], [0.5, 0.9]), "s1", (900, 420)
    )
    assert [line["id"] for line in prediction["regions"][0]["lines"]] == ["l2"]
    empty = pocr.result_to_prediction(native([], [], []), "s1", (900, 420))
    assert empty["text"] == ""
    assert "regions" not in empty
    assert errors(empty) == []


def test_inconsistent_or_preprocessed_results_are_rejected():
    with pytest.raises(pocr.AdapterError, match="differ in length"):
        pocr.result_to_prediction(native(["a"], [], [0.5]), "s1", (900, 420))
    preprocessed = native([], [], [])
    preprocessed["res"]["model_settings"]["use_doc_preprocessor"] = True
    with pytest.raises(pocr.AdapterError, match="preprocessing"):
        pocr.result_to_prediction(preprocessed, "s1", (900, 420))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lang": "en"}, "unknown parameters: lang"),
        ({"device": "cpu"}, "device"),
        ({"text_det_limit_type": "both"}, "text_det_limit_type"),
        ({"text_det_limit_side_len": 0}, "text_det_limit_side_len"),
        ({"text_recognition_batch_size": 2.5}, "text_recognition_batch_size"),
        ({"text_det_thresh": "0.3"}, "text_det_thresh"),
        ({"text_rec_score_thresh": -1}, "text_rec_score_thresh"),
        ({"text_rec_score_thresh": 2}, "text_rec_score_thresh"),  # would drop every line
        ({"text_det_box_thresh": 2}, "text_det_box_thresh"),
        ({"text_det_thresh": 0}, "text_det_thresh"),
        ({"text_det_box_thresh": 0.0}, "text_det_box_thresh"),
        ({"text_det_unclip_ratio": 0}, "text_det_unclip_ratio"),
        ({"text_det_unclip_ratio": float("inf")}, "text_det_unclip_ratio"),
        ({"text_det_thresh": float("nan")}, "text_det_thresh"),
        ({"text_rec_score_thresh": True}, "text_rec_score_thresh"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(pocr.AdapterError, match=message):
        pocr.parse_params(params)


@pytest.mark.parametrize(
    "params",
    [
        {"text_rec_score_thresh": 0},  # the pipeline default
        {"text_rec_score_thresh": 1.0, "text_det_thresh": 1, "text_det_box_thresh": 0.01},
        {"text_det_unclip_ratio": 25.0},  # an expansion ratio, not a score: no upper bound
    ],
)
def test_valid_thresholds_are_accepted(params):
    assert pocr.parse_params(params).items() >= params.items()


def test_pipeline_uses_pinned_models_without_document_preprocessing():
    params = pocr.parse_params({"text_det_limit_side_len": 1280, "text_det_limit_type": "max"})
    arguments = pocr.pipeline_arguments(params)
    assert arguments["text_detection_model_name"] == "PP-OCRv6_medium_det"
    assert arguments["text_recognition_model_name"] == "PP-OCRv6_medium_rec"
    assert arguments["text_recognition_model_dir"].endswith("/PP-OCRv6_medium_rec")
    assert arguments["use_doc_orientation_classify"] is False
    assert arguments["use_doc_unwarping"] is False
    assert arguments["use_textline_orientation"] is False
    assert arguments["device"] == "gpu:0"
    assert arguments["text_det_limit_side_len"] == 1280
    assert arguments["text_det_limit_type"] == "max"
    assert "text_det_thresh" not in arguments  # None keeps the pipeline default


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
    env = {"PYTHONPATH": str(FIXTURES / "fake"), **(env or {})}
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
        "blank": ("success", None),
        "boom": ("failure", "engine-error"),
        "page2": ("success", None),  # the pipeline keeps going after a failed sample
    }
    assert "RuntimeError: fake pipeline error" in records["boom"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    outputs = {o["path"] for o in records["page"]["outputs"]}
    assert outputs == {"paddleocr.json", "prediction.json", "result.json"}
    assert json.loads((page / "output" / "paddleocr.json").read_text()) == NATIVE
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n".join(FIXTURE_LINES)
    extra = records["page"]["engine_extra"]
    assert extra["environment"]["paddleocr"] == "3.7.0-fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    assert extra["models"]["job_model"] == CONFIG["model"]
    assert extra["params"]["device"] == "gpu:0"
    assert extra["text_det_params"] == NATIVE["res"]["text_det_params"]
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"device": "gpu:0", "psm": 3}, {}, "unknown parameters: psm"),
        ({"device": "gpu:0"}, {"FAKE_PADDLE_GPUS": "0"}, "0 CUDA devices are visible"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params=params, env=env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def paddleocr_image():
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
def test_gpu_smoke_benchmark(tmp_path, paddleocr_image):
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--run-id", "paddle"]) == 0
    run = load_run(tmp_path / "runs" / "paddle")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    environment = records["page-1"]["engine_extra"]["environment"]
    assert environment["paddleocr"] == "3.7.0"
    assert environment["paddle"] == "3.4.0"
    assert environment["gpu"]
    host = json.loads((run.path / "run.json").read_text())["host"]
    assert host["gpus"]  # the runner recorded the host GPU too
    prediction = json.loads(
        (run.path / "samples" / "page-1" / "attempt-1" / "prediction.json").read_text()
    )
    [region] = prediction["regions"]
    assert len(region["lines"]) == 5
    assert all(len(line["polygon"]) == 4 for line in region["lines"])
    blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
    assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure

    # Scoring and re-scoring read only stored outputs: no container runs.
    for _ in range(2):
        assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
    rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < 0.1
    assert rates["para-1"] < 0.1
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # A GPU that is not visible is an explicit engine failure for every sample.
    bad = tmp_path / "bad.toml"
    bad.write_text(
        (ENGINE_DIR / "engine.toml").read_text().replace('"gpu:0"', '"gpu:7"'), encoding="utf-8"
    )
    args[3] = str(bad)
    assert main([*args, "--run-id", "bad-gpu"]) == 0
    run = load_run(tmp_path / "runs" / "bad-gpu")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("gpu:7" in r["message"] for r in records)
