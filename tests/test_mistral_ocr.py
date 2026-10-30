"""Mistral OCR 4.1 hosted-API adapter (engines/mistral-ocr): parameters, the request, the block
conversion to a positioned prediction with region markup, transport failures and the engine
protocol.

Offline tests run the adapter as a host process against a fake API on the loopback interface
(tests/fixtures/mistral-ocr/fake_api.py), started as a separate process so that the test process
itself opens no sockets. Its hand-written response (tests/fixtures/mistral-ocr/response.json) has
the documented shape of an OCR 4.1 response with blocks, in page dimensions of half the image size.
The fake API also throttles, fails, answers from another model and returns malformed bodies. The
opt-in API test builds the engine image and sends the synthetic smoke dataset to the real API
through ``ocrbench run`` with both presets, then evaluates and reports the runs; it makes paid
requests and needs MISTRAL_API_KEY.
"""

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import page_xml, smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "mistral-ocr"
FIXTURES = Path(__file__).parent / "fixtures" / "mistral-ocr"
ADAPTER_PATH = ENGINE_DIR / "mistral_ocr_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
MARKDOWN = tomllib.loads((ENGINE_DIR / "presets" / "markdown.toml").read_text(encoding="utf-8"))
RESPONSE = json.loads((FIXTURES / "response.json").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("mistral_ocr_engine", ADAPTER_PATH)
mo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mo)

# The block texts of tests/fixtures/mistral-ocr/response.json after host conversion, in order;
# the image block has none.
FIXTURE_TEXTS = [
    "Page 3",
    "The Daily Chronicle",
    "The quick brown fox jumps over\nthe lazy dog & the river.",
    "First item\nSecond item",
    "E = mc^2",
    "Year\tCopies\n1923\t2026",
    "3",
]
FAST = {"min_interval_seconds": 0, "retry_base_seconds": 0.01, "max_retries": 2}


def test_presets_are_pinned_and_need_network():
    for config in (CONFIG, MARKDOWN):
        engine = parse_engine_config(config)
        assert engine.model == mo.MODEL
        assert "latest" not in engine.model
        assert (engine.network, engine.secrets) == ("egress", ("MISTRAL_API_KEY",))
        assert engine.gpus is None
        # A request waits at most request_timeout and is retried max_retries times.
        params = mo.parse_params(engine.params)
        assert engine.sample_timeout >= params["request_timeout"] * (params["max_retries"] + 1)
    assert (CONFIG["params"]["mode"], MARKDOWN["params"]["mode"]) == ("blocks", "markdown")
    assert mo.API_URL == "https://api.mistral.ai/v1/ocr"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"mode": "pdf"}, "mode"),
        ({"table_format": "csv"}, "table_format"),
        ({"extract_header": 1}, "extract_header"),
        ({"confidence_scores_granularity": "line"}, "confidence_scores_granularity"),
        ({"request_timeout": 0}, "request_timeout"),
        ({"max_retries": 11}, "max_retries"),
        ({"max_retries": 1.0}, "max_retries"),
        ({"min_interval_seconds": -1}, "min_interval_seconds"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(mo.AdapterError, match=message):
        mo.parse_params(params)


def test_request_sends_only_the_image_and_pinned_options():
    params = mo.parse_params(CONFIG["params"])
    body = mo.request_body(params, "image/jpeg", b"\xff\xd8bytes")
    assert body == {
        "model": "mistral-ocr-4-1",
        "document": {"type": "image_url", "image_url": "data:image/jpeg;base64,/9hieXRlcw=="},
        "include_image_base64": False,
        "include_blocks": True,
        "extract_header": False,
        "extract_footer": False,
        "confidence_scores_granularity": "block",
    }
    plain = mo.request_body(mo.parse_params(MARKDOWN["params"]), "image/png", b"")
    assert plain["include_blocks"] is False
    assert plain["confidence_scores_granularity"] == "page"
    assert "table_format" not in plain  # tables stay inline in the Markdown


def test_blocks_become_positioned_regions_with_markup():
    page = RESPONSE["pages"][0]
    prediction = mo.blocks_to_prediction(page, "s", 900, 420)
    regions = prediction["regions"]
    assert [r["native_type"] for r in regions] == [
        "header",
        "title",
        "text",
        "list",
        "equation",
        "image",
        "table",
        "footer",
    ]
    assert [r["type"] for r in regions] == [
        "page-header",
        "heading",
        "text",
        "text",
        "formula",
        "figure",
        "table",
        "page-footer",
    ]
    assert prediction["reading_order"] == [r["id"] for r in regions]
    # Boxes in page dimensions (450 x 210) scaled to the 900 x 420 image.
    assert regions[1]["polygon"] == [[200, 40], [700, 40], [700, 80], [200, 80]]
    assert regions[1]["markup"] == {"format": "markdown", "content": "# The **Daily** Chronicle"}
    assert regions[1]["confidence"] == 0.91
    assert "markup" not in regions[5]  # the image block references a picture: no text
    assert regions[1]["extra"]["confidence_scores"]["block_type_confidence_score"] == 0.99
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(900, 420))
    assert issues == []
    assert [r.text for r in standardized.regions if r.text] == FIXTURE_TEXTS
    assert standardized.text == "\n\n".join(FIXTURE_TEXTS)
    assert standardized.conversion == {"from": "markdown", "version": 1}


@pytest.mark.parametrize(
    ("text", "loops"),
    [
        ("v Brně, " * 8, True),  # a short unit eight times
        ("v Brně, " * 7, False),
        ("x" + "A long looping sentence that goes on for quite a while, past 64. " * 3, True),
        ("x" + "A long looping sentence that goes on for quite a while, past 64. " * 2, False),
        ("Normal page text.\n\n3", False),
        ("\n" * 20, False),  # blank units do not count
    ],
)
def test_tail_repetition(text, loops):
    assert mo.tail_repeats(text) is loops


def test_repetition_names_the_page_markdown_and_blocks():
    loop = "П. " * 10
    page = {"markdown": "a\n\n" + loop, "blocks": [{"content": "a"}, {"content": loop}]}
    assert mo.repetition(page) == "the page Markdown and block 2"
    assert mo.repetition({"markdown": "a", "blocks": [{"content": loop}] * 2}) == "blocks 1, 2"
    assert mo.repetition({"markdown": "a"}) is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dimensions": None}, "no valid dimensions"),
        ({"dimensions": {"width": 0, "height": 10}}, "no valid dimensions"),
        ({"blocks": None}, "no blocks list"),
        ({"blocks": [{"type": "text", "content": "x"}]}, "no valid box"),
    ],
)
def test_malformed_blocks_are_errors(change, message):
    page = dict(RESPONSE["pages"][0], **change)
    with pytest.raises(mo.AdapterError, match=message):
        mo.blocks_to_prediction(page, "s", 900, 420)


def page_image(path, size, kind="PNG"):
    image = Image.new("L", size, 255)
    ImageDraw.Draw(image).rectangle((10, 10, size[0] // 2, size[1] // 2), fill=0)
    image.save(path, kind)


def make_dataset(root, samples):
    data = root / "data"
    data.mkdir()
    entries = []
    for sid, (size, kind) in samples.items():
        suffix = {"PNG": "png", "JPEG": "jpg", "TIFF": "tif"}[kind]
        page_image(data / f"{sid}.{suffix}", size, kind)
        page_xml(data / f"{sid}.xml", [[(text, (10, 10, 20, 20))] for text in FIXTURE_TEXTS], size)
        entries.append({"id": sid, "image": f"{sid}.{suffix}", "gt": f"{sid}.xml"})
    return write_manifest(data, entries)


@pytest.fixture
def fake_api(tmp_path):
    """The fake API's URL and request log."""
    log = tmp_path / "requests.jsonl"
    process = subprocess.Popen(
        [sys.executable, str(FIXTURES / "fake_api.py"), str(log)],
        stdout=subprocess.PIPE,
        text=True,
    )
    port = int(process.stdout.readline())
    yield f"http://127.0.0.1:{port}/v1/ocr", log
    process.kill()
    process.wait()


def run_adapter(tmp_path, url, samples, config=CONFIG, params=None, env=None):
    config = dict(config, params={**config["params"], **FAST, **(params or {})})
    del config["resources"]
    env = {"OCRBENCH_MISTRAL_URL": url, "MISTRAL_API_KEY": "test-key", **(env or {})}
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, samples),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in samples}


def requests(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_adapter_follows_the_engine_protocol(tmp_path, fake_api):
    url, log = fake_api
    samples = {
        "page": ((900, 420), "JPEG"),
        "blank": ((300, 200), "PNG"),
        "throttled": ((429, 200), "PNG"),
        "down": ((503, 200), "PNG"),
        "denied": ((401, 200), "PNG"),
        "other-model": ((777, 200), "PNG"),
        "two-pages": ((778, 200), "PNG"),
        "not-json": ((779, 200), "PNG"),
        "tiff": ((900, 420), "TIFF"),
        "loop": ((555, 300), "PNG"),
        "after": ((900, 420), "PNG"),
    }
    run, records = run_adapter(tmp_path, url, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # an empty transcription is a result, not a failure
        "throttled": ("success", None),  # retried after Retry-After
        "down": ("failure", "engine-error"),
        "denied": ("failure", "engine-error"),
        "other-model": ("failure", "engine-error"),
        "two-pages": ("failure", "engine-error"),
        "not-json": ("failure", "engine-error"),
        "tiff": ("failure", "engine-error"),
        "loop": ("truncated", None),  # scored as produced
        "after": ("success", None),  # the adapter keeps going after failed samples
    }
    assert records["down"]["message"].startswith("HTTP 503 after 3 attempts")
    assert records["denied"]["message"].startswith("HTTP 401 after 1 attempts")  # not retried
    assert (
        "'mistral-ocr-latest', not the pinned 'mistral-ocr-4-1'"
        in records["other-model"]["message"]
    )
    assert "expected one page in the response, got 2" in records["two-pages"]["message"]
    assert "not JSON" in records["not-json"]["message"]
    assert "does not convert images" in records["tiff"]["message"]
    assert records["loop"]["message"] == "repetition loop at the end of block 3"
    assert records["loop"]["engine_extra"]["tail_repetition"] == "block 3"
    assert records["page"]["engine_extra"]["tail_repetition"] is None

    # What was sent: the image file unchanged with its media type, the pinned model and options;
    # nothing about the sample.
    sent = requests(log)
    assert len(sent) == 1 + 1 + 2 + 3 + 1 + 1 + 1 + 1 + 1 + 1  # no request for the TIFF image
    first = sent[0]
    assert first["document"] == {"type": "image_url", "header": "data:image/jpeg;base64"}
    assert first["bytes"] == (tmp_path / "data" / "page.jpg").stat().st_size
    assert first["authorization"] == "Bearer test-key"
    assert {k: v for k, v in first.items() if k not in ("document", "authorization", "bytes")} == {
        "model": "mistral-ocr-4-1",
        "include_image_base64": False,
        "include_blocks": True,
        "extract_header": False,
        "extract_footer": False,
        "confidence_scores_granularity": "block",
        "path": "/v1/ocr",
        "width": 900,
    }
    assert "page" not in json.dumps([{k: v for k, v in r.items() if k != "path"} for r in sent])

    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "response.json",
        "page.md",
        "prediction.json",
        "result.json",
    }
    assert json.loads((page / "output" / "response.json").read_text()) == RESPONSE  # unchanged
    native = (page / "output" / "page.md").read_text(encoding="utf-8")
    assert native == RESPONSE["pages"][0]["markdown"]
    assert records["page"]["findings"] == []
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": 1}  # converted on the host
    assert [r["text"] for r in prediction["regions"] if "text" in r] == FIXTURE_TEXTS
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n\n".join(FIXTURE_TEXTS)

    extra = records["page"]["engine_extra"]
    assert extra["response_model"] == "mistral-ocr-4-1"
    assert extra["usage"] == {"pages_processed": 1, "doc_size_bytes": 12345}
    assert extra["estimated_cost_usd"] == 0.004
    assert extra["image"]["media_type"] == "image/jpeg"
    assert (extra["image"]["width"], extra["image"]["height"]) == (900, 420)
    assert extra["dimensions"] == {"dpi": 200, "height": 210, "width": 450}
    assert extra["blocks"] == 8
    transport = extra["transport"]
    assert [a["status"] for a in transport["attempts"]] == [200]
    assert transport["headers"] == {
        "x-ratelimit-remaining-req-minute": "59",
        "mistral-correlation-id": "req-900",
    }
    assert transport["latency_seconds"] >= 0
    settings = extra["settings"]
    assert settings["model"] == "mistral-ocr-4-1"
    assert settings["request"]["include_blocks"] is True
    throttled = records["throttled"]["engine_extra"]["transport"]["attempts"]
    assert [a["status"] for a in throttled] == [429, 200]
    down = records["down"]["engine_extra"]["transport"]["attempts"]
    assert [a["status"] for a in down] == [503, 503, 503]
    blank = records["blank"]["engine_extra"]
    assert blank["empty"] is True
    assert (run.path / "samples" / "blank" / "attempt-1" / "prediction.txt").read_text() == ""
    # The response of a rejected answer is kept for inspection.
    kept = run.path / "samples" / "other-model" / "attempt-1" / "output" / "response.json"
    assert json.loads(kept.read_text())["model"] == "mistral-ocr-latest"

    # The API key is never written to an output or record.
    for path in run.path.rglob("*"):
        if path.is_file():
            assert b"test-key" not in path.read_bytes(), path

    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["page"]["chars"]["rate"] < 0.05
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


def test_markdown_preset_converts_the_page_markdown_on_the_host(tmp_path, fake_api):
    url, log = fake_api
    run, records = run_adapter(tmp_path, url, {"page": ((900, 420), "PNG")}, MARKDOWN)
    record = records["page"]
    assert (record["status"], record["primary"]) == ("success", "page.md")
    assert requests(log)[0]["include_blocks"] is False
    page = run.path / "samples" / "page" / "attempt-1"
    assert not (page / "output" / "prediction.json").exists()
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": 1}
    assert "regions" not in prediction or all("polygon" not in r for r in prediction["regions"])
    text = (page / "prediction.txt").read_text(encoding="utf-8")
    assert "**" not in text
    assert "|" not in text
    assert "The Daily Chronicle" in text


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"MISTRAL_API_KEY": ""}, "MISTRAL_API_KEY is not set"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, fake_api, params, env, message):
    url, log = fake_api
    samples = {"a": ((900, 420), "PNG"), "b": ((300, 200), "PNG")}
    _, records = run_adapter(tmp_path, url, samples, params=params, env=env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())
    assert not log.exists()  # no request was sent


def test_job_model_must_be_the_pinned_one(tmp_path, fake_api):
    url, log = fake_api
    config = dict(CONFIG, model="mistral-ocr-latest")
    _, records = run_adapter(tmp_path, url, {"a": ((900, 420), "PNG")}, config)
    assert (
        "the job's model 'mistral-ocr-latest' is not 'mistral-ocr-4-1'" in records["a"]["message"]
    )
    assert not log.exists()


@pytest.fixture(scope="module")
def mistral_image():
    if not os.environ.get("MISTRAL_API_KEY"):
        pytest.fail("MISTRAL_API_KEY is not set")
    subprocess.run(["make", "-C", str(ROOT), "docker-mistral-ocr"], check=True)


@pytest.mark.docker
@pytest.mark.api
def test_api_smoke_benchmark(tmp_path, mistral_image):
    manifest = smoke_dataset(tmp_path / "data")
    runs = str(tmp_path / "runs")
    for name, config in (
        ("blocks", ENGINE_DIR / "engine.toml"),
        ("markdown", ENGINE_DIR / "presets" / "markdown.toml"),
    ):
        args = ["run", "start", "--engine", str(config), "--dataset", str(manifest)]
        assert main([*args, "--runs-dir", runs, "--run-id", name]) == 1  # needs --allow-network
        assert main([*args, "--runs-dir", runs, "--run-id", name, "--allow-network"]) == 0
        run = load_run(tmp_path / "runs" / name)
        records = {s.id: latest_record(run, s.id) for s in run.samples}
        assert {sid: r["status"] for sid, r in records.items()} == {
            "page-1": "success",
            "para-1": "success",
            "blank-1": "success",
        }
        extra = records["page-1"]["engine_extra"]
        assert extra["response_model"] == mo.MODEL
        assert extra["usage"]["pages_processed"] == 1
        if name == "blocks":
            prediction = json.loads(
                (run.path / "samples" / "page-1" / "attempt-1" / "prediction.json").read_text()
            )
            assert prediction["conversion"] == {"from": "markdown", "version": 1}
            assert all("polygon" in region for region in prediction["regions"])
        assert main(["evaluate", str(run.path)]) == 0
        scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
        by_id = {s["sample_id"]: s for s in scores["samples"]}
        assert by_id["para-1"]["chars"]["rate"] < 0.1
        report = tmp_path / f"report-{name}"
        assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
        assert (report / "index.html").is_file()
