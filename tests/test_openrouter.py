"""OpenRouter hosted vision-model adapter (engines/openrouter): parameters, the pinned model and
provider routing, the request, transport failures, finish reasons and the engine protocol.

Offline tests run the adapter as a host process against a fake API on the loopback interface
(tests/fixtures/openrouter/fake_api.py), started as a separate process so that the test process
itself opens no sockets. Its hand-written response (tests/fixtures/openrouter/response.json) has
the shape of a real chat completion from google/gemini-3.8-flash with a plain-text answer; the
fake API also answers in Markdown, throttles, fails, answers from another model, stops at the
token limit or a content filter and returns malformed bodies. The opt-in API test builds the
engine image and sends the synthetic smoke dataset to the real API through ``ocrbench run`` with
every preset, then evaluates and reports the runs; it makes paid requests and needs
OPENROUTER_API_KEY.
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
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import page_xml, smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "openrouter"
FIXTURES = Path(__file__).parent / "fixtures" / "openrouter"
ADAPTER_PATH = ENGINE_DIR / "openrouter_engine.py"
# The model cohort (#59): model, provider, the provider's name in responses, and whether the
# provider accepts temperature and seed for the model.
COHORT = {
    "gemini-3.8-flash": ("google/gemini-3.8-flash", "google-ai-studio", "Google AI Studio", 1, 1),
    "qwen3.8-flash": ("qwen/qwen3.8-flash", "alibaba", "Alibaba", 1, 1),
    "gpt-6.1-terra": ("openai/gpt-6.1-terra", "openai", "OpenAI", 0, 1),
    "claude-sonnet-5.5": ("anthropic/claude-sonnet-5.5", "anthropic", "Anthropic", 0, 0),
    "gpt-6-luna": ("openai/gpt-6-luna", "openai", "OpenAI", 0, 1),
    "claude-haiku-5.5": ("anthropic/claude-haiku-5.5", "anthropic", "Anthropic", 0, 0),
}
PRESET_FILES = {
    "openrouter-gemini-3.8-flash-literal": ENGINE_DIR / "engine.toml",
    "openrouter-gemini-3.8-flash-layout": ENGINE_DIR / "presets" / "layout.toml",
    "openrouter-gemini-3.8-flash-literal-original": (
        ENGINE_DIR / "presets" / "literal-original.toml"
    ),
} | {
    f"openrouter-{slug}-{prompt}": ENGINE_DIR / "presets" / f"{slug}-{prompt}.toml"
    for slug in list(COHORT)[1:]
    for prompt in ("literal", "layout")
}
PRESETS = {n: tomllib.loads(f.read_text(encoding="utf-8")) for n, f in PRESET_FILES.items()}
CONFIG = PRESETS["openrouter-gemini-3.8-flash-literal"]
RESPONSE = json.loads((FIXTURES / "response.json").read_text(encoding="utf-8"))
MODEL = "google/gemini-3.8-flash"


def cohort_entry(name):
    """The preset's cohort entry: model, provider, provider name, temperature and seed sent."""
    return next(v for slug, v in COHORT.items() if name.startswith(f"openrouter-{slug}-"))


_spec = importlib.util.spec_from_file_location("openrouter_engine", ADAPTER_PATH)
oe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oe)

# The plain-text answer in tests/fixtures/openrouter/response.json, and the fake API's Markdown
# answer (image width 910), which converts to it.
FIXTURE_TEXT = "The Daily Chronicle\n\nThe quick brown fox jumps over\nthe lazy dog & the river."
MARKDOWN = (
    "# The **Daily** Chronicle\n\nThe quick brown fox jumps over\nthe lazy dog &amp; the river."
)
LAYOUT = "openrouter-gemini-3.8-flash-layout"
ORIGINAL = "openrouter-gemini-3.8-flash-literal-original"
FAST = {"min_interval_seconds": 0, "retry_base_seconds": 0.01, "max_retries": 2}
REQUIRED = {"provider": "google-ai-studio", "prompt": "OCR"}


@pytest.mark.parametrize("name", PRESETS)
def test_presets_are_pinned_and_need_network(name):
    engine = parse_engine_config(PRESETS[name])
    assert engine.name == name
    assert (engine.engine_version, engine.image) == ("2", "ocrbench/engine-openrouter:2-v2")
    model, provider, _, temperature, seed = cohort_entry(name)
    assert engine.model == model == oe.check_model(engine.model)
    assert (engine.network, engine.secrets) == ("egress", ("OPENROUTER_API_KEY",))
    assert engine.gpus is None
    params = oe.parse_params(engine.params)
    assert params["provider"] == provider
    # Temperature 0 and seed 0 where the provider accepts them, else not sent.
    assert params["temperature"] == (0 if temperature else None)
    assert params["seed"] == (0 if seed else None)
    assert params["answer_format"] == "text"  # the prompts ask for plain text
    assert params["reasoning_effort"] == "minimal"
    assert params["image_detail"] == ("original" if name.endswith("-original") else None)
    # Every attempt may wait its full timeout and every retry the longest delay.
    assert oe.max_request_seconds(params) == 5 * 301 + 4 * 120
    assert engine.sample_timeout >= oe.max_request_seconds(params)
    assert oe.API_URL == "https://openrouter.ai/api/v1/chat/completions"


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/auto",
        "~google/gemini-flash-latest",
        "google/gemini-flash-latest",
        "google/gemini-3.8-flash:batch",
        "google/gemini-3.8-flash:free",
        "gemini-3.8-flash",
        "google/",
        None,
    ],
)
def test_routers_aliases_and_variants_are_refused(model):
    with pytest.raises(oe.AdapterError, match="job's model"):
        oe.check_model(model)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"psm": 3}, "unknown parameters: psm"),
        ({"prompt": "OCR"}, "provider"),
        ({"provider": "google-ai-studio"}, "prompt"),
        (dict(REQUIRED, prompt=" "), "prompt"),
        (dict(REQUIRED, max_tokens=8), "max_tokens"),
        (dict(REQUIRED, max_tokens=100.0), "max_tokens"),
        (dict(REQUIRED, temperature=3), "temperature"),
        (dict(REQUIRED, seed=1.5), "seed"),
        (dict(REQUIRED, reasoning_effort="max"), "reasoning_effort"),
        (dict(REQUIRED, image_detail="ultra_high"), "image_detail"),  # ignored by OpenRouter
        (dict(REQUIRED, answer_format="html"), "answer_format"),
        (dict(REQUIRED, request_timeout=0), "request_timeout"),
        (dict(REQUIRED, max_retries=11), "max_retries"),
        (dict(REQUIRED, min_interval_seconds=-1), "min_interval_seconds"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(oe.AdapterError, match=message):
        oe.parse_params(params)


def test_request_pins_the_provider_without_fallbacks():
    params = oe.parse_params(CONFIG["params"])
    body = oe.request_body(MODEL, params, "data:image/png;base64,AAAA")
    assert body == {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CONFIG["params"]["prompt"]},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ],
        "provider": {
            "only": ["google-ai-studio"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        },
        "max_tokens": 32768,
        "temperature": 0.0,
        "seed": 0,
        "reasoning": {"effort": "minimal"},
        "stream": False,
    }
    plain = oe.request_body(MODEL, oe.parse_params(REQUIRED), "x")  # null settings are not sent
    assert not {"temperature", "seed", "reasoning"} & set(plain)
    zero = oe.request_body(MODEL, oe.parse_params(dict(REQUIRED, temperature=0, seed=0)), "x")
    assert (zero["temperature"], zero["seed"]) == (0, 0)
    detailed = oe.request_body(MODEL, oe.parse_params(PRESETS[ORIGINAL]["params"]), "x")
    assert detailed["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "x", "detail": "original"},
    }


def test_presets_differ_only_in_prompt_or_image_detail():
    literal = oe.parse_params(CONFIG["params"])
    layout = oe.parse_params(PRESETS[LAYOUT]["params"])
    original = oe.parse_params(PRESETS[ORIGINAL]["params"])
    assert dict(layout, prompt=None) == dict(literal, prompt=None)
    assert dict(original, image_detail=None) == literal
    for config in PRESETS.values():
        assert {k: v for k, v in config.items() if k not in ("name", "params", "model")} == {
            k: v for k, v in CONFIG.items() if k not in ("name", "params", "model")
        }


@pytest.mark.parametrize("name", [n for n in PRESETS if not n.startswith("openrouter-gemini")])
def test_cohort_presets_share_the_gemini_prompts_and_settings(name):
    """Each further model has the literal and layout prompts and settings of the Gemini presets;
    only the model, the provider, and temperature and seed where the provider refuses them,
    differ (#59)."""
    gemini = PRESETS["openrouter-gemini-3.8-flash-" + name.rsplit("-", 1)[1]]
    params, base = oe.parse_params(PRESETS[name]["params"]), oe.parse_params(gemini["params"])
    differ = {"provider", "temperature", "seed"}
    assert {k: v for k, v in params.items() if k not in differ} == {
        k: v for k, v in base.items() if k not in differ
    }


def test_settings_fingerprints_are_stable_and_exclude_the_image():
    params = oe.parse_params(CONFIG["params"])
    settings = oe.settings_of(MODEL, params)
    assert settings == oe.settings_of(MODEL, dict(params))
    assert settings["request"]["messages"][0]["content"][1]["image_url"]["url"] == "<image>"
    assert settings["prompt_sha256"] == oe.fingerprint(CONFIG["params"]["prompt"])
    changed = oe.settings_of(MODEL, dict(params, temperature=0.5))
    assert changed["request_sha256"] != settings["request_sha256"]
    assert changed["prompt_sha256"] == settings["prompt_sha256"]


@pytest.mark.parametrize(
    ("reported", "matches"),
    [
        (MODEL, True),
        (f"{MODEL}-20260902", True),  # a dated snapshot of the pinned model
        (f"{MODEL}-lite", False),
        (f"{MODEL}-2026", False),
        ("google/gemini-3.7-flash", False),
        (None, False),
    ],
)
def test_model_matches_the_pinned_one_or_its_snapshot(reported, matches):
    assert oe.model_matches(MODEL, reported) is matches


def test_retry_delay_and_statuses():
    client = oe.Client(oe.parse_params(REQUIRED), "key")
    assert client.delay(0, oe.lower_names({"Retry-After": "7"})) == 7.0
    assert client.delay(0, {"retry-after": "100000"}) == oe.MAX_RETRY_AFTER
    assert [client.delay(n, {}) for n in range(3)] == [2.0, 4.0, 8.0]
    assert client.delay(10, {}) == oe.MAX_RETRY_AFTER  # backoff is capped too
    assert {408, 429, 500, 502, 503, 599} <= oe.RETRY_STATUSES
    assert not {400, 401, 402, 403, 404, 413} & oe.RETRY_STATUSES
    # An error object in a 200 body without a valid status is a provider error.
    assert oe.embedded_error({"code": 429, "message": "slow"}) == (
        429,
        "error in the response body: slow",
    )
    assert oe.embedded_error({"code": "x"})[0] == 502
    assert oe.embedded_error("broken")[0] == 502


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
        lines = [[(line, (10, 10, 20, 20))] for line in FIXTURE_TEXT.split("\n") if line]
        page_xml(data / f"{sid}.xml", lines, size)
        entries.append({"id": sid, "image": f"{sid}.{suffix}", "gt": f"{sid}.xml"})
    return write_manifest(data, entries)


def start_fake_api(log, *args):
    process = subprocess.Popen(
        [sys.executable, str(FIXTURES / "fake_api.py"), str(log), *args],
        stdout=subprocess.PIPE,
        text=True,
    )
    port = int(process.stdout.readline())
    return process, f"http://127.0.0.1:{port}/api/v1/chat/completions"


@pytest.fixture
def fake_api(tmp_path):
    """The fake API's URL and request log."""
    log = tmp_path / "requests.jsonl"
    process, url = start_fake_api(log)
    yield url, log
    process.kill()
    process.wait()


def run_adapter(tmp_path, url, samples, config=CONFIG, params=None, env=None):
    config = dict(config, params={**config["params"], **FAST, **(params or {})})
    del config["resources"]
    env = {"OCRBENCH_OPENROUTER_URL": url, "OPENROUTER_API_KEY": "test-key", **(env or {})}
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
        "markup": ((910, 420), "PNG"),
        "blank": ((300, 200), "PNG"),
        "throttled": ((429, 200), "PNG"),
        "error-in-body": ((430, 200), "PNG"),
        "error-in-choice": ((431, 200), "PNG"),
        "down": ((503, 200), "PNG"),
        "denied": ((401, 200), "PNG"),
        "no-credits": ((402, 200), "PNG"),
        "loop": ((555, 300), "PNG"),
        "token-limit": ((600, 200), "PNG"),
        "filtered": ((601, 200), "PNG"),
        "snapshot": ((776, 200), "PNG"),
        "other-model": ((777, 200), "PNG"),
        "two-choices": ((778, 200), "PNG"),
        "not-json": ((779, 200), "PNG"),
        "tiff": ((900, 420), "TIFF"),
        "after": ((900, 420), "PNG"),
    }
    run, records = run_adapter(tmp_path, url, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "markup": ("success", None),
        "blank": ("success", None),  # an empty transcription is a result, not a failure
        "throttled": ("success", None),  # retried after Retry-After
        "error-in-body": ("success", None),  # a provider error in a 200 body is retried
        "error-in-choice": ("success", None),  # so is one during generation
        "down": ("failure", "engine-error"),
        "denied": ("failure", "engine-error"),
        "no-credits": ("failure", "engine-error"),
        "loop": ("truncated", None),  # scored as produced
        "token-limit": ("truncated", None),
        "filtered": ("failure", "engine-error"),
        "snapshot": ("success", None),
        "other-model": ("failure", "engine-error"),
        "two-choices": ("failure", "engine-error"),
        "not-json": ("failure", "engine-error"),
        "tiff": ("failure", "engine-error"),
        "after": ("success", None),  # the adapter keeps going after failed samples
    }
    assert records["down"]["message"].startswith("HTTP 503 after 3 attempts")
    assert records["denied"]["message"].startswith("HTTP 401 after 1 attempts")  # not retried
    assert records["no-credits"]["message"].startswith("HTTP 402 after 1 attempts")
    assert records["loop"]["message"] == "repetition loop at the end of the answer"
    assert records["token-limit"]["message"] == "the answer reached max_tokens"
    assert "'content_filter' (native 'RECITATION')" in records["filtered"]["message"]
    assert "'google/gemini-2.5-flash', not the pinned" in records["other-model"]["message"]
    assert "expected one choice in the response, got 2" in records["two-choices"]["message"]
    assert "not JSON" in records["not-json"]["message"]
    assert "does not convert images" in records["tiff"]["message"]

    # What was sent: the prompt and the image file unchanged with its media type, the pinned
    # model, provider routing and settings; nothing about the sample.
    sent = requests(log)
    assert len(sent) == 1 + 1 + 1 + 2 + 2 + 2 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # no TIFF
    first = sent[0]
    image_part = first["messages"][0]["content"][1]
    assert image_part == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64"}}
    assert first["bytes"] == (tmp_path / "data" / "page.jpg").stat().st_size
    assert first["authorization"] == "Bearer test-key"
    expected = oe.request_body(MODEL, oe.parse_params({**CONFIG["params"], **FAST}), "")
    assert {k: v for k, v in first.items() if k in expected and k != "messages"} == {
        k: v for k, v in expected.items() if k != "messages"
    }
    assert first["messages"][0]["content"][0] == {
        "type": "text",
        "text": CONFIG["params"]["prompt"],
    }
    dump = json.dumps(sent).replace(json.dumps(CONFIG["params"]["prompt"])[1:-1], "")  # "page."
    assert not [sid for sid in samples if f'"{sid}' in dump or f"{sid}." in dump]

    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "request.json",
        "response.json",
        "page.txt",
        "result.json",
    }
    assert json.loads((page / "output" / "response.json").read_text()) == RESPONSE  # unchanged
    native = (page / "output" / "page.txt").read_text(encoding="utf-8")
    assert native == RESPONSE["choices"][0]["message"]["content"] == FIXTURE_TEXT
    logged = json.loads((page / "output" / "request.json").read_text())
    image_sha = records["page"]["engine_extra"]["image"]["sha256"]
    assert logged["messages"][0]["content"][1]["image_url"]["url"] == (
        f"data:image/jpeg;base64,<sha256 {image_sha}>"
    )
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"]["from"] == "text"
    assert (page / "prediction.txt").read_text(encoding="utf-8") == FIXTURE_TEXT
    # A plain-text answer is scored as given: Markdown-like characters are text.
    markup = run.path / "samples" / "markup" / "attempt-1" / "prediction.txt"
    assert markup.read_text(encoding="utf-8") == MARKDOWN

    extra = records["page"]["engine_extra"]
    assert extra["response_model"] == MODEL
    assert (extra["provider"], extra["service_tier"]) == ("Google AI Studio", "default")
    assert extra["generation_id"] == RESPONSE["id"]
    assert (extra["finish_reason"], extra["native_finish_reason"]) == ("stop", "STOP")
    assert extra["usage"] == RESPONSE["usage"]
    assert extra["cost_usd"] == 0.00097725
    assert extra["image"]["media_type"] == "image/jpeg"
    assert (extra["image"]["width"], extra["image"]["height"]) == (900, 420)
    assert extra["tail_repetition"] is False
    transport = extra["transport"]
    assert [a["status"] for a in transport["attempts"]] == [200]
    assert transport["attempts"][0]["started"].endswith("+00:00")
    assert transport["headers"] == {"x-generation-id": "gen-fake"}
    settings = extra["settings"]
    assert settings["model"] == MODEL
    assert settings["request"]["provider"]["allow_fallbacks"] is False
    assert settings["prompt_sha256"] == oe.fingerprint(CONFIG["params"]["prompt"])
    throttled = records["throttled"]["engine_extra"]["transport"]["attempts"]
    assert [a["status"] for a in throttled] == [429, 200]
    in_body = records["error-in-body"]["engine_extra"]["transport"]["attempts"]
    assert [(a["status"], a.get("error_in_body")) for a in in_body] == [(200, 502), (200, None)]
    in_choice = records["error-in-choice"]["engine_extra"]["transport"]["attempts"]
    assert [(a["status"], a.get("error_in_body")) for a in in_choice] == [(200, 500), (200, None)]
    assert "Internal Server Error" in in_choice[0]["error"]
    down = records["down"]["engine_extra"]["transport"]["attempts"]
    assert [a["status"] for a in down] == [503, 503, 503]
    assert records["blank"]["engine_extra"]["empty"] is True
    assert (run.path / "samples" / "blank" / "attempt-1" / "prediction.txt").read_text() == ""
    assert records["snapshot"]["engine_extra"]["response_model"] == f"{MODEL}-20260902"

    # The body of every failed attempt is kept unchanged, also after a successful retry.
    def kept_bodies(sid):
        output = run.path / "samples" / sid / "attempt-1" / "output"
        return sorted(p.name for p in output.glob("attempt-*-response.*"))

    assert kept_bodies("page") == []
    assert kept_bodies("down") == [f"attempt-{n}-response.json" for n in (1, 2, 3)]
    assert kept_bodies("denied") == ["attempt-1-response.json"]
    assert kept_bodies("not-json") == ["attempt-1-response.txt"]
    assert kept_bodies("error-in-choice") == ["attempt-1-response.json"]
    output = run.path / "samples" / "error-in-choice" / "attempt-1" / "output"
    failed_choice = json.loads((output / "attempt-1-response.json").read_text())["choices"][0]
    assert (failed_choice["finish_reason"], failed_choice["message"]["content"]) == (
        "error",
        "The image is",
    )
    assert json.loads((output / "response.json").read_text()) == RESPONSE
    # The response of a rejected answer is kept for inspection.
    kept = run.path / "samples" / "other-model" / "attempt-1" / "output" / "response.json"
    assert json.loads(kept.read_text())["model"] == "google/gemini-2.5-flash"

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


def test_markdown_answers_are_converted_on_the_host(tmp_path, fake_api):
    url, _ = fake_api
    params = {"answer_format": "markdown"}
    run, records = run_adapter(tmp_path, url, {"markup": ((910, 420), "PNG")}, params=params)
    record = records["markup"]
    assert (record["status"], record["primary"]) == ("success", "page.md")
    page = run.path / "samples" / "markup" / "attempt-1"
    assert (page / "output" / "page.md").read_text(encoding="utf-8") == MARKDOWN  # native
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": 1}
    assert (page / "prediction.txt").read_text(encoding="utf-8") == FIXTURE_TEXT


def test_redirects_are_refused_and_never_get_the_key(tmp_path):
    """A redirect to another server fails the sample; that server receives no request, so no
    Authorization header."""
    other_log = tmp_path / "other.jsonl"
    other, other_url = start_fake_api(other_log)
    log = tmp_path / "requests.jsonl"
    api, url = start_fake_api(log, other_url.replace("127.0.0.1", "localhost"))
    try:
        _, records = run_adapter(tmp_path, url, {"moved": ((302, 200), "PNG")})
    finally:
        for process in (api, other):
            process.kill()
            process.wait()
    assert not other_log.exists()  # checked first: the redirect target got nothing
    record = records["moved"]
    assert (record["status"], record["reason"]) == ("failure", "engine-error")
    assert record["message"].startswith("HTTP 302 after 1 attempts: redirect to")
    assert "refused" in record["message"]
    assert len(requests(log)) == 1


@pytest.mark.parametrize(
    ("config", "env", "message"),
    [
        (dict(CONFIG, params={**CONFIG["params"], "psm": 3}), {}, "unknown parameters: psm"),
        (CONFIG, {"OPENROUTER_API_KEY": ""}, "OPENROUTER_API_KEY is not set"),
        (dict(CONFIG, model="openrouter/auto"), {}, "router, alias or variant"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, fake_api, config, env, message):
    url, log = fake_api
    samples = {"a": ((900, 420), "PNG"), "b": ((300, 200), "PNG")}
    _, records = run_adapter(tmp_path, url, samples, config=config, env=env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())
    assert not log.exists()  # no request was sent


@pytest.fixture(scope="module")
def openrouter_image():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.fail("OPENROUTER_API_KEY is not set")
    subprocess.run(["make", "-C", str(ROOT), "docker-openrouter"], check=True)


@pytest.mark.docker
@pytest.mark.api
@pytest.mark.parametrize("name", PRESETS)
def test_api_smoke_benchmark(tmp_path, openrouter_image, name):
    manifest = smoke_dataset(tmp_path / "data")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", str(PRESET_FILES[name]), "--dataset", str(manifest)]
    assert main([*args, "--runs-dir", runs, "--run-id", "smoke"]) == 1  # needs --allow-network
    assert main([*args, "--runs-dir", runs, "--run-id", "smoke", "--allow-network"]) == 0
    run = load_run(tmp_path / "runs" / "smoke")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    extra = records["page-1"]["engine_extra"]
    model, _, provider_name, _, _ = cohort_entry(name)
    assert oe.model_matches(model, extra["response_model"])
    assert extra["provider"] == provider_name
    assert extra["cost_usd"] > 0
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["para-1"]["chars"]["rate"] < 0.1
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()
