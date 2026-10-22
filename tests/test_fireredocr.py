"""FireRed-OCR adapter (engines/fireredocr): parameters, prompt, generation and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and
``transformers`` modules (tests/fixtures/fireredocr/fake) that "generate" fixed outputs, including
a hand-written Markdown page with an HTML table (tests/fixtures/fireredocr/page.md) and a
newspaper page whose output never ends. They also evaluate and report that run. The opt-in GPU
test builds the engine image and runs the real model from the weights cache through
``ocrbench run``, ``evaluate`` and ``report``.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import page_xml, smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "fireredocr"
FIXTURES = Path(__file__).parent / "fixtures" / "fireredocr"
ADAPTER_PATH = ENGINE_DIR / "fireredocr_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
# SHA-256 of the PROMPT string in conv_for_infer.py of github.com/FireRedTeam/FireRed-OCR at
# commit 9a368c297c8e54ab3c4c66f403a5d3b4e792552c.
UPSTREAM_PROMPT_SHA256 = "e9ecb7425bd2acee69f034de04f431004afddf23eadadc7d73bec65724df93d5"

_spec = importlib.util.spec_from_file_location("fireredocr_engine", ADAPTER_PATH)
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)

FIXTURE_LINES = [
    "The Daily Chronicle",
    "The quick brown fox jumps over the lazy dog near the river.",
    "A second paragraph with more text and a formula ( E = mc^2 ).",
    "Year\tCopies",
    "1923\t2026",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{fr.REPOSITORY}@{fr.REVISION}"
    assert fr.PINNED_DIR.endswith(fr.REVISION)
    names = CHECKSUMS[1::2]
    assert "model.safetensors" in names
    assert "preprocessor_config.json" in names
    assert "chat_template.jinja" in names
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_prompt_is_the_official_one():
    assert hashlib.sha256(fr.PROMPT.encode()).hexdigest() == UPSTREAM_PROMPT_SHA256
    assert fr.PROMPT.startswith("You are an AI assistant specialized in converting PDF images")
    assert fr.CONVERSATION[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": fr.PROMPT},
    ]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "gpu:0"}, "device"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 100.0}, "max_new_tokens"),
        ({"max_new_tokens": 262144}, "less than the context"),
        ({"seed": -1}, "seed"),
        ({"top_k": 0}, "top_k"),
        ({"max_pixels": 65535}, "max_pixels"),
        ({"max_pixels": True}, "max_pixels"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": "0.2"}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"top_p": 0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
        ({"model_dir": "/cache/models/x"}, "model_dir"),
        ({"model_dir": "models/../../etc"}, "model_dir"),
        ({"model_dir": ""}, "model_dir"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(fr.AdapterError, match=message):
        fr.parse_params(params)


def test_params_defaults():
    assert fr.parse_params({"seed": None}) == fr.DEFAULT_PARAMS
    assert fr.DEFAULT_PARAMS["model_dir"] == fr.PINNED_DIR
    assert fr.DEFAULT_PARAMS["temperature"] == 0  # greedy, as the official vLLM script
    params = fr.parse_params({"temperature": 0.7, "top_p": 1, "device": "cpu"})
    assert (params["temperature"], params["top_p"], params["device"]) == (0.7, 1, "cpu")


def test_undecodable_image_is_an_adapter_error(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(fr.AdapterError, match="cannot decode image x.png"):
        fr.read_image(tmp_path / "x.png")
    Image.new("LA", (4, 3)).save(tmp_path / "la.png")
    assert fr.read_image(tmp_path / "la.png").mode == "RGB"


def model_cache(root, corrupt=False):
    """A cache with fake model files at the pinned path, and checksums that match them."""
    directory = root / "cache" / fr.PINNED_DIR
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "model.safetensors").write_text("changed", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_FIREREDOCR_CACHE": str(root / "cache"),
        "OCRBENCH_FIREREDOCR_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000


def make_dataset(root, sizes):
    data = root / "data"
    data.mkdir()
    samples = []
    for sid, size in sizes.items():
        Image.new("L", size, 255).save(data / f"{sid}.png")
        if size[0] == 2000:  # a newspaper page whose reference has 2,000 lines
            lines = [_newspaper_line(n) for n in range(1, NEWSPAPER_LINES + 1)]
            regions = [[(text, (10, n, 20, n + 1)) for n, text in enumerate(lines)]]
        elif size[0] == 900:
            lines = [line.replace("\t", " ") for line in FIXTURE_LINES]
            regions = [[(text, (10, 10, 20, 20))] for text in lines]
        else:
            regions = []
        page_xml(data / f"{sid}.xml", regions, size)
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def _newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def run_adapter(tmp_path, sizes, params=None, env=None):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"])
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {"PYTHONPATH": str(FIXTURES / "fake"), **model_cache(tmp_path), **(env or {})}
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, sizes),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in sizes}


def test_adapter_follows_the_engine_protocol(tmp_path):
    sizes = {
        "page": (900, 420),
        "blank": (300, 200),
        "boom": (666, 400),
        "news": (2000, 3000),
        "page2": (900, 420),
    }
    run, records = run_adapter(tmp_path, sizes)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # only an end token: a genuine empty result
        "boom": ("failure", "engine-error"),
        "news": ("truncated", None),  # stopped at max_new_tokens, scored as produced
        "page2": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: CUDA out of memory" in records["boom"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {"page.md", "result.json"}
    native = (page / "output" / "page.md").read_text(encoding="utf-8")
    assert native == (FIXTURES / "page.md").read_text(encoding="utf-8")  # unchanged Markdown
    # Converted on the host: no Markdown or HTML markup in the text, regions without positions.
    text = (page / "prediction.txt").read_text(encoding="utf-8")
    assert text == "\n\n".join([*FIXTURE_LINES[:3], "\n".join(FIXTURE_LINES[3:])])
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"]["from"] == "markdown"
    assert [r["type"] for r in prediction["regions"]] == ["heading", "text", "text", "table"]
    assert not any("polygon" in r or "lines" in r for r in prediction["regions"])
    assert "image" not in prediction

    extra = records["page"]["engine_extra"]
    assert extra["stop_reason"] == "end-token"
    assert extra["generated_tokens"] == len(native)  # one fake token per character
    assert extra["model_image"] == {"width": 896, "height": 416}  # whole 32-pixel units
    assert extra["image_tokens"] == 28 * 13  # 32 x 32 pixels each
    assert extra["prompt_tokens"] == 28 * 13 + 5
    assert extra["max_new_tokens"] == 8192
    assert extra["gpu_peak_memory_mb"] == 3072
    assert extra["environment"]["transformers"] == "5.19.0+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["verified_official_checkpoint"] is True
    assert (models["repository"], models["revision"]) == (fr.REPOSITORY, fr.REVISION)
    assert models["model_dir"] == fr.PINNED_DIR
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert models["prompt"] == fr.CONVERSATION
    assert models["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["params"] == fr.DEFAULT_PARAMS
    # A small image is enlarged to at least 65,536 pixels, as by the model's own processor.
    assert records["blank"]["engine_extra"]["model_image"] == {"width": 320, "height": 224}

    news = records["news"]
    assert news["message"] == "generation stopped at max_new_tokens (8192 tokens)"
    assert news["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert news["engine_extra"]["generated_tokens"] == 8192
    assert news["engine_extra"]["model_image"] == {"width": 1984, "height": 3008}
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""

    # Common metrics on text-only output, including the truncated 2,000-line newspaper page.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["page"]["chars"]["rate"] < 0.05  # the table cells are tab-separated
    news = by_id["news"]
    assert news["status"] == "truncated"
    assert news["scored_as"] == "prediction"
    assert news["flexible_units"]["reference"] == NEWSPAPER_LINES
    assert 0 < news["chars"]["rate"] < 1  # only the beginning of the page was transcribed
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


def test_sampling_max_pixels_and_context_limit(tmp_path):
    params = {"device": "cpu", "temperature": 0.7, "max_pixels": 1_000_000}
    _, records = run_adapter(tmp_path, {"news": (2000, 3000)}, params, {"FAKE_TORCH_GPUS": "0"})
    extra = records["news"]["engine_extra"]
    # 2000 x 3000 pixels scaled to at most a million: 25 x 38 units of 32 pixels.
    assert extra["model_image"] == {"width": 800, "height": 1216}
    assert extra["image_tokens"] == 25 * 38
    assert extra["params"]["top_k"] == 20  # sampled: the fake model checks top_k is passed
    assert "gpu" not in extra["environment"]
    assert "gpu_peak_memory_mb" not in extra

    params = {"device": "cpu", "max_new_tokens": 262000}
    _, records = run_adapter(
        tmp_path / "context", {"news": (2000, 3000)}, params, {"FAKE_TORCH_GPUS": "0"}
    )
    extra = records["news"]["engine_extra"]
    assert extra["prompt_tokens"] == 62 * 94 + 5
    assert extra["max_new_tokens"] == 262144 - extra["prompt_tokens"]
    assert extra["stop_reason"] == "context"
    assert records["news"]["status"] == "truncated"
    assert "model's context" in records["news"]["message"]


@pytest.mark.parametrize(("content", "official"), [("other weights", False), (None, True)])
def test_a_custom_model_dir_claims_the_official_checkpoint_only_when_verified(
    tmp_path, content, official
):
    custom = tmp_path / "cache" / "models" / "custom"
    custom.mkdir(parents=True)
    for name in ("config.json", "model.safetensors"):  # None: a copy of the official files
        (custom / name).write_text(content or f"fake {name}", encoding="utf-8")
    params = CONFIG["params"] | {"model_dir": "models/custom"}
    _, records = run_adapter(tmp_path, {"page": (900, 420)}, params)
    assert records["page"]["status"] == "success"
    models = records["page"]["engine_extra"]["models"]
    assert models["verified_official_checkpoint"] is official
    assert ("repository" in models, "revision" in models) == (official, official)
    assert models["model_dir"] == "models/custom"
    digest = hashlib.sha256((content or "fake model.safetensors").encode()).hexdigest()
    assert models["files"]["model.safetensors"] == digest  # hashes are recorded either way


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({"model_dir": "models/other"}, {}, "does not exist; download the weights"),
        ({}, "corrupt", "not matching models.sha256: ['model.safetensors']"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "corrupt":
        env = model_cache(tmp_path / "corrupt", corrupt=True)
    _, records = run_adapter(tmp_path, {"a": (900, 420), "b": (300, 200)}, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_FIREREDOCR_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{fr.REPOSITORY}/resolve/{fr.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / fr.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / fr.PINNED_DIR).iterdir()) == [
        "model.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def fireredocr_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / fr.PINNED_DIR / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make fireredocr-model first")
    subprocess.run(
        ["docker", "build", "-q", "-t", CONFIG["image"], str(ENGINE_DIR)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=True,
    )
    return cache


@pytest.mark.docker
@pytest.mark.gpu
def test_gpu_smoke_benchmark(tmp_path, fireredocr_image):
    cache = str(fireredocr_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "fireredocr"]) == 0
    run = load_run(tmp_path / "runs" / "fireredocr")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"]["transformers"] == "5.19.0"
    assert extra["environment"]["torch"] == "2.14.1+cu130"
    assert extra["environment"]["gpu"]
    assert (
        extra["models"]["files"]["model.safetensors"]
        == CHECKSUMS[CHECKSUMS.index("model.safetensors") - 1]
    )
    assert extra["models"]["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["stop_reason"] == "end-token"
    assert extra["gpu_peak_memory_mb"] > 0

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

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make fireredocr-model" in r["message"] for r in records)
