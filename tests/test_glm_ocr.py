"""GLM-OCR adapter (engines/glm-ocr): parameters, prompt, generation and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and
``transformers`` modules (tests/fixtures/glm-ocr/fake) that "generate" fixed outputs, including
a hand-written Markdown page with historic spelling, a formula and a table
(tests/fixtures/glm-ocr/page.md), the empty fenced block the real model writes for a blank page
and a newspaper page whose output never ends. They also evaluate and report that run. The opt-in
GPU test builds the engine image and runs the real model from the weights cache through
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
ENGINE_DIR = ROOT / "engines" / "glm-ocr"
FIXTURES = Path(__file__).parent / "fixtures" / "glm-ocr"
ADAPTER_PATH = ENGINE_DIR / "glm_ocr_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()

_spec = importlib.util.spec_from_file_location("glm_ocr_engine", ADAPTER_PATH)
glm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm)

FIXTURE_LINES = [
    "Der Hinkende Bote",
    "Es iſt kund und zu wiſſen, daß der Fluß im Jahre 1784 über die Ufer trat.",
    "Ein zweyter Abſatz mit mehr Text und einer Formel E = mc^2",
    "Jahr\tExemplare",
    "1784\t2026",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{glm.REPOSITORY}@{glm.REVISION}"
    assert glm.PINNED_DIR.endswith(glm.REVISION)
    names = CHECKSUMS[1::2]
    assert "model.safetensors" in names
    assert "generation_config.json" in names  # its end tokens are used
    assert "preprocessor_config.json" in names
    assert "chat_template.jinja" in names
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_prompt_is_the_model_cards_text_recognition_prompt():
    assert glm.PROMPT == "Text Recognition:"
    assert glm.CONVERSATION[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "Text Recognition:"},
    ]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "gpu:0"}, "device"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 100.0}, "max_new_tokens"),
        ({"max_new_tokens": 131072}, "less than the context"),
        ({"temperature": 0.7}, "unknown parameters: temperature"),  # greedy only
        ({"max_pixels": 12543}, "max_pixels"),
        ({"max_pixels": True}, "max_pixels"),
        ({"repetition_penalty": 0.9}, "repetition_penalty"),
        ({"repetition_penalty": 2.5}, "repetition_penalty"),
        ({"repetition_penalty": True}, "repetition_penalty"),
        ({"repetition_penalty": "1.1"}, "repetition_penalty"),
        ({"model_dir": "/cache/models/x"}, "model_dir"),
        ({"model_dir": "models/../../etc"}, "model_dir"),
        ({"model_dir": ""}, "model_dir"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(glm.AdapterError, match=message):
        glm.parse_params(params)


def test_params_defaults():
    assert glm.parse_params({"max_new_tokens": None}) == glm.DEFAULT_PARAMS
    assert glm.DEFAULT_PARAMS["model_dir"] == glm.PINNED_DIR
    # The official SDK's decoding and the checkpoint's preprocessor_config.json pixel range.
    assert glm.DEFAULT_PARAMS["max_new_tokens"] == 8192
    assert glm.DEFAULT_PARAMS["repetition_penalty"] == 1.1
    assert glm.DEFAULT_PARAMS["max_pixels"] == 9633792
    assert glm.MIN_PIXELS == 112 * 112
    params = glm.parse_params({"max_new_tokens": 100, "device": "cpu", "repetition_penalty": 1})
    assert (params["max_new_tokens"], params["device"]) == (100, "cpu")
    assert params["repetition_penalty"] == 1.0
    assert isinstance(params["repetition_penalty"], float)


def test_undecodable_image_is_an_adapter_error(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(glm.AdapterError, match="cannot decode image x.png"):
        glm.read_image(tmp_path / "x.png")
    Image.new("LA", (4, 3)).save(tmp_path / "la.png")
    assert glm.read_image(tmp_path / "la.png").mode == "RGB"


def model_cache(root, corrupt=False, extra=None):
    """A cache with fake model files at the pinned path, and checksums that match them; ``extra``
    adds a file that is not in the checksums."""
    directory = root / "cache" / glm.PINNED_DIR
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "model.safetensors").write_text("changed", encoding="utf-8")
    if extra:
        (directory / extra).parent.mkdir(parents=True, exist_ok=True)
        (directory / extra).write_text("{}", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_GLM_OCR_CACHE": str(root / "cache"),
        "OCRBENCH_GLM_OCR_CHECKSUMS": str(checksums),
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
        elif size[0] in (900, 901):
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
        "page2": (901, 420),
        "tiny": (60, 40),
    }
    run, records = run_adapter(tmp_path, sizes)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # an empty fenced block: a genuine empty result
        "boom": ("failure", "engine-error"),
        "news": ("truncated", None),  # stopped at max_new_tokens, scored as produced
        "page2": ("success", None),  # the adapter keeps going after failed samples
        "tiny": ("success", None),
    }
    assert "RuntimeError: CUDA out of memory" in records["boom"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {"page.md", "result.json"}
    native = (page / "output" / "page.md").read_text(encoding="utf-8")
    assert native == (FIXTURES / "page.md").read_text(encoding="utf-8")  # unchanged Markdown
    # Converted on the host: no Markdown markup in the text, historic spelling (long s, ß, y)
    # unchanged, regions without positions.
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
    assert extra["model_image"] == {"width": 896, "height": 420}  # whole 28-pixel units
    assert extra["image_tokens"] == 32 * 15  # 28 x 28 pixels each
    assert extra["prompt_tokens"] == 32 * 15 + 5
    assert extra["max_new_tokens"] == 8192
    assert extra["gpu_peak_memory_mb"] == 3072
    assert extra["environment"]["transformers"] == "5.19.0+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["verified_official_checkpoint"] is True
    assert (models["repository"], models["revision"]) == (glm.REPOSITORY, glm.REVISION)
    assert models["model_dir"] == glm.PINNED_DIR
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert models["prompt"] == glm.CONVERSATION
    assert extra["params"] == glm.DEFAULT_PARAMS
    # <|user|>, the start of the next turn, also ends generation.
    assert records["page2"]["engine_extra"]["stop_reason"] == "end-token"
    assert records["page2"]["engine_extra"]["generated_tokens"] == 1
    # A small image is enlarged until both frames have at least 112 x 112 pixels.
    assert records["tiny"]["engine_extra"]["model_image"] == {"width": 112, "height": 84}
    assert records["blank"]["engine_extra"]["model_image"] == {"width": 308, "height": 196}

    news = records["news"]
    assert news["message"] == "generation stopped at max_new_tokens (8192 tokens)"
    assert news["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert news["engine_extra"]["generated_tokens"] == 8192
    # Scaled down to at most 9,633,792 / 2 pixels: 64 x 96 units, the 6,144 image tokens of the
    # model's design.
    assert news["engine_extra"]["model_image"] == {"width": 1792, "height": 2688}
    assert news["engine_extra"]["image_tokens"] == 64 * 96 == 6144
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "output" / "page.md").read_text(encoding="utf-8") == "```markdown\n\n```"
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


def test_max_pixels_repetition_penalty_and_context_limit(tmp_path):
    params = {"device": "cpu", "max_pixels": 2_000_000, "repetition_penalty": 1.0}
    _, records = run_adapter(tmp_path, {"news": (2000, 3000)}, params, {"FAKE_TORCH_GPUS": "0"})
    extra = records["news"]["engine_extra"]
    # 2000 x 3000 pixels scaled to at most a million (2,000,000 for both frames): 29 x 43 units.
    assert extra["model_image"] == {"width": 812, "height": 1204}
    assert extra["image_tokens"] == 29 * 43
    assert extra["params"]["repetition_penalty"] == 1.0
    assert "gpu" not in extra["environment"]
    assert "gpu_peak_memory_mb" not in extra

    params = {"device": "cpu", "max_new_tokens": 131000}
    _, records = run_adapter(
        tmp_path / "context", {"news": (2000, 3000)}, params, {"FAKE_TORCH_GPUS": "0"}
    )
    extra = records["news"]["engine_extra"]
    assert extra["prompt_tokens"] == 64 * 96 + 5
    assert extra["max_new_tokens"] == 131072 - extra["prompt_tokens"]
    assert extra["stop_reason"] == "context"
    assert records["news"]["status"] == "truncated"
    assert "model's context" in records["news"]["message"]


@pytest.mark.parametrize(
    ("content", "extra", "official"),
    [
        ("other weights", None, False),
        (None, None, True),
        # Extra files may change what transformers loads, for example the chat template.
        (None, "additional_chat_templates/other.jinja", False),
    ],
)
def test_a_custom_model_dir_claims_the_official_checkpoint_only_when_verified(
    tmp_path, content, extra, official
):
    custom = tmp_path / "cache" / "models" / "custom"
    custom.mkdir(parents=True)
    for name in ("config.json", "model.safetensors"):  # None: a copy of the official files
        (custom / name).write_text(content or f"fake {name}", encoding="utf-8")
    if extra:
        (custom / extra).parent.mkdir(parents=True, exist_ok=True)
        (custom / extra).write_text("{}", encoding="utf-8")
    params = CONFIG["params"] | {"model_dir": "models/custom"}
    _, records = run_adapter(tmp_path, {"page": (900, 420)}, params)
    assert records["page"]["status"] == "success"
    models = records["page"]["engine_extra"]["models"]
    assert models["verified_official_checkpoint"] is official
    assert ("repository" in models, "revision" in models) == (official, official)
    assert models["model_dir"] == "models/custom"
    digest = hashlib.sha256((content or "fake model.safetensors").encode()).hexdigest()
    assert models["files"]["model.safetensors"] == digest  # hashes are recorded either way
    if extra:
        assert extra in models["files"]


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({"model_dir": "models/other"}, {}, "does not exist; download the weights"),
        ({}, "corrupt", "not matching models.sha256: ['model.safetensors']"),
        ({}, "sub/x.json", "not in models.sha256: ['sub/x.json']"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "corrupt":
        env = model_cache(tmp_path / "corrupt", corrupt=True)
    elif isinstance(env, str):
        env = model_cache(tmp_path / "extra", extra=env)
    _, records = run_adapter(tmp_path, {"a": (900, 420), "b": (300, 200)}, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "glm_ocr_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_GLM_OCR_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{glm.REPOSITORY}/resolve/{glm.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / glm.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / glm.PINNED_DIR).iterdir()) == [
        "model.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def glm_ocr_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / glm.PINNED_DIR / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make glm-ocr-model first")
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
def test_gpu_smoke_benchmark(tmp_path, glm_ocr_image):
    cache = str(glm_ocr_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "glm-ocr"]) == 0
    run = load_run(tmp_path / "runs" / "glm-ocr")
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

    # A small max_new_tokens truncates the page; the partial Markdown is kept and scored.
    limited = tmp_path / "engine.toml"
    config = Path(engine).read_text(encoding="utf-8")
    limited.write_text(config.replace("max_new_tokens = 8192", "max_new_tokens = 16"), "utf-8")
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["message"] == "generation stopped at max_new_tokens (16 tokens)"
    assert page["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert page["engine_extra"]["generated_tokens"] == 16
    attempt = "samples/page-1/attempt-1/output/page.md"
    partial = (short_run.path / attempt).read_text(encoding="utf-8")
    assert partial
    assert (run.path / attempt).read_text(encoding="utf-8").startswith(partial)
    assert main(["evaluate", str(short_run.path)]) == 0
    scores = json.loads((short_run.path / "evaluations" / "1" / "scores.json").read_text())
    short_rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < short_rates["page-1"] < 1  # scored as produced, not as empty

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make glm-ocr-model" in r["message"] for r in records)
