"""olmOCR-2 adapter (engines/olmocr2): parameters, prompt, image scaling, front matter and the
engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and
``transformers`` modules (tests/fixtures/olmocr2/fake) that "generate" fixed outputs, including
a hand-written page with a front matter, a figure and an HTML table
(tests/fixtures/olmocr2/output.md) and a newspaper page whose output never ends. They also
evaluate and report that run. The opt-in GPU test builds the engine image and runs the real model
from the weights cache through ``ocrbench run``, ``evaluate`` and ``report``.
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
ENGINE_DIR = ROOT / "engines" / "olmocr2"
FIXTURES = Path(__file__).parent / "fixtures" / "olmocr2"
ADAPTER_PATH = ENGINE_DIR / "olmocr2_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
# SHA-256 of build_no_anchoring_v4_yaml_prompt() in olmocr/prompts/prompts.py of
# github.com/allenai/olmocr at v0.4.27 (commit 1e139a5ea61f1668164e6b63357d7284a8391615).
UPSTREAM_PROMPT_SHA256 = "c12f21aca8c6c5599cea6176edd974b7ea6a05cf451497abd6370dd449db3979"

_spec = importlib.util.spec_from_file_location("olmocr2_engine", ADAPTER_PATH)
olmo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(olmo)

FIXTURE_LINES = [
    "The Daily Chronicle",
    "The quick brown fox jumps over the lazy dog near the river.",
    "A second paragraph with more text and a formula ( E = mc^2 ).",
    "Year\tCopies",
    "1923\t2026",
]
FRONT_MATTER = {
    "primary_language": "en",
    "is_rotation_valid": True,
    "rotation_correction": 0,
    "is_table": False,
    "is_diagram": False,
}


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{olmo.REPOSITORY}@{olmo.REVISION}"
    assert olmo.PINNED_DIR.endswith(olmo.REVISION)
    names = CHECKSUMS[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == [
        f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)
    ]
    assert "preprocessor_config.json" in names
    assert "chat_template.jinja" in names
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_prompt_is_the_official_one():
    assert hashlib.sha256(olmo.PROMPT.encode()).hexdigest() == UPSTREAM_PROMPT_SHA256
    assert olmo.PROMPT.startswith("Attached is one page of a document that you must process.")
    assert olmo.CONVERSATION[0]["content"] == [  # the text before the image, as in the pipeline
        {"type": "text", "text": olmo.PROMPT},
        {"type": "image"},
    ]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "gpu:0"}, "device"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 100.0}, "max_new_tokens"),
        ({"max_new_tokens": 128000}, "less than the context"),
        ({"seed": -1}, "seed"),
        ({"target_longest_image_dim": 55}, "target_longest_image_dim"),
        ({"target_longest_image_dim": True}, "target_longest_image_dim"),
        ({"temperature": -0.1}, "temperature"),
        ({"temperature": "0.2"}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"repetition_penalty": 0}, "repetition_penalty"),
        ({"repetition_penalty": float("inf")}, "repetition_penalty"),
        ({"model_dir": "/cache/models/x"}, "model_dir"),
        ({"model_dir": "models/../../etc"}, "model_dir"),
        ({"model_dir": ""}, "model_dir"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(olmo.AdapterError, match=message):
        olmo.parse_params(params)


def test_params_defaults():
    assert olmo.parse_params({"seed": None}) == olmo.DEFAULT_PARAMS
    assert olmo.DEFAULT_PARAMS["model_dir"] == olmo.PINNED_DIR
    # The olmocr pipeline's first attempt, and the checkpoint's repetition penalty.
    assert olmo.DEFAULT_PARAMS["target_longest_image_dim"] == 1288
    assert olmo.DEFAULT_PARAMS["max_new_tokens"] == 8000
    assert olmo.DEFAULT_PARAMS["temperature"] == 0.1
    assert olmo.DEFAULT_PARAMS["repetition_penalty"] == 1.05
    params = olmo.parse_params({"temperature": 0, "repetition_penalty": 1, "device": "cpu"})
    assert (params["temperature"], params["repetition_penalty"], params["device"]) == (0, 1, "cpu")


def test_images_are_scaled_to_the_longest_side(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(olmo.AdapterError, match="cannot decode image x.png"):
        olmo.read_image(tmp_path / "x.png", 1288)
    Image.new("LA", (4000, 3000)).save(tmp_path / "la.png")
    image, original = olmo.read_image(tmp_path / "la.png", 1288)
    assert (image.mode, image.size, original) == (
        "RGB",
        (1288, 966),
        {"width": 4000, "height": 3000},
    )
    Image.new("RGB", (300, 600)).save(tmp_path / "small.png")
    assert olmo.read_image(tmp_path / "small.png", 1288)[0].size == (644, 1288)  # enlarged
    Image.new("RGB", (1288, 20)).save(tmp_path / "exact.png")
    assert olmo.read_image(tmp_path / "exact.png", 1288)[0].size == (1288, 20)


@pytest.mark.parametrize(
    ("output", "front_matter", "text", "problem"),
    [
        (
            "---\nprimary_language: en\nis_rotation_valid: True\nrotation_correction: 0\n"
            "is_table: False\nis_diagram: False\n---\n# Title\n\nText.\n",
            FRONT_MATTER,
            "# Title\n\nText.",
            None,
        ),
        (  # a blank page: the front matter only; Norwegian "no" is YAML's false
            "---\nprimary_language: no\nis_rotation_valid: False\nrotation_correction: 90\n"
            "is_table: False\nis_diagram: False\n---",
            FRONT_MATTER
            | {"primary_language": False, "is_rotation_valid": False}
            | {"rotation_correction": 90},
            "",
            None,
        ),
        ("Just text.\n", None, "Just text.", "no front matter"),
        ("---\nprimary_language: en\n", None, "---\nprimary_language: en", "no front matter"),
        ("---\nkey: [unclosed\n---\nText", None, "---\nkey: [unclosed\n---\nText", "invalid YAML"),
        ("---\n- a list\n---\nText", None, "Text", "not a mapping"),
        (
            "---\nprimary_language: en\ndate: 2025-10-22\n---\nText",
            {"primary_language": "en", "date": "2025-10-22"},
            "Text",
            "missing: ['is_rotation_valid', 'rotation_correction', 'is_table', 'is_diagram'], "
            "unexpected: ['date']",
        ),
    ],
)
def test_front_matter_is_split_as_olmocr_does(output, front_matter, text, problem):
    got_front_matter, got_text, got_problem = olmo.split_front_matter(output)
    assert (got_front_matter, got_text) == (front_matter, text)
    assert (got_problem is None) if problem is None else (problem in got_problem)
    json.dumps(got_front_matter)  # stored in result.json: a date is kept as its string


def model_cache(root, corrupt=False):
    """A cache with fake model files at the pinned path, and checksums that match them."""
    directory = root / "cache" / olmo.PINNED_DIR
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model-00001-of-00004.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "model-00001-of-00004.safetensors").write_text("changed", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_OLMOCR2_CACHE": str(root / "cache"),
        "OCRBENCH_OLMOCR2_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000
# The fake model chooses its output by the gray level of the image (fake/README.md).
GRAY = {"page": 255, "blank": 250, "boom": 200, "plain": 150, "news": 100}


def make_dataset(root, samples):
    data = root / "data"
    data.mkdir()
    entries = []
    for sid, (kind, size) in samples.items():
        Image.new("L", size, GRAY.get(kind, 0)).save(data / f"{sid}.png")
        if kind == "news":  # a newspaper page whose reference has 2,000 lines
            lines = [_newspaper_line(n) for n in range(1, NEWSPAPER_LINES + 1)]
            regions = [[(text, (10, n, 20, n + 1)) for n, text in enumerate(lines)]]
        elif kind == "page":
            lines = [line.replace("\t", " ") for line in FIXTURE_LINES]
            regions = [[(text, (10, 10, 20, 20))] for text in lines]
        else:
            regions = []
        page_xml(data / f"{sid}.xml", regions, size)
        entries.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, entries)


def _newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def run_adapter(tmp_path, samples, params=None, env=None):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"])
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {"PYTHONPATH": str(FIXTURES / "fake"), **model_cache(tmp_path), **(env or {})}
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, samples),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in samples}


def test_adapter_follows_the_engine_protocol(tmp_path):
    samples = {
        "page": ("page", (900, 420)),
        "blank": ("blank", (300, 200)),
        "boom": ("boom", (666, 400)),
        "plain": ("plain", (500, 700)),
        "news": ("news", (2000, 3000)),
        "page2": ("page", (900, 420)),
    }
    run, records = run_adapter(tmp_path, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # a front matter only: a genuine empty result
        "boom": ("failure", "engine-error"),
        "plain": ("success", None),  # without a front matter, the whole output is the text
        "news": ("truncated", None),  # stopped at max_new_tokens, scored as produced
        "page2": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: CUDA out of memory" in records["boom"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "output.md",
        "page.md",
        "result.json",
    }
    native = (page / "output" / "output.md").read_text(encoding="utf-8")
    assert native == (FIXTURES / "output.md").read_text(encoding="utf-8")  # unchanged output
    markdown = (page / "output" / "page.md").read_text(encoding="utf-8")
    assert markdown == native.split("---\n", 2)[2].strip()  # the front matter is not text
    # Converted on the host: no Markdown or HTML markup and no figure alt text in the text,
    # regions without positions.
    text = (page / "prediction.txt").read_text(encoding="utf-8")
    assert text == "\n\n".join([*FIXTURE_LINES[:3], "\n".join(FIXTURE_LINES[3:])])
    assert "woodcut" not in text
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"]["from"] == "markdown"
    assert [r["type"] for r in prediction["regions"]] == ["heading", "text", "text", "table"]
    assert not any("polygon" in r or "lines" in r for r in prediction["regions"])
    assert "image" not in prediction

    extra = records["page"]["engine_extra"]
    assert extra["front_matter"] == FRONT_MATTER
    assert "front_matter_error" not in extra
    assert extra["stop_reason"] == "end-token"
    assert extra["generated_tokens"] == len(native)  # one fake token per character
    # 900 x 420 scaled to 1288 x 601, then whole 28-pixel units.
    assert extra["source_image"] == {"width": 900, "height": 420}
    assert extra["model_image"] == {"width": 1288, "height": 588}
    assert extra["image_tokens"] == 46 * 21  # 28 x 28 pixels each
    assert extra["prompt_tokens"] == 46 * 21 + 5
    assert extra["max_new_tokens"] == 8000
    assert extra["gpu_peak_memory_mb"] == 3072
    assert extra["environment"]["transformers"] == "5.19.0+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["verified_official_checkpoint"] is True
    assert (models["repository"], models["revision"]) == (olmo.REPOSITORY, olmo.REVISION)
    assert models["model_dir"] == olmo.PINNED_DIR
    assert set(models["files"]) == {"config.json", "model-00001-of-00004.safetensors"}
    assert models["prompt"] == olmo.CONVERSATION
    assert models["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["params"] == olmo.DEFAULT_PARAMS
    # A small image is enlarged to a longest side of 1288 pixels, as olmocr renders a page.
    assert records["blank"]["engine_extra"]["model_image"] == {"width": 1288, "height": 868}
    assert records["blank"]["engine_extra"]["front_matter"]["primary_language"] is None
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "output" / "page.md").read_text(encoding="utf-8") == ""
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""
    plain = records["plain"]["engine_extra"]
    assert (plain["front_matter"], plain["front_matter_error"]) == (None, "no front matter")
    plain_text = run.path / "samples" / "plain" / "attempt-1" / "prediction.txt"
    assert plain_text.read_text(encoding="utf-8") == "Plain text without front matter."

    news = records["news"]
    assert news["message"] == "generation stopped at max_new_tokens (8000 tokens)"
    assert news["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert news["engine_extra"]["generated_tokens"] == 8000
    assert news["engine_extra"]["model_image"] == {"width": 868, "height": 1288}
    assert news["engine_extra"]["front_matter"] == FRONT_MATTER

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


def test_greedy_decoding_image_size_and_context_limit(tmp_path):
    params = {"device": "cpu", "temperature": 0, "target_longest_image_dim": 640}
    samples = {"news": ("news", (2000, 3000))}
    _, records = run_adapter(tmp_path, samples, params, {"FAKE_TORCH_GPUS": "0"})
    extra = records["news"]["engine_extra"]
    # 2000 x 3000 pixels scaled to 427 x 640, then 15 x 23 units of 28 pixels.
    assert extra["model_image"] == {"width": 420, "height": 644}
    assert extra["image_tokens"] == 15 * 23
    assert extra["params"]["temperature"] == 0  # the fake model checks do_sample is off
    assert "gpu" not in extra["environment"]
    assert "gpu_peak_memory_mb" not in extra

    params = {"device": "cpu", "max_new_tokens": 127999}
    _, records = run_adapter(tmp_path / "context", samples, params, {"FAKE_TORCH_GPUS": "0"})
    extra = records["news"]["engine_extra"]
    assert extra["prompt_tokens"] == 31 * 46 + 5
    assert extra["max_new_tokens"] == 128000 - extra["prompt_tokens"]
    assert extra["stop_reason"] == "context"
    assert records["news"]["status"] == "truncated"
    assert "model's context" in records["news"]["message"]


@pytest.mark.parametrize(("content", "official"), [("other weights", False), (None, True)])
def test_a_custom_model_dir_claims_the_official_checkpoint_only_when_verified(
    tmp_path, content, official
):
    custom = tmp_path / "cache" / "models" / "custom"
    custom.mkdir(parents=True)
    for name in ("config.json", "model-00001-of-00004.safetensors"):  # None: the official files
        (custom / name).write_text(content or f"fake {name}", encoding="utf-8")
    params = CONFIG["params"] | {"model_dir": "models/custom"}
    _, records = run_adapter(tmp_path, {"page": ("page", (900, 420))}, params)
    assert records["page"]["status"] == "success"
    models = records["page"]["engine_extra"]["models"]
    assert models["verified_official_checkpoint"] is official
    assert ("repository" in models, "revision" in models) == (official, official)
    assert models["model_dir"] == "models/custom"
    digest = hashlib.sha256((content or "fake model-00001-of-00004.safetensors").encode())
    assert models["files"]["model-00001-of-00004.safetensors"] == digest.hexdigest()


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({"model_dir": "models/other"}, {}, "does not exist; download the weights"),
        ({}, "corrupt", "not matching models.sha256: ['model-00001-of-00004.safetensors']"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "corrupt":
        env = model_cache(tmp_path / "corrupt", corrupt=True)
    samples = {"a": ("page", (900, 420)), "b": ("blank", (300, 200))}
    _, records = run_adapter(tmp_path, samples, params, env)
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
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_OLMOCR2_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{olmo.REPOSITORY}/resolve/{olmo.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / olmo.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / olmo.PINNED_DIR).iterdir()) == [
        "model-00001-of-00004.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def olmocr2_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / olmo.PINNED_DIR / "model-00004-of-00004.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make olmocr2-model first")
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
def test_gpu_smoke_benchmark(tmp_path, olmocr2_image):
    cache = str(olmocr2_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "olmocr2"]) == 0
    run = load_run(tmp_path / "runs" / "olmocr2")
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
    name = "model-00001-of-00004.safetensors"
    assert extra["models"]["files"][name] == CHECKSUMS[CHECKSUMS.index(name) - 1]
    assert extra["models"]["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["stop_reason"] == "end-token"
    assert extra["front_matter"]["is_rotation_valid"] is True
    assert max(extra["model_image"].values()) == 1288
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

    # A small max_new_tokens truncates the page; the partial output is kept and scored.
    limited = tmp_path / "engine.toml"
    config = Path(engine).read_text(encoding="utf-8")
    limited.write_text(config.replace("max_new_tokens = 8000", "max_new_tokens = 64"), "utf-8")
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["message"] == "generation stopped at max_new_tokens (64 tokens)"
    assert page["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert page["engine_extra"]["generated_tokens"] == 64
    attempt = "samples/page-1/attempt-1/output/output.md"
    partial = (short_run.path / attempt).read_text(encoding="utf-8")
    assert partial
    assert main(["evaluate", str(short_run.path)]) == 0
    scores = json.loads((short_run.path / "evaluations" / "1" / "scores.json").read_text())
    short_rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < short_rates["page-1"] <= 1  # scored as produced

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make olmocr2-model" in r["message"] for r in records)
