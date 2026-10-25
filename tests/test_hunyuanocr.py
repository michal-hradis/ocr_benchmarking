"""HunyuanOCR-1.5 adapter (engines/hunyuanocr): parameters, prompts, the repetition stop and
clean-up, and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and
``transformers`` modules (tests/fixtures/hunyuanocr/fake) that "generate" fixed outputs, including
a hand-written Markdown page with an HTML table (tests/fixtures/hunyuanocr/output.md), a page that
ends repeating one line and a newspaper page whose output never ends. They also evaluate and
report that run. The opt-in GPU test builds the engine image and runs the real model from the
weights cache through ``ocrbench run``, ``evaluate`` and ``report``, for both presets.
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
ENGINE_DIR = ROOT / "engines" / "hunyuanocr"
FIXTURES = Path(__file__).parent / "fixtures" / "hunyuanocr"
ADAPTER_PATH = ENGINE_DIR / "hunyuanocr_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
TEXT_PRESET = tomllib.loads((ENGINE_DIR / "presets" / "text.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
# SHA-256 of TASK_PROMPTS["doc_parse"] and ["structured_parse"] in inference/utils/tasks.py of
# github.com/Tencent-Hunyuan/HunyuanOCR at commit 1ef4179e53cf860f7e6fd8276a292e5d05b3e927.
UPSTREAM_PROMPT_SHA256 = {
    "doc_parse": "16f15a15a1eb51817cd30c2dbf607e77a9f67eec41448526aab5aeb87f2311d4",
    "structured_parse": "ed471d18b8266adbdb756173820e77bd935241d15d24a0f90be2a0a7868c42e9",
}

_spec = importlib.util.spec_from_file_location("hunyuanocr_engine", ADAPTER_PATH)
hy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hy)

FIXTURE_LINES = [
    "The Daily Chronicle",
    "The quick brown fox jumps over the lazy dog near the river.",
    "A second paragraph with more text and a formula E = mc^2.",  # without the $ delimiters
    "Year\tCopies",
    "1923\t2026",
]
REPEATED = "The same line again and again."


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == TEXT_PRESET["model"] == f"{hy.REPOSITORY}@{hy.REVISION}"
    assert CONFIG["image"] == TEXT_PRESET["image"]
    assert hy.PINNED_DIR.endswith(hy.REVISION)
    names = CHECKSUMS[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == ["model.safetensors"]
    assert {"preprocessor_config.json", "chat_template.jinja", "tokenizer.json"} <= set(names)
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_presets_differ_only_in_the_task_prompt():
    assert (CONFIG["params"]["task"], TEXT_PRESET["params"]["task"]) == (
        "doc_parse",
        "structured_parse",
    )
    assert CONFIG["name"] != TEXT_PRESET["name"]
    for key in ("engine_version", "resources", "timeouts"):
        assert CONFIG[key] == TEXT_PRESET[key]
    other = {k: v for k, v in CONFIG["params"].items() if k != "task"}
    assert other == {k: v for k, v in TEXT_PRESET["params"].items() if k != "task"}


@pytest.mark.parametrize("task", ["doc_parse", "structured_parse"])
def test_prompts_are_the_official_ones(task):
    prompt = hy.TASK_PROMPTS[task]
    assert hashlib.sha256(prompt.encode()).hexdigest() == UPSTREAM_PROMPT_SHA256[task]
    assert hy.conversation(task) == [  # the official clients' request: the image first
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "gpu:0"}, "device"),
        ({"task": "layout"}, "task must be one of doc_parse, structured_parse"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 100.0}, "max_new_tokens"),
        ({"max_new_tokens": 131072}, "below the context"),
        ({"repetition_penalty": 0}, "repetition_penalty"),
        ({"repetition_penalty": float("inf")}, "repetition_penalty"),
        ({"repetition_penalty": "1.08"}, "repetition_penalty"),
        ({"repetition_stop": 1}, "repetition_stop"),
        ({"model_dir": "/cache/models/x"}, "model_dir"),
        ({"model_dir": "models/../../etc"}, "model_dir"),
        ({"model_dir": ""}, "model_dir"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(hy.AdapterError, match=message):
        hy.parse_params(params)


def test_params_defaults():
    assert hy.parse_params({"task": None}) == hy.DEFAULT_PARAMS
    assert hy.DEFAULT_PARAMS["model_dir"] == hy.PINNED_DIR
    # The official clients' decoding.
    assert hy.DEFAULT_PARAMS["task"] == "doc_parse"
    assert hy.DEFAULT_PARAMS["max_new_tokens"] == 32768
    assert hy.DEFAULT_PARAMS["repetition_penalty"] == 1.08
    assert hy.DEFAULT_PARAMS["repetition_stop"] is True
    params = hy.parse_params({"repetition_penalty": 1, "repetition_stop": False, "device": "cpu"})
    assert (params["repetition_penalty"], params["repetition_stop"]) == (1, False)


def test_images_are_read_as_stored(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(hy.AdapterError, match="cannot decode image x.png"):
        hy.read_image(tmp_path / "x.png")
    Image.new("LA", (4000, 3000)).save(tmp_path / "la.png")
    image, original = hy.read_image(tmp_path / "la.png")
    assert (image.mode, image.size, original) == (
        "RGB",
        (4000, 3000),
        {"width": 4000, "height": 3000},
    )


@pytest.mark.parametrize(
    ("text", "repeats"),
    [
        ("intro " + "ab" * 8, True),
        ("intro " + "ab" * 7, False),
        ("x" * 7 + "y" + "ab" * 8, True),
        ("intro " + "line\n" * 8, True),
        ("intro" + "\n" * 40, False),  # a blank unit is not a repetition
        ("intro " + ("u" * 255 + "v") * 8, True),  # a unit of 256 characters
        ("intro " + ("u" * 256 + "v") * 8, False),  # longer units are not checked
        ("ab" * 7 + "a", False),
    ],
)
def test_tail_repetition_is_detected_as_the_official_clients_do(text, repeats):
    assert hy.tail_repeats(text) is repeats


def test_a_repeated_tail_is_trimmed_to_one_copy():
    prefix = "".join(f"Line {n}.\n" for n in range(300))
    assert len(prefix) >= 2000
    assert hy.trim_repeated_tail(prefix + "ab" * 10) == prefix + "ab"
    assert hy.trim_repeated_tail(prefix + "ab" * 9) == prefix + "ab" * 9  # fewer than 10
    assert hy.trim_repeated_tail(prefix + "ab" * 40) == prefix + "ab"
    line = REPEATED + "\n"
    assert hy.trim_repeated_tail(prefix + line * 12) == prefix + line
    short = "Line 1.\n" + "ab" * 50
    assert hy.trim_repeated_tail(short) == short  # under 2,000 characters: unchanged
    assert hy.trim_repeated_tail(prefix) == prefix


def model_cache(root, corrupt=False):
    """A cache with fake model files at the pinned path, and checksums that match them."""
    directory = root / "cache" / hy.PINNED_DIR
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
        "OCRBENCH_HUNYUANOCR_CACHE": str(root / "cache"),
        "OCRBENCH_HUNYUANOCR_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000
# The fake model chooses its output by the gray level of the image (fake/README.md).
GRAY = {"page": 255, "blank": 250, "boom": 200, "loop": 150, "news": 100}


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
    config["timeouts"] = {"sample": 60, "startup": 20}
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
        "loop": ("loop", (1000, 1400)),
        "news": ("news", (2000, 3000)),
        "big": ("big", (5000, 7000)),
        "page2": ("page", (900, 420)),
    }
    run, records = run_adapter(tmp_path, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # nothing but an end token: a genuine empty result
        "boom": ("failure", "engine-error"),
        "loop": ("truncated", None),  # stopped when the output repeated itself
        "news": ("truncated", None),  # stopped at max_new_tokens, scored as produced
        "big": ("success", None),
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
    assert (page / "output" / "page.md").read_text(encoding="utf-8") == native.strip()
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
    assert extra["trimmed_chars"] == 0
    # 900 x 420 in whole 32-pixel units: 28 x 13, one token per unit and row end, and two more.
    assert extra["source_image"] == {"width": 900, "height": 420}
    assert extra["model_image"] == {"width": 896, "height": 416}
    assert extra["image_tokens"] == 13 * (28 + 1) + 2
    prompt = hy.TASK_PROMPTS["doc_parse"]
    assert extra["prompt_tokens"] == extra["image_tokens"] + len(prompt) + 3
    assert extra["max_new_tokens"] == 32768
    assert extra["gpu_peak_memory_mb"] == 3072
    assert extra["environment"]["transformers"] == "5.19.0+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["verified_official_checkpoint"] is True
    assert (models["repository"], models["revision"]) == (hy.REPOSITORY, hy.REVISION)
    assert models["model_dir"] == hy.PINNED_DIR
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert models["prompt"] == hy.conversation("doc_parse")
    assert models["prompt_sha256"] == UPSTREAM_PROMPT_SHA256["doc_parse"]
    assert extra["params"] == hy.DEFAULT_PARAMS
    # A small image is enlarged to at least 262,144 pixels, a large one reduced to 16,777,216.
    assert records["blank"]["engine_extra"]["model_image"] == {"width": 640, "height": 448}
    assert records["big"]["engine_extra"]["model_image"] == {"width": 3456, "height": 4832}
    assert records["big"]["engine_extra"]["source_image"] == {"width": 5000, "height": 7000}
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "output" / "page.md").read_text(encoding="utf-8") == ""
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""

    loop = records["loop"]
    assert loop["engine_extra"]["stop_reason"] == "repetition"
    assert loop["message"].startswith("generation stopped because the output repeats itself (")
    loop_dir = run.path / "samples" / "loop" / "attempt-1" / "output"
    output = (loop_dir / "output.md").read_text(encoding="utf-8")
    assert len(output) >= hy.REPEAT_START  # the first check is after 4,000 characters
    assert output.count(REPEATED) >= hy.REPEAT_TIMES
    assert output.count(REPEATED) < hy.REPEAT_TIMES + 2 * hy.REPEAT_STEP // len(REPEATED)
    page_md = (loop_dir / "page.md").read_text(encoding="utf-8")
    assert page_md == hy.trim_repeated_tail(output).strip()
    assert loop["engine_extra"]["trimmed_chars"] == len(output) - len(hy.trim_repeated_tail(output))
    assert page_md.startswith(_newspaper_line(1))
    assert _newspaper_line(100) in page_md

    news = records["news"]
    assert news["message"] == "generation stopped at max_new_tokens (32768 of 32768 tokens)"
    assert news["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert news["engine_extra"]["generated_tokens"] == 32768
    assert news["engine_extra"]["model_image"] == {"width": 1984, "height": 3008}

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


def test_text_preset_and_repetition_stop_off(tmp_path):
    params = TEXT_PRESET["params"] | {"repetition_stop": False, "max_new_tokens": 9000}
    run, records = run_adapter(tmp_path, {"loop": ("loop", (1000, 1400))}, params)
    record = records["loop"]
    extra = record["engine_extra"]
    assert extra["models"]["prompt"] == hy.conversation("structured_parse")
    assert extra["models"]["prompt_sha256"] == UPSTREAM_PROMPT_SHA256["structured_parse"]
    assert extra["prompt_tokens"] == extra["image_tokens"] + len("提取图中的文字。") + 3
    # Without the early stop, the repetition runs to max_new_tokens; page.md keeps one copy.
    assert extra["stop_reason"] == "max_new_tokens"
    assert extra["generated_tokens"] == 9000
    assert record["message"] == "generation stopped at max_new_tokens (9000 of 9000 tokens)"
    out = run.path / "samples" / "loop" / "attempt-1" / "output"
    output = (out / "output.md").read_text(encoding="utf-8")
    page_md = (out / "page.md").read_text(encoding="utf-8")
    assert len(output) == 9000
    assert output.count(REPEATED) > 100
    assert page_md.count(REPEATED) <= 2  # one copy, which may start inside the line
    assert extra["trimmed_chars"] == len(output) - len(hy.trim_repeated_tail(output))


def test_cpu_and_context_limit(tmp_path):
    samples = {"news": ("news", (2000, 3000))}
    params = {"device": "cpu", "max_new_tokens": 131071}
    _, records = run_adapter(tmp_path, samples, params, {"FAKE_TORCH_GPUS": "0"})
    extra = records["news"]["engine_extra"]
    assert "gpu" not in extra["environment"]
    assert "gpu_peak_memory_mb" not in extra
    assert extra["max_new_tokens"] == 131072 - extra["prompt_tokens"]
    assert extra["stop_reason"] == "context"
    assert records["news"]["status"] == "truncated"
    assert "the model's context" in records["news"]["message"]


@pytest.mark.parametrize(("content", "official"), [("other weights", False), (None, True)])
def test_a_custom_model_dir_claims_the_official_checkpoint_only_when_verified(
    tmp_path, content, official
):
    custom = tmp_path / "cache" / "models" / "custom"
    custom.mkdir(parents=True)
    for name in ("config.json", "model.safetensors"):  # None: the official files
        (custom / name).write_text(content or f"fake {name}", encoding="utf-8")
    params = CONFIG["params"] | {"model_dir": "models/custom"}
    _, records = run_adapter(tmp_path, {"page": ("page", (900, 420))}, params)
    assert records["page"]["status"] == "success"
    models = records["page"]["engine_extra"]["models"]
    assert models["verified_official_checkpoint"] is official
    assert ("repository" in models, "revision" in models) == (official, official)
    assert models["model_dir"] == "models/custom"
    digest = hashlib.sha256((content or "fake model.safetensors").encode())
    assert models["files"]["model.safetensors"] == digest.hexdigest()


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
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_HUNYUANOCR_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{hy.REPOSITORY}/resolve/{hy.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / hy.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / hy.PINNED_DIR).iterdir()) == [
        "model.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def hunyuanocr_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / hy.PINNED_DIR / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make hunyuanocr-model first")
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
@pytest.mark.parametrize("preset", ["engine.toml", "presets/text.toml"])
def test_gpu_smoke_benchmark(tmp_path, hunyuanocr_image, preset):
    cache = str(hunyuanocr_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / preset)
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "hunyuanocr"]) == 0
    run = load_run(tmp_path / "runs" / "hunyuanocr")
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
    name = "model.safetensors"
    assert extra["models"]["files"][name] == CHECKSUMS[CHECKSUMS.index(name) - 1]
    assert extra["models"]["verified_official_checkpoint"] is True
    task = tomllib.loads(Path(engine).read_text(encoding="utf-8"))["params"]["task"]
    assert extra["models"]["prompt_sha256"] == UPSTREAM_PROMPT_SHA256[task]
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

    # A small max_new_tokens truncates the page; the partial output is kept and scored.
    limited = tmp_path / "engine.toml"
    config = Path(engine).read_text(encoding="utf-8")
    limited.write_text(config.replace("max_new_tokens = 32768", "max_new_tokens = 16"), "utf-8")
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["message"] == "generation stopped at max_new_tokens (16 of 16 tokens)"
    assert page["engine_extra"]["stop_reason"] == "max_new_tokens"
    attempt = "samples/page-1/attempt-1/output/output.md"
    assert (short_run.path / attempt).read_text(encoding="utf-8")
    assert main(["evaluate", str(short_run.path)]) == 0
    scores = json.loads((short_run.path / "evaluations" / "1" / "scores.json").read_text())
    short_rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < short_rates["page-1"] <= 1  # scored as produced

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make hunyuanocr-model" in r["message"] for r in records)
