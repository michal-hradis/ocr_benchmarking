"""Nanonets-OCR2-3B adapter (engines/nanonets-ocr2): parameters, the pinned files, the vLLM server
command and request, and the engine protocol with host conversion of the model's Markdown.

Offline tests import the adapter and run it as a host process with a fake ``vllm`` executable that
serves fixed outputs over the loopback interface and a fake ``torch`` module
(tests/fixtures/nanonets-ocr2/fake). The outputs are a hand-written page in the model's output style
(tests/fixtures/nanonets-ocr2/page.md: a page number tag, an image description, an HTML table, a
checkbox, a watermark and a formula), an empty output, a newspaper page cut off at max_tokens or at
the end of the context in a repetition loop, an aborted request, a failed request and a server that
exits. They also evaluate and report those runs. The opt-in GPU test builds the engine image and
runs the preset with the real model from the weights cache through ``ocrbench run``, ``evaluate``
and ``report``.
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
from PIL import Image, ImageDraw

from ocrbench.cli import main
from ocrbench.convert import CONVERTER_VERSION
from ocrbench.engine import parse_engine_config
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import page_xml, smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "nanonets-ocr2"
FIXTURES = Path(__file__).parent / "fixtures" / "nanonets-ocr2"
FAKE = FIXTURES / "fake"
ADAPTER_PATH = ENGINE_DIR / "nanonets_ocr2_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("nanonets_ocr2_engine", ADAPTER_PATH)
nne = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nne)

# tests/fixtures/nanonets-ocr2/page.md after host conversion: the page number tag and the watermark
# keep their text, the image description adds nothing, the HTML table is tab-separated.
PAGE_TEXTS = [
    "12",
    "The Daily Chronicle",
    "The quick brown fox jumps over\nthe lazy dog & the river.",
    "Year\tCopies\n1923\t2O26",
    "☐ Subscribe SPECIMEN",
    "E = mc^2",
]
OFFICIAL_PROMPT = (
    "Extract the text from the above document as if you were reading it naturally. Return the "
    "tables in html format. Return the equations in LaTeX representation. If there is an image in "
    "the document and image caption is not present, add a small description of the image inside "
    "the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should "
    "be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be "
    "wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
    "Prefer using ☐ and ☑ for check boxes."
)


def test_preset_is_the_pinned_model_card_configuration():
    assert CONFIG["name"] == "nanonets-ocr2-3b"
    assert CONFIG["model"] == f"{nne.REPOSITORY}@{nne.REVISION}"
    assert nne.parse_params(CONFIG["params"]) == nne.DEFAULT_PARAMS
    assert parse_engine_config(CONFIG).image == "ocrbench/engine-nanonets-ocr2:3B-v1"


def test_prompt_is_the_official_prompt():
    assert nne.PROMPT == OFFICIAL_PROMPT  # the model card's and the cookbook's, byte for byte


def test_pinned_model_files():
    pinned = nne.PINNED_DIR
    assert pinned == f"models/nanonets-ocr2/Nanonets-OCR2-3B/{nne.REVISION}"
    lines = (ENGINE_DIR / "models.sha256").read_text().split()
    names = lines[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    for name in ("config.json", "chat_template.jinja", "tokenizer.json", "preprocessor_config.json",
                 "generation_config.json", "model.safetensors.index.json"):  # fmt: skip
        assert name in names
    assert not [n for n in names if n.endswith(".py")]  # vLLM's own Qwen2.5-VL, no remote code
    assert all(len(digest) == 64 for digest in lines[::2])
    # The weights' SHA-256 is the Hugging Face LFS object ID at the pinned revision.
    digests = dict(zip(names, lines[::2], strict=True))
    assert digests["model-00001-of-00002.safetensors"].startswith("84c3306e843b")
    assert digests["model-00002-of-00002.safetensors"].startswith("8002ad9c8c62")


def test_server_command():
    params = nne.parse_params({})
    command = nne.server_command(params, 1234)
    directory = f"/cache/models/nanonets-ocr2/Nanonets-OCR2-3B/{nne.REVISION}"
    assert command[:7] == ["vllm", "serve", directory, "--host", "127.0.0.1", "--port", "1234"]
    options = " ".join(command[7:])
    for expected in (
        "--served-model-name nanonets-ocr2 --dtype bfloat16",
        "--distributed-executor-backend uni --tensor-parallel-size 1",
        '--limit-mm-per-prompt {"image": 1}',
        "--max-model-len 32768",
        "--max-num-seqs 8 --gpu-memory-utilization 0.6",
    ):
        assert expected in options
    assert "--trust-remote-code" not in command
    assert nne.server_env(params) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    assert nne.server_env(nne.parse_params({"device": "cuda:1"}))["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "cpu"}, "device"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 100.0}, "max_tokens"),
        ({"max_tokens": 32768}, "below the 32768-token context"),
        ({"temperature": -1}, "temperature"),
        ({"repetition_penalty": 0}, "repetition_penalty"),
        ({"seed": 1.5}, "seed"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 0.9}, r"gpu_memory_utilization must be in \(0, 0.6\]"),
        ({"max_num_seqs": 0}, "max_num_seqs"),
        ({"startup_seconds": -1}, "startup_seconds"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(nne.AdapterError, match=message):
        nne.parse_params(params)


def test_params_defaults():
    assert nne.DEFAULT_PARAMS == {
        "device": "cuda:0",
        "max_tokens": 15000,
        "temperature": 0.0,
        "repetition_penalty": 1.05,
        "seed": 0,
        "gpu_memory_utilization": 0.6,
        "max_num_seqs": 8,
        "startup_seconds": 900,
    }
    # Largest processed image (16,384 image tokens), prompt and max_tokens fit the context.
    assert 16384 + 256 + nne.DEFAULT_PARAMS["max_tokens"] < nne.CONTEXT


def test_request_is_the_model_cards_request():
    image = Image.new("RGB", (40, 30), "white")
    request = nne.payload(image, nne.parse_params({}))
    (message,) = request["messages"]
    picture, prompt = message["content"]  # the image first, then the prompt
    assert message["role"] == "user"
    assert picture["image_url"]["url"].startswith("data:image/png;base64,")
    assert prompt == {"type": "text", "text": OFFICIAL_PROMPT}
    assert (request["temperature"], request["max_tokens"]) == (0.0, 15000)
    assert (request["repetition_penalty"], request["seed"]) == (1.05, 0)


def test_tail_repetition():
    assert nne.tail_repeats("Intro " + "abc " * 8)
    assert not nne.tail_repeats("Intro " + "abc " * 7)
    assert not nne.tail_repeats(" " * 50)
    sentence = "A long looping sentence of a newspaper column, " * 2  # 94 characters
    assert nne.tail_repeats("Intro " + sentence * 3 + sentence[:40])
    assert not nne.tail_repeats("Intro " + sentence * 2 + sentence[:40])


def page_image(gray, size):
    """A non-uniform page: a checkerboard and the fake server's selector gray level in the top-left
    corner."""
    width, height = size
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 4):
        for y in range(0, height, 4):
            if (x + y) % 8 == 0:
                draw.rectangle((x, y, x + 3, y + 3), fill=255)
    draw.rectangle((0, 0, width // 10, height // 10), fill=gray)
    return image


def model_cache(root, corrupt=False, extra_file=False):
    """A cache with fake model files at the pinned path, and checksums that match."""
    directory = root / "cache" / nne.PINNED_DIR
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model-00001-of-00002.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "config.json").write_text("changed", encoding="utf-8")
    if extra_file:
        (directory / "modeling_qwen.py").write_text("", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", "utf-8")
    return {
        "OCRBENCH_NANONETS_OCR2_CACHE": str(root / "cache"),
        "OCRBENCH_NANONETS_OCR2_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000
# The fake server chooses its output by the gray level in the top-left corner (fake/README.md).
GRAY = {"page": 200, "news": 100, "abort": 80, "boom": 50, "die": 20, "empty": 70}


def make_dataset(root, samples):
    data = root / "data"
    data.mkdir()
    entries = []
    for sid, (kind, size) in samples.items():
        page_image(GRAY[kind], size).save(data / f"{sid}.png")
        if kind == "news":  # a newspaper page whose reference has 2,000 lines
            lines = [_newspaper_line(n) for n in range(1, NEWSPAPER_LINES + 1)]
            regions = [[(text, (10, n, 20, n + 1)) for n, text in enumerate(lines)]]
        elif kind == "page":
            lines = [line for text in PAGE_TEXTS for line in text.replace("\t", " ").split("\n")]
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
    config["timeouts"] = {"sample": 60, "startup": 30}
    del config["resources"]
    env = {
        "PYTHONPATH": str(FAKE),
        "PATH": f"{FAKE / 'bin'}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}"
        + os.environ["PATH"],
        "OCRBENCH_NANONETS_OCR2_LOG": str(tmp_path / "vllm-server.log"),
        **model_cache(tmp_path),
        **(env or {}),
    }
    run = create_run(
        parse_engine_config(config), make_dataset(tmp_path, samples), tmp_path / "runs",
        runtime="process",
    )  # fmt: skip
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in samples}


def test_engine_protocol_and_host_conversion(tmp_path):
    samples = {
        "page": ("page", (1000, 700)),
        "empty": ("empty", (300, 200)),
        "boom": ("boom", (666, 400)),
        "news": ("news", (2000, 3000)),
        "abort": ("abort", (300, 200)),
        "big": ("page", (5000, 7000)),
        "huge": ("news", (4194, 6192)),
        "die": ("die", (500, 500)),
        "after": ("page", (1000, 700)),
    }
    run, records = run_adapter(tmp_path, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "empty": ("success", None),  # an empty output is an empty page
        "boom": ("failure", "engine-error"),
        "news": ("truncated", None),  # stopped at max_tokens, scored as produced
        "abort": ("failure", "engine-error"),
        "big": ("success", None),  # the adapter keeps going after failed samples
        "huge": ("truncated", None),
        "die": ("failure", "engine-error"),
        "after": ("failure", "engine-error"),
    }
    assert records["empty"]["message"] == "the model returned an empty page"
    assert "request failed: HTTP 500" in records["boom"]["message"]
    assert "fake CUDA out of memory" in records["boom"]["message"]
    assert records["abort"]["message"] == "generation failed (finish_reason 'abort')"
    assert records["after"]["message"].startswith("the vLLM server has exited")

    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {"page.md", "result.json"}
    native = (page / "output" / "page.md").read_text(encoding="utf-8")
    assert native == (FIXTURES / "page.md").read_text(encoding="utf-8")  # unchanged
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": CONVERTER_VERSION}
    assert all("polygon" not in r for r in prediction["regions"])  # no positions are invented
    assert "reading_order" not in prediction
    text = (page / "prediction.txt").read_text(encoding="utf-8")
    assert text == "\n\n".join(PAGE_TEXTS)
    for markup in ("<", "**", "##", "woodcut", "&amp;", "$$"):
        assert markup not in text
    assert records["page"]["findings"] == []

    extra = records["page"]["engine_extra"]
    assert (extra["finish_reason"], extra["tail_repetition"]) == ("stop", False)
    assert extra["generated_tokens"] == len(native)
    assert extra["prompt_tokens"] == (1000 // 28) * (700 // 28) + 150
    assert extra["max_tokens"] == 15000
    assert extra["request_seconds"] >= 0
    assert extra["source_image"] == {"width": 1000, "height": 700}
    # The adapter sends the image at its stored size; the model's processor resizes it.
    assert records["big"]["engine_extra"]["prompt_tokens"] == 16384 + 150
    assert records["big"]["engine_extra"]["source_image"] == {"width": 5000, "height": 7000}
    environment = extra["environment"]
    assert (environment["vllm"], environment["transformers"]) == ("0.30.0+fake", "5.16.1+fake")
    assert (environment["gpu"], environment["gpu_capability"]) == ("Fake GPU", "8.9")
    assert environment["server_command"][:2] == ["vllm", "serve"]
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["repository"], models["revision"]) == (nne.REPOSITORY, nne.REVISION)
    assert set(models["files"]) == {"config.json", "model-00001-of-00002.safetensors"}
    assert models["prompt"] == OFFICIAL_PROMPT
    assert models["prompt_sha256"] == hashlib.sha256(OFFICIAL_PROMPT.encode()).hexdigest()
    assert models["context"] == 32768
    assert extra["params"] == nne.DEFAULT_PARAMS

    empty = run.path / "samples" / "empty" / "attempt-1"
    assert (empty / "prediction.txt").read_text(encoding="utf-8") == ""
    assert (empty / "output" / "page.md").read_text(encoding="utf-8") == ""  # kept
    abort = run.path / "samples" / "abort" / "attempt-1" / "output" / "page.md"
    assert abort.read_text(encoding="utf-8") == "Cut off"  # kept on failure
    abort_extra = records["abort"]["engine_extra"]  # the diagnostics are kept on failure
    assert (abort_extra["finish_reason"], abort_extra["generated_tokens"]) == ("abort", 7)

    news = records["news"]
    assert news["message"] == (
        "generation stopped at max_tokens (15000 tokens); the output ends in a repetition loop"
    )
    assert news["engine_extra"]["tail_repetition"] is True
    news_text = (run.path / "samples" / "news" / "attempt-1" / "prediction.txt").read_text()
    assert news_text.startswith(_newspaper_line(1))
    assert _newspaper_line(99) in news_text
    huge = records["huge"]  # the dev-01 newspaper size: 16,384 image tokens
    assert huge["message"] == (
        "generation stopped at max_tokens (15000 tokens); the output ends in a repetition loop"
    )

    # Common metrics, including the truncated 2,000-line newspaper page.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["page"]["chars"]["rate"] < 0.05
    news = by_id["news"]
    assert news["status"] == "truncated"
    assert news["flexible_units"]["reference"] == NEWSPAPER_LINES
    assert 0 < news["chars"]["rate"] < 1  # only the beginning of the page was transcribed
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


def test_context_truncation(tmp_path):
    params = dict(CONFIG["params"], max_tokens=32000)
    _, records = run_adapter(tmp_path, {"news": ("news", (4194, 6192))}, params)
    news = records["news"]
    assert news["status"] == "truncated"
    assert news["message"] == (
        f"generation stopped at the end of the 32768-token context ({32768 - 16384 - 150} "
        "tokens); the output ends in a repetition loop"
    )


def test_max_tokens_truncation_without_loop(tmp_path):
    params = dict(CONFIG["params"], max_tokens=500)
    _, records = run_adapter(tmp_path, {"news": ("news", (2000, 3000))}, params)
    news = records["news"]
    assert news["status"] == "truncated"
    assert news["message"] == "generation stopped at max_tokens (500 tokens)"
    assert news["engine_extra"]["tail_repetition"] is False
    assert news["engine_extra"]["max_tokens"] == 500


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "corrupt", "not matching models.sha256: ['config.json']"),
        ({}, "extra", "model files not in models.sha256: ['modeling_qwen.py']"),
        ({}, "missing", "does not exist; download the weights with make nanonets-ocr2-model"),
        ({}, {"FAKE_VLLM_EXIT": "1"}, "vLLM exited with status 3 during start-up"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env in ("corrupt", "extra"):
        env = model_cache(tmp_path / env, corrupt=env == "corrupt", extra_file=env == "extra")
    elif env == "missing":
        env = {"OCRBENCH_NANONETS_OCR2_CACHE": str(tmp_path / "empty")}
    samples = {"a": ("page", (900, 420)), "b": ("empty", (300, 200))}
    _, records = run_adapter(tmp_path, samples, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_other_model_is_rejected(tmp_path):
    config = dict(CONFIG, model="nanonets/Nanonets-OCR2-1.5B-exp@main")
    config["timeouts"] = {"sample": 60, "startup": 30}
    del config["resources"]
    env = {
        "PYTHONPATH": str(FAKE),
        "PATH": f"{FAKE / 'bin'}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}"
        + os.environ["PATH"],
        "OCRBENCH_NANONETS_OCR2_LOG": str(tmp_path / "vllm-server.log"),
        **model_cache(tmp_path),
    }
    run = create_run(
        parse_engine_config(config), make_dataset(tmp_path, {"a": ("page", (900, 420))}),
        tmp_path / "runs", runtime="process",
    )  # fmt: skip
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    record = latest_record(run, "a")
    assert record["status"] == "failure"
    assert "is not the pinned checkpoint" in record["message"]


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "nanonets_ocr2_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_NANONETS_OCR2_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main([str(tmp_path / "cache")])
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(
            f"https://huggingface.co/nanonets/Nanonets-OCR2-3B/resolve/{nne.REVISION}/"
        )
        Path(path).write_text("wrong", encoding="utf-8")

    directory = tmp_path / "cache" / nne.PINNED_DIR
    (directory / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs from models.sha256"):
        fetch_model.main([str(tmp_path / "cache")])
    assert sorted(p.name for p in directory.iterdir()) == ["model-00001-of-00002.safetensors"]
    with pytest.raises(SystemExit, match="Usage"):
        fetch_model.main([])


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def nanonets_ocr2_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / nne.PINNED_DIR / "model-00002-of-00002.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make nanonets-ocr2-model first")
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
def test_gpu_smoke_benchmark(tmp_path, nanonets_ocr2_image):
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", str(nanonets_ocr2_image), "--run-id", "nanonets"]) == 0
    run = load_run(tmp_path / "runs" / "nanonets")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert records["page-1"]["status"] == "success"
    assert records["para-1"]["status"] == "success"
    assert records["blank-1"]["status"] in ("success", "truncated")
    extra = records["page-1"]["engine_extra"]
    environment = extra["environment"]
    assert environment["vllm"] == "0.30.0"
    assert environment["torch"] == "2.13.0+cu130"
    assert environment["gpu"]
    lines = (ENGINE_DIR / "models.sha256").read_text().split()
    weights = lines[lines.index("model-00001-of-00002.safetensors") - 1]
    assert extra["models"]["files"]["model-00001-of-00002.safetensors"] == weights
    assert extra["finish_reason"] == "stop"
    for sid in ("page-1", "para-1"):
        prediction = json.loads(
            (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
        )
        assert prediction["conversion"] == {"from": "markdown", "version": CONVERTER_VERSION}
        assert "<page_number>" not in prediction["text"]  # markup converted on the host

    # Scoring and re-scoring read only stored outputs: no container runs.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["para-1"]["chars"]["rate"] < 0.1
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()
