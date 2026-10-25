"""Surya 2 adapter (engines/surya2): parameters, the vLLM server command, the block conversion to a
positioned prediction with region markup, and the engine protocol.

Offline tests import the adapter and run it as a host process with a fake ``vllm`` executable that
serves fixed outputs over the loopback interface, and fake ``surya``, ``openai`` and ``torch``
modules (tests/fixtures/surya2/fake). The outputs include hand-written layout blocks
(tests/fixtures/surya2/output.html), output without blocks, a newspaper page cut off at
``max_tokens`` in a repetition loop, a failed request and a server that exits. They also evaluate
and report that run, whose region markup is converted on the host. The opt-in GPU test builds the
engine image and runs the real model from the weights cache through ``ocrbench run``, ``evaluate``
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
from ocrbench.engine import parse_engine_config
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import page_xml, smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "surya2"
FIXTURES = Path(__file__).parent / "fixtures" / "surya2"
FAKE = FIXTURES / "fake"
ADAPTER_PATH = ENGINE_DIR / "surya2_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
# SHA-256 of HIGH_ACCURACY_BBOX_PROMPT in surya/inference/prompts.py of surya-ocr 0.22.1.
UPSTREAM_PROMPT_SHA256 = "289c0de98f71e9bc47617bc265cae2b4fbb4613c334524f2c7da474942d7d43a"

_spec = importlib.util.spec_from_file_location("surya2_engine", ADAPTER_PATH)
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)

# The region texts of tests/fixtures/surya2/output.html after host conversion, in output order;
# the Image block has none, the block without a box and the one over a white area are left out.
FIXTURE_TEXTS = [
    "Page 3",
    "The Daily Chronicle",
    "The quick brown fox jumps over\nthe lazy dog & the river.",
    "First item\nSecond item",
    "E = mc^2",
    "Year\tCopies\n1923\t2026",
    "3",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{s2.REPOSITORY}@{s2.REVISION}"
    assert s2.PINNED_DIR.endswith(s2.REVISION)
    names = CHECKSUMS[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == ["model.safetensors"]
    assert {"config.json", "preprocessor_config.json", "tokenizer.json"} <= set(names)
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])
    # The weights' SHA-256 is the Hugging Face LFS object ID at the pinned revision.
    assert CHECKSUMS[CHECKSUMS.index("model.safetensors") - 1].startswith("5755f82a997d")


def test_server_command_has_suryas_settings():
    params = s2.parse_params({})
    command = s2.server_command(params, 1234)
    assert command[:7] == [
        "vllm",
        "serve",
        f"/cache/models/surya2/surya-ocr-2/{s2.REVISION}",
        "--host",
        "127.0.0.1",
        "--port",
        "1234",
    ]
    options = " ".join(command[7:])
    for expected in (
        "--no-enforce-eager --max-num-seqs 32 --dtype bfloat16 --max-model-len 18000",
        "--max-num-batched-tokens 8192 --enable-prefix-caching",
        '--mm-processor-kwargs {"min_pixels": 3136, "max_pixels": 6291456}',
        "--served-model-name datalab-to/surya-ocr-2 --gpu-memory-utilization 0.6",
        '--speculative-config {"method": "mtp", "num_speculative_tokens": 2}',
    ):
        assert expected in options
    no_mtp = s2.server_command(s2.parse_params({"mtp": False}), 1234)
    assert "--speculative-config" not in no_mtp


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "cpu"}, "device"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 100.0}, "max_tokens"),
        ({"max_tokens": 18000}, "below the 18000-token context"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 0.85}, r"gpu_memory_utilization must be in \(0, 0.6\]"),
        ({"mtp": 1}, "mtp"),
        ({"startup_seconds": -1}, "startup_seconds"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(s2.AdapterError, match=message):
        s2.parse_params(params)


def test_params_defaults():
    assert s2.parse_params({"max_tokens": None}) == s2.DEFAULT_PARAMS
    assert s2.DEFAULT_PARAMS == {
        "device": "cuda:0",
        "max_tokens": 12288,
        "gpu_memory_utilization": 0.6,
        "mtp": True,
        "startup_seconds": 900,
    }
    assert s2.parse_params(CONFIG["params"]) == s2.DEFAULT_PARAMS


def page_image(gray, size, white_area=False):
    """A non-uniform page: a checkerboard, the fake server's selector gray level in the top-left
    corner and, optionally, a white area under the fixture's last Text block."""
    width, height = size
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 4):
        for y in range(0, height, 4):
            if (x + y) % 8 == 0:
                draw.rectangle((x, y, x + 3, y + 3), fill=255)
    draw.rectangle((0, 0, width // 10, height // 10), fill=gray)
    if white_area:
        draw.rectangle((width * 0.72, height * 0.82, width, height * 0.98), fill=255)
    return image


def test_blocks_become_positioned_regions_with_markup(monkeypatch):
    monkeypatch.syspath_prepend(str(FAKE))
    from surya.inference.parsers import parse_full_page_html

    image = page_image(200, (1000, 700), white_area=True)
    blocks = parse_full_page_html((FIXTURES / "output.html").read_text(encoding="utf-8"))
    prediction, counts = s2.blocks_to_prediction(blocks, "s", image)
    assert counts == {"blank_blocks": 1, "skipped_blocks": 1}
    regions = prediction["regions"]
    assert [r["native_type"] for r in regions] == [
        "Page-Header",
        "Section-Header",
        "Text",
        "List-Group",
        "Equation-Block",
        "Image",
        "Table",
        "Page-Footer",
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
    corners = ([200, 56], [800, 56], [800, 98], [200, 98])
    assert regions[1]["polygon"] == [pytest.approx(corner) for corner in corners]
    heading = "<h1>The <i>Daily</i> Chronicle</h1>"
    assert regions[1]["markup"] == {"format": "html", "content": heading}
    assert "markup" not in regions[5]  # the Image block keeps no HTML
    assert prediction["text"] == ""
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(1000, 700))
    assert issues == []
    assert [r.text for r in standardized.regions if r.text] == FIXTURE_TEXTS
    assert standardized.text == "\n\n".join(FIXTURE_TEXTS)
    assert standardized.conversion == {"from": "html", "version": 1}


def test_blocks_skipped_by_their_canonical_label(monkeypatch):
    """Surya skips Complex-Block, canonically Figure, although the raw label is not in
    SKIP_OCR_LABELS: the region keeps its box but not the block's text."""
    monkeypatch.syspath_prepend(str(FAKE))
    from surya.inference.parsers import parse_full_page_html

    blocks = parse_full_page_html(
        '<div data-bbox="100 100 500 400" data-label="Complex-Block"><p>Chart: sales 2024</p></div>'
        '<div data-bbox="100 500 900 600" data-label="Text"><p>Body text.</p></div>'
    )
    prediction, counts = s2.blocks_to_prediction(blocks, "s", page_image(200, (1000, 700)))
    assert counts == {"blank_blocks": 0, "skipped_blocks": 1}
    figure, text = prediction["regions"]
    assert (figure["native_type"], figure["type"]) == ("Complex-Block", "figure")
    corners = ([100, 70], [500, 70], [500, 280], [100, 280])
    assert figure["polygon"] == [pytest.approx(corner) for corner in corners]
    assert "markup" not in figure
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(1000, 700))
    assert issues == []
    assert standardized.text == "Body text."


def model_cache(root, corrupt=False, extra_file=False):
    """A cache with fake model files at the pinned path, and checksums that match them."""
    directory = root / "cache" / s2.PINNED_DIR
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "model.safetensors").write_text("changed", encoding="utf-8")
    if extra_file:
        (directory / "extra.json").write_text("{}", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_SURYA2_CACHE": str(root / "cache"),
        "OCRBENCH_SURYA2_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000
# The fake server chooses its output by the gray level in the top-left corner (fake/README.md).
# "blank" is a uniform page of the empty-output gray level, which is_blank_region accepts.
GRAY = {"page": 200, "nohtml": 150, "news": 100, "boom": 50, "die": 20, "empty": 70, "blank": 70}


def make_dataset(root, samples):
    data = root / "data"
    data.mkdir()
    entries = []
    for sid, (kind, size) in samples.items():
        if kind == "blank":
            image = Image.new("L", size, GRAY[kind])
        else:
            image = page_image(GRAY[kind], size, white_area=kind == "page")
        image.save(data / f"{sid}.png")
        if kind == "news":  # a newspaper page whose reference has 2,000 lines
            lines = [_newspaper_line(n) for n in range(1, NEWSPAPER_LINES + 1)]
            regions = [[(text, (10, n, 20, n + 1)) for n, text in enumerate(lines)]]
        elif kind == "page":
            lines = [line for text in FIXTURE_TEXTS for line in text.replace("\t", " ").split("\n")]
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
        "OCRBENCH_SURYA2_LOG": str(tmp_path / "vllm-server.log"),
        **model_cache(tmp_path),
        **(env or {}),
    }
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
        "page": ("page", (1000, 700)),
        "blank": ("blank", (300, 200)),
        "empty": ("empty", (300, 200)),
        "nohtml": ("nohtml", (500, 500)),
        "boom": ("boom", (666, 400)),
        "news": ("news", (2000, 3000)),
        "big": ("page", (5000, 7000)),
        "page2": ("page", (1000, 700)),
        "die": ("die", (500, 500)),
        "after": ("page", (1000, 700)),
    }
    run, records = run_adapter(tmp_path, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # an empty output is accepted only for a blank page
        "empty": ("failure", "engine-error"),
        "nohtml": ("failure", "engine-error"),
        "boom": ("failure", "engine-error"),
        "news": ("truncated", None),  # stopped at max_tokens, scored as produced
        "big": ("success", None),
        "page2": ("success", None),  # the adapter keeps going after failed samples
        "die": ("failure", "engine-error"),
        "after": ("failure", "engine-error"),
    }
    assert (
        records["empty"]["message"] == "the output is empty but the page is not blank (output.html)"
    )
    assert records["nohtml"]["message"] == "the output has no layout blocks (output.html)"
    assert "fake CUDA out of memory" in records["boom"]["message"]
    assert records["after"]["message"].startswith("the vLLM server has exited")

    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "output.html",
        "prediction.json",
        "result.json",
    }
    native = (page / "output" / "output.html").read_text(encoding="utf-8")
    assert native == (FIXTURES / "output.html").read_text(encoding="utf-8").strip()  # unchanged
    assert records["page"]["findings"] == []
    engine_prediction = json.loads((page / "output" / "prediction.json").read_text())
    assert engine_prediction["text"] == ""
    assert "markup" in engine_prediction["regions"][0]
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "html", "version": 1}  # converted on the host
    regions = prediction["regions"]
    assert [r["text"] for r in regions if "text" in r] == FIXTURE_TEXTS
    assert all("markup" not in r for r in regions)
    corners = ([400, 14], [600, 14], [600, 42], [400, 42])
    assert regions[0]["polygon"] == [pytest.approx(corner) for corner in corners]
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n\n".join(FIXTURE_TEXTS)

    extra = records["page"]["engine_extra"]
    assert extra["stop_reason"] == "end-token"
    assert extra["generated_tokens"] == len(native)  # one fake token per character
    assert (extra["blocks"], extra["blank_blocks"], extra["skipped_blocks"]) == (9, 1, 1)
    assert extra["repeat_loop"] is False
    assert extra["source_image"] == {"width": 1000, "height": 700}
    assert extra["model_image"] == {"width": 1008, "height": 700}  # whole 28-pixel units
    assert extra["max_tokens"] == 12288
    environment = extra["environment"]
    assert (environment["vllm"], environment["surya-ocr"]) == ("0.20.1+fake", "0.22.1+fake")
    assert environment["gpu"] == "Fake GPU"
    assert environment["server_command"][:2] == ["vllm", "serve"]
    assert "0.6" in environment["server_command"]
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["repository"], models["revision"]) == (s2.REPOSITORY, s2.REVISION)
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert models["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert (models["temperature"], models["top_p"]) == (0.0, 0.1)
    assert extra["params"] == s2.DEFAULT_PARAMS

    big = records["big"]["engine_extra"]["model_image"]
    assert big["width"] * big["height"] <= 3072 * 2048
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""
    empty = run.path / "samples" / "empty" / "attempt-1"
    assert (empty / "output" / "output.html").read_text(encoding="utf-8") == ""  # kept

    news = records["news"]
    assert news["message"].startswith("generation stopped: length (12288 of 12288 tokens); ")
    assert news["engine_extra"]["repeat_loop"] is True  # recorded, not regenerated
    news_text = (run.path / "samples" / "news" / "attempt-1" / "prediction.txt").read_text()
    assert news_text.startswith(_newspaper_line(1))
    assert _newspaper_line(100) in news_text

    # Common metrics on positioned output, including the truncated 2,000-line newspaper page.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["page"]["chars"]["rate"] < 0.05  # the table cells are tab-separated
    news = by_id["news"]
    assert news["status"] == "truncated"
    assert news["flexible_units"]["reference"] == NEWSPAPER_LINES
    assert 0 < news["chars"]["rate"] < 1  # only the beginning of the page was transcribed
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "corrupt", "not matching models.sha256: ['model.safetensors']"),
        ({}, "extra", "model files not in models.sha256: ['extra.json']"),
        ({}, "missing", "does not exist; download the weights with make surya2-model"),
        ({}, {"FAKE_VLLM_EXIT": "1"}, "vLLM exited with status 3 during start-up"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env in ("corrupt", "extra"):
        env = model_cache(tmp_path / env, corrupt=env == "corrupt", extra_file=env == "extra")
    elif env == "missing":
        env = {"OCRBENCH_SURYA2_CACHE": str(tmp_path / "empty")}
    samples = {"a": ("page", (900, 420)), "b": ("empty", (300, 200))}
    _, records = run_adapter(tmp_path, samples, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "surya2_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_SURYA2_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{s2.REPOSITORY}/resolve/{s2.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / s2.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / s2.PINNED_DIR).iterdir()) == [
        "model.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def surya2_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / s2.PINNED_DIR / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make surya2-model first")
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
def test_gpu_smoke_benchmark(tmp_path, surya2_image):
    cache = str(surya2_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "surya2"]) == 0
    run = load_run(tmp_path / "runs" / "surya2")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    extra = records["page-1"]["engine_extra"]
    environment = extra["environment"]
    assert (environment["vllm"], environment["surya-ocr"]) == ("0.20.1", "0.22.1")
    assert environment["torch"] == "2.11.0+cu130"
    assert environment["gpu"]
    assert (
        extra["models"]["files"]["model.safetensors"]
        == CHECKSUMS[CHECKSUMS.index("model.safetensors") - 1]
    )
    assert extra["models"]["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["stop_reason"] == "end-token"
    for sid in ("page-1", "para-1"):
        prediction = json.loads(
            (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
        )
        assert prediction["conversion"] == {"from": "html", "version": 1}
        assert prediction["reading_order"]
        assert all("polygon" in region for region in prediction["regions"])
        assert "<" not in prediction["text"]  # markup converted on the host

    # Scoring and re-scoring read only stored outputs: no container runs.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["para-1"]["chars"]["rate"] < 0.1
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()
