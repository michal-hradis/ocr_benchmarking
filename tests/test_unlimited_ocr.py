"""Unlimited-OCR adapter (engines/unlimited-ocr): parameters, the vLLM server command and request,
the layout block parsing and conversion to a positioned prediction with region markup, and the
engine protocol.

Offline tests import the adapter and run it as a host process with a fake ``vllm`` executable that
serves fixed outputs over the loopback interface and a fake ``torch`` module
(tests/fixtures/unlimited-ocr/fake). The outputs include hand-written layout blocks
(tests/fixtures/unlimited-ocr/output.txt), output without tags, an empty output, a newspaper page
cut off at the end of the context in a repetition loop, an aborted request, a failed request and a
server that exits. They also evaluate and report that run, whose region markup is converted on the
host. The opt-in GPU test builds the engine image and runs the real model from the weights cache
through ``ocrbench run``, ``evaluate`` and ``report``.
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
ENGINE_DIR = ROOT / "engines" / "unlimited-ocr"
FIXTURES = Path(__file__).parent / "fixtures" / "unlimited-ocr"
FAKE = FIXTURES / "fake"
ADAPTER_PATH = ENGINE_DIR / "unlimited_ocr_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
CONTEXT = 32768

_spec = importlib.util.spec_from_file_location("unlimited_ocr_engine", ADAPTER_PATH)
uoe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uoe)

# The region texts of tests/fixtures/unlimited-ocr/output.txt after host conversion, in output
# order; the image block has none. "\[...\]" math is read as Markdown escapes, a documented
# limitation of converter version 1 (docs/PREDICTIONS.md).
FIXTURE_TEXTS = [
    "-12-",
    "The Daily Chronicle",
    "The quick brown fox jumps over\nthe lazy dog & the river.",
    "Figure 1: A fox.",
    "Year\tCopies\n1923\t2O26",
    "[E = mc^2]",
    "Footer",
    "A block without a valid box.",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{uoe.REPOSITORY}@{uoe.REVISION}"
    assert uoe.PINNED_DIR.endswith(uoe.REVISION)
    names = CHECKSUMS[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == ["model-00001-of-000001.safetensors"]
    assert {"config.json", "model.safetensors.index.json", "tokenizer.json", "LICENSE"} <= set(
        names
    )
    assert not [n for n in names if n.endswith(".py")]  # vLLM's own model code, no remote code
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])
    # The weights' SHA-256 is the Hugging Face LFS object ID at the pinned revision.
    weights = CHECKSUMS[CHECKSUMS.index("model-00001-of-000001.safetensors") - 1]
    assert weights.startswith("2bc48a7a1100")


def test_server_command_has_the_recipe_settings():
    params = uoe.parse_params({})
    command = uoe.server_command(params, 1234)
    directory = f"/cache/models/unlimited-ocr/Unlimited-OCR/{uoe.REVISION}"
    assert command[:7] == ["vllm", "serve", directory, "--host", "127.0.0.1", "--port", "1234"]
    options = " ".join(command[7:])
    for expected in (
        "--served-model-name unlimited-ocr --dtype bfloat16",
        "--distributed-executor-backend uni --tensor-parallel-size 1 --max-model-len 32768",
        "--no-enable-prefix-caching --mm-processor-cache-gb 0",
        '--limit-mm-per-prompt {"image": 1}',
        "--logits-processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--max-num-seqs 8 --gpu-memory-utilization 0.6",
    ):
        assert expected in options
    assert "--trust-remote-code" not in command
    assert "--attention-config" not in command  # vLLM picks an R-SWA backend for the GPU
    assert uoe.server_env(params) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    assert uoe.server_env(uoe.parse_params({"device": "cuda:1"}))["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "cpu"}, "device"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 100.0}, "max_tokens"),
        ({"max_tokens": CONTEXT}, "below the 32768-token context"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 0.9}, r"gpu_memory_utilization must be in \(0, 0.6\]"),
        ({"max_num_seqs": 0}, "max_num_seqs"),
        ({"startup_seconds": -1}, "startup_seconds"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(uoe.AdapterError, match=message):
        uoe.parse_params(params)


def test_params_defaults():
    assert uoe.parse_params({"max_tokens": None}) == uoe.DEFAULT_PARAMS
    assert uoe.DEFAULT_PARAMS == {
        "device": "cuda:0",
        "max_tokens": None,
        "gpu_memory_utilization": 0.6,
        "max_num_seqs": 8,
        "startup_seconds": 900,
    }
    assert uoe.parse_params(CONFIG["params"]) == uoe.DEFAULT_PARAMS


def test_request_is_the_recipe_request():
    image = Image.new("RGB", (40, 30), "white")
    request = uoe.payload(image, None)
    assert "max_tokens" not in request  # vLLM generates up to the end of the context
    text, picture = request["messages"][0]["content"]
    assert text == {"type": "text", "text": "<image>document parsing."}
    assert picture["image_url"]["url"].startswith("data:image/png;base64,")
    assert (request["temperature"], request["skip_special_tokens"]) == (0.0, False)
    assert request["vllm_xargs"] == {"ngram_size": 35, "window_size": 128}
    assert uoe.payload(image, 500)["max_tokens"] == 500


def test_layout_blocks_keep_boxes_as_written():
    text = (
        "Loose text\n<|det|>title [10, 20, 30, 40]<|/det|>## Head\n"
        "<|det|>text [999, 999, 500, 0]<|/det|>Body\nmore\n"
        "<|det|>text [1, 2, x, 4]<|/det|>Bad box\n<|det|>[1, 2, 3, 4]<|/det|>No label"
    )
    blocks = uoe.layout_blocks(text, 999, 1998)
    assert [(b["label"], b["box"], b["content"]) for b in blocks] == [
        (None, None, "Loose text"),
        ("title", [10, 20, 30, 40], "## Head"),
        ("text", [999, 999, 500, 0], "Body\nmore"),
        ("text", None, "Bad box"),
        (None, None, "No label"),
    ]
    assert blocks[1]["box_px"] == [10, 40, 30, 80]
    assert blocks[2]["box_px"] == [500, 0, 999, 1998]  # corners sorted
    assert all("box_px" not in b for b in blocks[3:])
    start, end = blocks[1]["span"]
    assert text[start:end] == "<|det|>title [10, 20, 30, 40]<|/det|>"
    # Clipped to the image.
    (block,) = uoe.layout_blocks("<|det|>text [-5, 0, 1200, 999]<|/det|>x", 999, 999)
    assert block["box_px"] == [0, 0, 999, 999]


def test_reference_wrappers_are_not_transcribed():
    text = (
        "<|ref|>Running head<|/ref|>\n"
        "<|ref|>title<|/ref|><|det|>[[10, 20, 30, 40]]<|/det|>## Head\n"
        "<|ref|>text<|/ref|> <|det|>[100, 200, 300, 400]<|/det|>See <|ref|>Fig. 2<|/ref|> here.\n"
        "<|ref|>the fox<|/ref|><|det|>[[1, 2, 3, 4], [5, 6, 7, 8]]<|/det|>\n"
        "<|det|>text [0, 0, 999, 999]<|/det|>Stray <|/ref|>end<|grounding|>"
    )
    blocks = uoe.layout_blocks(text, 999, 999)
    assert [(b.get("ref"), b["label"], b["box"], b["content"]) for b in blocks] == [
        (None, None, None, "Running head"),
        ("title", "title", [10, 20, 30, 40], "## Head"),
        ("text", "text", [100, 200, 300, 400], "See Fig. 2 here."),
        ("the fox", None, None, "the fox"),  # referenced text is kept; no box from two boxes
        (None, "text", [0, 0, 999, 999], "Stray end"),
    ]
    start, end = blocks[1]["span"]
    assert text[start:end] == "<|ref|>title<|/ref|><|det|>[[10, 20, 30, 40]]<|/det|>"
    image = Image.new("RGB", (999, 999), "white")
    prediction = uoe.blocks_to_prediction(blocks, "s", image)
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(999, 999))
    assert issues == []
    assert "<|" not in standardized.text
    assert [r.text for r in standardized.regions] == [
        "Running head", "Head", "See Fig. 2 here.", "the fox", "Stray end"
    ]  # fmt: skip


def test_unterminated_tags_end_at_the_line_or_the_next_tag():
    text = "<|det|>broken\n<|det|>text [0, 0, 999, 999]<|/det|>Hello"
    blocks = uoe.layout_blocks(text, 999, 999)
    assert [(b["det"], b["label"], b["box"], b["content"]) for b in blocks] == [
        ("broken", "broken", None, ""),
        ("text [0, 0, 999, 999]", "text", [0, 0, 999, 999], "Hello"),
    ]
    assert blocks[0]["unterminated"] is True
    assert "unterminated" not in blocks[1]
    # A header without its closing delimiter, then the block's text on the next lines.
    text = "<|det|>title [10, 20, 30, 40]\nHead\nmore<|det|>text [1, 2"
    blocks = uoe.layout_blocks(text, 999, 999)
    assert [(b["label"], b["box"], b["content"]) for b in blocks] == [
        ("title", [10, 20, 30, 40], "Head\nmore"),
        (None, None, ""),  # a header cut off at the end of the output: no label or text
    ]
    assert text[slice(*blocks[1]["span"])] == "<|det|>text [1, 2"
    assert [b["unterminated"] for b in blocks] == [True, True]


def test_blocks_become_positioned_regions_with_markup():
    image = Image.new("RGB", (1000, 700), "white")
    raw = (FIXTURES / "output.txt").read_text(encoding="utf-8")
    prediction = uoe.blocks_to_prediction(uoe.layout_blocks(raw, *image.size), "s", image)
    regions = prediction["regions"]
    assert [r["native_type"] for r in regions] == [
        "page_number",
        "title",
        "text",
        "image",
        "image_caption",
        "table",
        "equation",
        "footer",
        "text",
    ]
    assert [r["type"] for r in regions] == [
        "page-number",
        "heading",
        "text",
        "figure",
        "caption",
        "table",
        "formula",
        "page-footer",
        "text",
    ]
    assert prediction["reading_order"] == [r["id"] for r in regions]
    corners = ([100 * 1000 / 999, 80 * 700 / 999], [900 * 1000 / 999, 80 * 700 / 999])
    assert regions[1]["polygon"][:2] == [pytest.approx(corner) for corner in corners]
    assert regions[1]["markup"] == {"format": "markdown", "content": "The **Daily** Chronicle"}
    assert regions[7]["polygon"][2] == [1000, 700]  # y 1010 is clipped to the image
    assert "markup" not in regions[3]  # the image block has no text
    assert "polygon" not in regions[8]  # no valid box: no position is invented
    assert prediction["text"] == ""
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(1000, 700))
    assert issues == []
    assert [r.text for r in standardized.regions if r.text] == FIXTURE_TEXTS
    assert standardized.text == "\n\n".join(FIXTURE_TEXTS)
    assert standardized.conversion == {"from": "markdown", "version": 1}


def test_markdown_is_the_model_cards_rendering():
    raw = (FIXTURES / "output.txt").read_text(encoding="utf-8")
    rendered = uoe.markdown(raw)
    assert rendered.startswith(
        "-12-\n\nThe **Daily** Chronicle\n\nThe quick brown fox jumps over\n"
    )
    assert "<table>" in rendered
    assert "\\[E = mc^2\\]" in rendered
    assert rendered.count("\n\n") == 7  # nine blocks, the image block dropped


def test_tail_repetition():
    assert uoe.tail_repeats("Intro " + "abc " * 8)
    assert not uoe.tail_repeats("Intro " + "abc " * 7)
    assert not uoe.tail_repeats(" " * 50)


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
    """A cache with fake model files at the pinned path, and checksums that match them."""
    directory = root / "cache" / uoe.PINNED_DIR
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model-00001-of-000001.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "config.json").write_text("changed", encoding="utf-8")
    if extra_file:
        (directory / "modeling_unlimitedocr.py").write_text("", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_UNLIMITED_OCR_CACHE": str(root / "cache"),
        "OCRBENCH_UNLIMITED_OCR_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000
# The fake server chooses its output by the gray level in the top-left corner (fake/README.md).
GRAY = {"page": 200, "notags": 150, "news": 100, "abort": 80, "boom": 50, "die": 20, "empty": 70}


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
        "OCRBENCH_UNLIMITED_OCR_LOG": str(tmp_path / "vllm-server.log"),
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
        "empty": ("empty", (300, 200)),
        "notags": ("notags", (500, 500)),
        "boom": ("boom", (666, 400)),
        "news": ("news", (2000, 3000)),
        "abort": ("abort", (300, 200)),
        "big": ("page", (5000, 7000)),
        "die": ("die", (500, 500)),
        "after": ("page", (1000, 700)),
    }
    run, records = run_adapter(tmp_path, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "empty": ("success", None),  # an empty output is an empty page
        "notags": ("success", None),  # text without tags: one region without a position
        "boom": ("failure", "engine-error"),
        "news": ("truncated", None),  # stopped at the end of the context, scored as produced
        "abort": ("failure", "engine-error"),
        "big": ("success", None),  # the adapter keeps going after failed samples
        "die": ("failure", "engine-error"),
        "after": ("failure", "engine-error"),
    }
    assert records["empty"]["message"] == "the model returned an empty page"
    assert "request failed: HTTP 500" in records["boom"]["message"]
    assert "fake CUDA out of memory" in records["boom"]["message"]
    assert records["abort"]["message"] == "generation failed (finish_reason 'abort')"
    assert records["after"]["message"].startswith("the vLLM server has exited")

    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "output.txt",
        "layout.json",
        "page.md",
        "prediction.json",
        "result.json",
    }
    native = (page / "output" / "output.txt").read_text(encoding="utf-8")
    assert native == (FIXTURES / "output.txt").read_text(encoding="utf-8")  # unchanged
    layout = json.loads((page / "output" / "layout.json").read_text())
    assert len(layout["blocks"]) == 9
    assert layout["image"] == {"width": 1000, "height": 700}
    assert records["page"]["findings"] == []
    engine_prediction = json.loads((page / "output" / "prediction.json").read_text())
    assert engine_prediction["text"] == ""
    assert "markup" in engine_prediction["regions"][0]
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": 1}  # converted on the host
    regions = prediction["regions"]
    assert [r["text"] for r in regions if "text" in r] == FIXTURE_TEXTS
    assert all("markup" not in r for r in regions)
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n\n".join(FIXTURE_TEXTS)

    extra = records["page"]["engine_extra"]
    assert (extra["finish_reason"], extra["tail_repetition"]) == ("stop", False)
    assert extra["generated_tokens"] == len(native)
    assert extra["prompt_tokens"] == (1000 // 16) * (700 // 16) + len(uoe.PROMPT)
    assert extra["max_tokens"] is None
    assert (extra["blocks"], extra["blocks_without_box"]) == (9, 1)
    assert "title" in extra["labels"]
    assert extra["request_seconds"] >= 0
    assert extra["tokens_per_second"] >= 0
    assert extra["source_image"] == {"width": 1000, "height": 700}
    environment = extra["environment"]
    assert (environment["vllm"], environment["transformers"]) == ("0.26.0+fake", "5.14.1+fake")
    assert (environment["gpu"], environment["gpu_capability"]) == ("Fake GPU", "8.9")
    assert environment["server_command"][:2] == ["vllm", "serve"]
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["repository"], models["revision"]) == (uoe.REPOSITORY, uoe.REVISION)
    assert set(models["files"]) == {"config.json", "model-00001-of-000001.safetensors"}
    assert models["prompt"] == "<image>document parsing."
    assert models["sampling"] == {"temperature": 0.0, "ngram_size": 35, "window_size": 128}
    assert extra["params"] == uoe.DEFAULT_PARAMS

    empty = run.path / "samples" / "empty" / "attempt-1"
    assert (empty / "prediction.txt").read_text(encoding="utf-8") == ""
    assert (empty / "output" / "output.txt").read_text(encoding="utf-8") == ""  # kept
    notags = run.path / "samples" / "notags" / "attempt-1"
    assert (notags / "prediction.txt").read_text(
        encoding="utf-8"
    ) == "This page has no layout tags."
    (region,) = json.loads((notags / "prediction.json").read_text())["regions"]
    assert "polygon" not in region
    assert region["type"] == "other"
    abort = run.path / "samples" / "abort" / "attempt-1" / "output" / "output.txt"
    assert abort.read_text(encoding="utf-8").endswith("Cut off")  # kept on failure
    abort_extra = records["abort"]["engine_extra"]  # the diagnostics are kept on failure
    assert (abort_extra["finish_reason"], abort_extra["tail_repetition"]) == ("abort", False)
    assert abort_extra["generated_tokens"] == len(abort.read_text(encoding="utf-8"))
    assert abort_extra["prompt_tokens"] == (300 // 16) * (200 // 16) + len(uoe.PROMPT)
    assert abort_extra["request_seconds"] >= 0
    assert abort_extra["blocks"] == 1
    assert extra["unterminated_tags"] == 0

    news = records["news"]
    prompt = (2000 // 16) * (3000 // 16) + len(uoe.PROMPT)
    assert news["message"] == (
        f"generation stopped at the end of the 32768-token context ({CONTEXT - prompt} tokens); "
        "the output ends in a repetition loop"
    )
    assert news["engine_extra"]["finish_reason"] == "length"
    assert news["engine_extra"]["tail_repetition"] is True
    news_text = (run.path / "samples" / "news" / "attempt-1" / "prediction.txt").read_text()
    assert news_text.startswith(_newspaper_line(1))
    assert _newspaper_line(99) in news_text

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


def test_max_tokens_truncation(tmp_path):
    params = dict(CONFIG["params"], max_tokens=500)
    _, records = run_adapter(tmp_path, {"news": ("news", (2000, 3000))}, params)
    news = records["news"]
    assert news["status"] == "truncated"
    assert news["message"] == "generation stopped at max_tokens (500 tokens)"
    assert news["engine_extra"]["tail_repetition"] is False
    assert news["engine_extra"]["max_tokens"] == 500


def test_max_tokens_beyond_the_context_left_by_the_image(tmp_path):
    # The 2000 x 3000 image takes 23,399 of the 32,768 tokens; vLLM generates the rest.
    params = dict(CONFIG["params"], max_tokens=32000)
    _, records = run_adapter(tmp_path, {"news": ("news", (2000, 3000))}, params)
    news = records["news"]
    prompt = (2000 // 16) * (3000 // 16) + len(uoe.PROMPT)
    assert news["status"] == "truncated"
    assert news["message"] == (
        f"generation stopped at the end of the 32768-token context ({CONTEXT - prompt} tokens); "
        "the output ends in a repetition loop"
    )
    assert news["engine_extra"]["max_tokens"] == 32000


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "corrupt", "not matching models.sha256: ['config.json']"),
        ({}, "extra", "model files not in models.sha256: ['modeling_unlimitedocr.py']"),
        ({}, "missing", "does not exist; download the weights with make unlimited-ocr-model"),
        ({}, {"FAKE_VLLM_EXIT": "1"}, "vLLM exited with status 3 during start-up"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env in ("corrupt", "extra"):
        env = model_cache(tmp_path / env, corrupt=env == "corrupt", extra_file=env == "extra")
    elif env == "missing":
        env = {"OCRBENCH_UNLIMITED_OCR_CACHE": str(tmp_path / "empty")}
    samples = {"a": ("page", (900, 420)), "b": ("empty", (300, 200))}
    _, records = run_adapter(tmp_path, samples, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "unlimited_ocr_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "MODELS", Path(env["OCRBENCH_UNLIMITED_OCR_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main([str(tmp_path / "cache")])
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{uoe.REPOSITORY}/resolve/{uoe.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / uoe.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs from models.sha256"):
        fetch_model.main([str(tmp_path / "cache")])
    assert sorted(p.name for p in (tmp_path / "cache" / uoe.PINNED_DIR).iterdir()) == [
        "model-00001-of-000001.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def unlimited_ocr_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / uoe.PINNED_DIR / "model-00001-of-000001.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make unlimited-ocr-model first")
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
def test_gpu_smoke_benchmark(tmp_path, unlimited_ocr_image):
    cache = str(unlimited_ocr_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "unlimited-ocr"]) == 0
    run = load_run(tmp_path / "runs" / "unlimited-ocr")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert records["page-1"]["status"] == "success"
    assert records["para-1"]["status"] == "success"
    assert records["blank-1"]["status"] in ("success", "truncated")
    extra = records["page-1"]["engine_extra"]
    environment = extra["environment"]
    assert environment["vllm"] == "0.26.0"
    assert environment["torch"] == "2.11.0+cu130"
    assert environment["gpu"]
    weights = CHECKSUMS[CHECKSUMS.index("model-00001-of-000001.safetensors") - 1]
    assert extra["models"]["files"]["model-00001-of-000001.safetensors"] == weights
    assert extra["finish_reason"] == "stop"
    for sid in ("page-1", "para-1"):
        prediction = json.loads(
            (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
        )
        assert prediction["conversion"] == {"from": "markdown", "version": 1}
        assert prediction["reading_order"]
        assert all("polygon" in region for region in prediction["regions"])
        assert "**" not in prediction["text"]  # markup converted on the host

    # Scoring and re-scoring read only stored outputs: no container runs.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["para-1"]["chars"]["rate"] < 0.1
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()
