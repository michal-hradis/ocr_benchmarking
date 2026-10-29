"""LightOnOCR-3 adapter (engines/lightonocr3): parameters and presets, the vLLM server command and
request, the grounding block parsing and conversion to a positioned prediction with region markup,
and the engine protocol in plain and grounding mode.

Offline tests import the adapter and run it as a host process with a fake ``vllm`` executable that
serves fixed outputs over the loopback interface and a fake ``torch`` module
(tests/fixtures/lightonocr3/fake). The outputs include hand-written grounding blocks
(tests/fixtures/lightonocr3/output.md) and a Markdown page (page.md), output without markers, an
empty output, a newspaper page cut off at the end of the context in a repetition loop, an aborted
request, a failed request and a server that exits. They also evaluate and report those runs, whose
markup is converted on the host. The opt-in GPU test builds the engine image and runs every preset
with the real models from the weights cache through ``ocrbench run``, ``evaluate`` and ``report``.
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
ENGINE_DIR = ROOT / "engines" / "lightonocr3"
FIXTURES = Path(__file__).parent / "fixtures" / "lightonocr3"
FAKE = FIXTURES / "fake"
ADAPTER_PATH = ENGINE_DIR / "lightonocr3_engine.py"
PRESET_FILES = [ENGINE_DIR / "engine.toml", *sorted((ENGINE_DIR / "presets").glob("*.toml"))]
PRESETS = {path.name: tomllib.loads(path.read_text(encoding="utf-8")) for path in PRESET_FILES}
CONFIG = PRESETS["engine.toml"]

_spec = importlib.util.spec_from_file_location("lightonocr3_engine", ADAPTER_PATH)
loe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loe)

# The region texts of tests/fixtures/lightonocr3/output.md after host conversion, in output order;
# the image and chart blocks have none. A Markdown image in the text adds nothing.
FIXTURE_TEXTS = [
    "-12-",
    "The Daily Chronicle",
    "The quick brown fox jumps over\nthe lazy dog & the river.",
    "It swam to the other bank.",
    "Figure 1: A fox.",
    "Year\tCopies\n1923\t2O26",
    "First item\nSecond item",
    "E = mc^2",
    "Footer",
    "A block that goes on after an invalid marker, see .",
]
# tests/fixtures/lightonocr3/page.md after host conversion.
PAGE_TEXTS = [
    "The Daily Chronicle",
    "The quick brown fox jumps over\nthe lazy dog & the river.",
    "Year\tCopies\n1923\t2O26",
]


def test_presets_name_their_checkpoint_and_mode():
    assert len(PRESETS) == 6
    seen = set()
    for name, config in PRESETS.items():
        params = config["params"]
        spec = loe.CHECKPOINTS[params["checkpoint"]]
        assert config["model"] == f"{spec['repository']}@{spec['revision']}"
        assert config["name"] == f"lightonocr3-{params['checkpoint'].lower()}-{params['mode']}"
        assert name in ("engine.toml", f"{params['checkpoint'].lower()}-{params['mode']}.toml")
        assert config["image"] == CONFIG["image"]
        assert loe.parse_params(params) == loe.DEFAULT_PARAMS | {
            "checkpoint": params["checkpoint"],
            "mode": params["mode"],
        }
        seen.add((params["checkpoint"], params["mode"]))
    assert seen == {(c, m) for c in ("0.8B", "1B", "4B") for m in ("plain", "grounding")}
    assert (CONFIG["params"]["checkpoint"], CONFIG["params"]["mode"]) == ("4B", "grounding")


@pytest.mark.parametrize("checkpoint", ["0.8B", "1B", "4B"])
def test_pinned_model_files(checkpoint):
    spec = loe.CHECKPOINTS[checkpoint]
    name = spec["repository"].removeprefix("lightonai/")
    assert loe.pinned_dir(checkpoint) == f"models/lightonocr3/{name}/{spec['revision']}"
    lines = (ENGINE_DIR / f"models-{checkpoint}.sha256").read_text().split()
    names = lines[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == ["model.safetensors"]
    assert {"config.json", "chat_template.jinja", "tokenizer.json", "processor_config.json"} <= set(
        names
    )
    assert not [n for n in names if n.endswith(".py")]  # vLLM's own model code, no remote code
    assert all(len(digest) == 64 for digest in lines[::2])
    # The weights' SHA-256 is the Hugging Face LFS object ID at the pinned revision.
    weights = lines[lines.index("model.safetensors") - 1]
    prefix = {"0.8B": "ae46aa6fd713", "1B": "32879861d5cd", "4B": "6b0dab74a2da"}[checkpoint]
    assert weights.startswith(prefix)


def test_server_command_has_lightons_settings():
    params = loe.parse_params({"checkpoint": "4B", "mode": "grounding"})
    command = loe.server_command(params, 1234)
    directory = f"/cache/models/lightonocr3/LightOnOCR-3-4B/{loe.CHECKPOINTS['4B']['revision']}"
    assert command[:7] == ["vllm", "serve", directory, "--host", "127.0.0.1", "--port", "1234"]
    options = " ".join(command[7:])
    for expected in (
        "--served-model-name lightonocr3 --dtype bfloat16",
        "--distributed-executor-backend uni --tensor-parallel-size 1",
        '--limit-mm-per-prompt {"image": 1} --mm-processor-cache-gb 0 --no-enable-prefix-caching',
        "--max-model-len 18000",
        '--default-chat-template-kwargs {"enable_thinking": false}',
        "--max-num-seqs 8 --gpu-memory-utilization 0.6",
    ):
        assert expected in options
    assert "--trust-remote-code" not in command
    one = loe.server_command(loe.parse_params({"checkpoint": "1B", "mode": "plain"}), 1)
    assert "--max-model-len 16384" in " ".join(one)
    assert "--default-chat-template-kwargs" not in one  # the 1B template has no thinking
    assert loe.server_env(params) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    other = loe.parse_params({"checkpoint": "4B", "mode": "plain", "device": "cuda:1"})
    assert loe.server_env(other)["CUDA_VISIBLE_DEVICES"] == "1"


BASE = {"checkpoint": "4B", "mode": "grounding"}


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"mode": "plain"}, "checkpoint must be one of 0.8B, 1B, 4B: None"),
        ({"checkpoint": "2B", "mode": "plain"}, "checkpoint must be one of"),
        ({"checkpoint": "4B"}, "mode must be one of plain, grounding: None"),
        (BASE | {"mode": "layout"}, "mode must be one of"),
        (BASE | {"prompt": "OCR"}, "unknown parameters: prompt"),
        (BASE | {"device": "cpu"}, "device"),
        (BASE | {"max_tokens": 0}, "max_tokens"),
        (BASE | {"max_tokens": 100.0}, "max_tokens"),
        (BASE | {"max_tokens": 18000}, "below the 18000-token context"),
        ({"checkpoint": "1B", "mode": "plain", "max_tokens": 16384}, "16384-token context"),
        (BASE | {"temperature": -1}, "temperature"),
        (BASE | {"top_p": 0}, "top_p"),
        (BASE | {"seed": 1.5}, "seed"),
        (BASE | {"longest_edge": 10}, "longest_edge"),
        (BASE | {"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        (BASE | {"gpu_memory_utilization": 0.9}, r"gpu_memory_utilization must be in \(0, 0.6\]"),
        (BASE | {"max_num_seqs": 0}, "max_num_seqs"),
        (BASE | {"startup_seconds": -1}, "startup_seconds"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(loe.AdapterError, match=message):
        loe.parse_params(params)


def test_params_defaults():
    assert loe.DEFAULT_PARAMS == {
        "checkpoint": None,
        "mode": None,
        "device": "cuda:0",
        "max_tokens": None,
        "temperature": 0.2,
        "top_p": 1.0,
        "seed": 0,
        "longest_edge": None,
        "gpu_memory_utilization": 0.6,
        "max_num_seqs": 8,
        "startup_seconds": 900,
    }
    params = loe.parse_params(BASE | {"max_tokens": None})
    assert params == loe.DEFAULT_PARAMS | BASE
    assert loe.longest_edge(params) == 2048
    assert loe.longest_edge({**params, "checkpoint": "1B"}) == 1540
    assert loe.longest_edge({**params, "longest_edge": 1000}) == 1000


def test_request_is_lightons_client_request():
    image = Image.new("RGB", (40, 30), "white")
    plain = loe.payload(image, loe.parse_params({"checkpoint": "1B", "mode": "plain"}))
    assert "max_tokens" not in plain  # vLLM generates up to the end of the context
    (picture,) = plain["messages"][0]["content"]  # the image alone
    assert picture["image_url"]["url"].startswith("data:image/png;base64,")
    assert (plain["temperature"], plain["top_p"], plain["seed"]) == (0.2, 1.0, 0)
    assert plain["chat_template_kwargs"] == {"enable_thinking": False}
    grounding = loe.payload(image, loe.parse_params(BASE | {"max_tokens": 500}))
    assert grounding["messages"][0]["content"][1] == {"type": "text", "text": "grounding"}
    assert grounding["max_tokens"] == 500


def test_fit_scales_down_only():
    assert loe.fit(Image.new("RGB", (4194, 6192)), 2048).size == (1387, 2048)
    assert loe.fit(Image.new("RGB", (1600, 6975)), 1540).size == (353, 1540)
    small = Image.new("RGB", (300, 200))
    assert loe.fit(small, 2048) is small


def test_layout_blocks_keep_boxes_as_written():
    text = (
        "Loose text\n![title](10,20,30,40) ## Head\n\n"
        "![text](1000,1000,500,0)\nBody\nmore\n\n"
        "![text](1,2,3) Bad box\n![text](1.5,2,3,4) Float box\n"
        "![section+](0, 0, 10, 10) Next\nSee ![Fig](fig.png) and ![image](1,2,3,4,5)."
    )
    blocks = loe.layout_blocks(text, 1000, 2000)
    # Only a box of four integers makes a marker; anything else is text of the block it is in.
    assert [(b["label"], b.get("continues"), b["box"], b["content"]) for b in blocks] == [
        (None, None, None, "Loose text"),
        ("title", False, [10, 20, 30, 40], "## Head"),
        ("text", False, [1000, 1000, 500, 0], "Body\nmore\n\n![text](1,2,3) Bad box\n"
         "![text](1.5,2,3,4) Float box"),
        ("section", True, [0, 0, 10, 10], "Next\nSee ![Fig](fig.png) and ![image](1,2,3,4,5)."),
    ]  # fmt: skip
    assert blocks[1]["box_px"] == [10, 40, 30, 80]
    assert blocks[2]["box_px"] == [500, 0, 1000, 2000]  # corners sorted
    assert "box_px" not in blocks[0]
    start, end = blocks[1]["span"]
    assert text[start:end] == "![title](10,20,30,40)"
    # Clipped to the image.
    (block,) = loe.layout_blocks("![text](0,0,1200,1000) x", 999, 999)
    assert block["box_px"] == [0, 0, 999, 999]
    assert loe.layout_blocks("", 10, 10) == []


def test_invalid_image_marker_keeps_the_text_scored():
    """Regression (PR #123 review): an image marker with a wrong number of coordinates used to
    split the text block and become an unscored picture block taking the rest of the text."""
    image = Image.new("RGB", (1000, 1000))
    for box in ("1,2,3", "1,2,3,4,5", "-1,2,3,4", "1.0,2,3,4", ""):
        raw = f"![text](0,0,500,500) First half ![image]({box}) second half."
        prediction = loe.blocks_to_prediction(loe.layout_blocks(raw, 1000, 1000), "s", image)
        (region,) = prediction["regions"]
        assert region["native_type"] == "text"
        assert region["markup"]["content"] == f"First half ![image]({box}) second half."


def test_blocks_become_positioned_regions_with_markup():
    image = Image.new("RGB", (1000, 700), "white")
    raw = (FIXTURES / "output.md").read_text(encoding="utf-8")
    prediction = loe.blocks_to_prediction(loe.layout_blocks(raw, *image.size), "s", image)
    regions = prediction["regions"]
    assert [r["native_type"] for r in regions] == [
        "page_number", "title", "text", "text+", "image", "caption", "table", "chart", "list",
        "formula", "footer", "text",
    ]  # fmt: skip
    assert [r["type"] for r in regions] == [
        "page-number", "heading", "text", "text", "figure", "caption", "table", "figure",
        "list-item", "formula", "page-footer", "text",
    ]  # fmt: skip
    assert prediction["reading_order"] == [r["id"] for r in regions]
    assert regions[1]["polygon"] == [[100, 56], [900, 56], [900, 84], [100, 84]]
    assert regions[1]["markup"] == {"format": "markdown", "content": "## The **Daily** Chronicle"}
    assert regions[10]["polygon"][2] == [1000, 700]  # y 1010 is clipped to the image
    # The picture description and the chart data are not page text; their boxes are kept.
    assert "markup" not in regions[4]
    assert "polygon" in regions[4]
    assert "markup" not in regions[7]
    assert "polygon" in regions[7]
    # An invalid marker inside a block is its text, not a picture block without text.
    assert regions[11]["native_type"] == "text"
    assert "![image](1,2,3)" in regions[11]["markup"]["content"]
    assert prediction["text"] == ""
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(1000, 700))
    assert issues == []
    assert [r.text for r in standardized.regions if r.text] == FIXTURE_TEXTS
    assert standardized.text == "\n\n".join(FIXTURE_TEXTS)
    assert standardized.conversion == {"from": "markdown", "version": 1}


def test_tail_repetition():
    assert loe.tail_repeats("Intro " + "abc " * 8)
    assert not loe.tail_repeats("Intro " + "abc " * 7)
    assert not loe.tail_repeats(" " * 50)
    sentence = "A long looping sentence of a newspaper column, " * 2  # 94 characters
    assert loe.tail_repeats("Intro " + sentence * 3 + sentence[:40])
    assert not loe.tail_repeats("Intro " + sentence * 2 + sentence[:40])


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


def model_cache(root, checkpoint, corrupt=False, extra_file=False):
    """A cache with fake model files at the checkpoint's pinned path, and checksums that match."""
    directory = root / "cache" / loe.pinned_dir(checkpoint)
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "config.json").write_text("changed", encoding="utf-8")
    if extra_file:
        (directory / "modeling_lightonocr.py").write_text("", encoding="utf-8")
    checksums = root / "checksums"
    checksums.mkdir()
    (checksums / f"models-{checkpoint}.sha256").write_text("\n".join(lines) + "\n", "utf-8")
    return {
        "OCRBENCH_LIGHTONOCR3_CACHE": str(root / "cache"),
        "OCRBENCH_LIGHTONOCR3_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000
# The fake server chooses its output by the gray level in the top-left corner (fake/README.md).
GRAY = {"page": 200, "nomarkers": 150, "news": 100, "abort": 80, "boom": 50, "die": 20, "empty": 70}


def make_dataset(root, samples, texts):
    data = root / "data"
    data.mkdir()
    entries = []
    for sid, (kind, size) in samples.items():
        page_image(GRAY[kind], size).save(data / f"{sid}.png")
        if kind == "news":  # a newspaper page whose reference has 2,000 lines
            lines = [_newspaper_line(n) for n in range(1, NEWSPAPER_LINES + 1)]
            regions = [[(text, (10, n, 20, n + 1)) for n, text in enumerate(lines)]]
        elif kind == "page":
            lines = [line for text in texts for line in text.replace("\t", " ").split("\n")]
            regions = [[(text, (10, 10, 20, 20))] for text in lines]
        else:
            regions = []
        page_xml(data / f"{sid}.xml", regions, size)
        entries.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, entries)


def _newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def run_adapter(tmp_path, samples, params=None, env=None, config=CONFIG):
    config = dict(config, params=params if params is not None else config["params"])
    config["timeouts"] = {"sample": 60, "startup": 30}
    del config["resources"]
    checkpoint = config["params"].get("checkpoint") or "4B"
    texts = FIXTURE_TEXTS if config["params"].get("mode") == "grounding" else PAGE_TEXTS
    env = {
        "PYTHONPATH": str(FAKE),
        "PATH": f"{FAKE / 'bin'}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}"
        + os.environ["PATH"],
        "OCRBENCH_LIGHTONOCR3_LOG": str(tmp_path / "vllm-server.log"),
        **model_cache(tmp_path, checkpoint),
        **(env or {}),
    }
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, samples, texts),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in samples}


def test_grounding_follows_the_engine_protocol(tmp_path):
    samples = {
        "page": ("page", (1000, 700)),
        "empty": ("empty", (300, 200)),
        "nomarkers": ("nomarkers", (500, 500)),
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
        "nomarkers": ("success", None),  # text without markers: one region without a position
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
        "output.md",
        "layout.json",
        "prediction.json",
        "result.json",
    }
    native = (page / "output" / "output.md").read_text(encoding="utf-8")
    assert native == (FIXTURES / "output.md").read_text(encoding="utf-8")  # unchanged
    layout = json.loads((page / "output" / "layout.json").read_text())
    assert len(layout["blocks"]) == 12
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
    assert extra["prompt_tokens"] == (1000 // 32) * (700 // 32) + 10
    assert extra["max_tokens"] is None
    assert (extra["blocks"], extra["blocks_without_box"]) == (12, 0)
    assert (extra["continuation_blocks"], extra["unscored_blocks"]) == (1, 2)
    assert "title" in extra["labels"]
    assert extra["request_seconds"] >= 0
    assert extra["tokens_per_second"] >= 0
    assert extra["source_image"] == extra["sent_image"] == {"width": 1000, "height": 700}
    big = records["big"]["engine_extra"]
    assert big["sent_image"] == {"width": 1463, "height": 2048}  # scaled down to the longest edge
    big_prediction = json.loads(
        (run.path / "samples" / "big" / "attempt-1" / "prediction.json").read_text()
    )
    assert big_prediction["image"] == {"width": 5000, "height": 7000}  # boxes in source pixels
    assert big_prediction["regions"][1]["polygon"][0] == [500, 560]
    environment = extra["environment"]
    assert (environment["vllm"], environment["transformers"]) == ("0.30.0+fake", "5.16.1+fake")
    assert (environment["gpu"], environment["gpu_capability"]) == ("Fake GPU", "8.9")
    assert environment["server_command"][:2] == ["vllm", "serve"]
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["repository"], models["revision"]) == (
        "lightonai/LightOnOCR-3-4B",
        loe.CHECKPOINTS["4B"]["revision"],
    )
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert (models["mode"], models["prompt"], models["longest_edge"]) == (
        "grounding",
        "grounding",
        2048,
    )
    assert models["chat_template_kwargs"] == {"enable_thinking": False}
    assert extra["params"] == loe.DEFAULT_PARAMS | BASE

    empty = run.path / "samples" / "empty" / "attempt-1"
    assert (empty / "prediction.txt").read_text(encoding="utf-8") == ""
    assert (empty / "output" / "output.md").read_text(encoding="utf-8") == ""  # kept
    nomarkers = run.path / "samples" / "nomarkers" / "attempt-1"
    assert (nomarkers / "prediction.txt").read_text(encoding="utf-8") == (
        "This page has no layout markers."
    )
    (region,) = json.loads((nomarkers / "prediction.json").read_text())["regions"]
    assert "polygon" not in region
    assert region["type"] == "other"
    abort = run.path / "samples" / "abort" / "attempt-1" / "output" / "output.md"
    assert abort.read_text(encoding="utf-8").endswith("Cut off")  # kept on failure
    abort_extra = records["abort"]["engine_extra"]  # the diagnostics are kept on failure
    assert (abort_extra["finish_reason"], abort_extra["tail_repetition"]) == ("abort", False)
    assert abort_extra["generated_tokens"] == len(abort.read_text(encoding="utf-8"))
    assert abort_extra["blocks"] == 1

    news = records["news"]
    prompt = (1365 // 32) * (2048 // 32) + 10
    assert news["engine_extra"]["sent_image"] == {"width": 1365, "height": 2048}
    assert news["message"] == (
        f"generation stopped at the end of the 18000-token context ({18000 - prompt} tokens); "
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


def test_plain_mode_writes_markdown_converted_on_the_host(tmp_path):
    config = PRESETS["1b-plain.toml"]
    samples = {
        "page": ("page", (1000, 700)),
        "big": ("page", (5000, 7000)),
        "news": ("news", (2000, 3000)),
        "empty": ("empty", (300, 200)),
    }
    run, records = run_adapter(tmp_path, samples, config=config)
    got = {sid: r["status"] for sid, r in records.items()}
    assert got == {"page": "success", "big": "success", "news": "truncated", "empty": "success"}
    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {"page.md", "result.json"}
    native = (page / "output" / "page.md").read_text(encoding="utf-8")
    assert native == (FIXTURES / "page.md").read_text(encoding="utf-8")  # unchanged
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": 1}
    assert all("polygon" not in r for r in prediction["regions"])  # no positions are invented
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n\n".join(PAGE_TEXTS)
    extra = records["page"]["engine_extra"]
    assert "blocks" not in extra
    assert extra["models"]["prompt"] == ""
    assert extra["models"]["context"] == 16384
    assert records["big"]["engine_extra"]["sent_image"] == {"width": 1100, "height": 1540}
    news = records["news"]
    prompt = (1027 // 28) * (1540 // 28) + 10
    assert news["message"] == (
        f"generation stopped at the end of the 16384-token context ({16384 - prompt} tokens); "
        "the output ends in a repetition loop"
    )
    assert main(["evaluate", str(run.path)]) == 0


def test_max_tokens_truncation(tmp_path):
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
        (BASE | {"psm": 3}, {}, "unknown parameters: psm"),
        (BASE | {"checkpoint": "1B"}, {}, "does not match checkpoint 1B"),
        ({"mode": "plain"}, {}, "checkpoint must be one of"),
        (BASE, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        (BASE, "corrupt", "not matching models-4B.sha256: ['config.json']"),
        (BASE, "extra", "model files not in models-4B.sha256: ['modeling_lightonocr.py']"),
        (BASE, "missing", "does not exist; download the weights with make lightonocr3-model"),
        (BASE, {"FAKE_VLLM_EXIT": "1"}, "vLLM exited with status 3 during start-up"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env in ("corrupt", "extra"):
        env = model_cache(tmp_path / env, "4B", corrupt=env == "corrupt", extra_file=env == "extra")
    elif env == "missing":
        env = {"OCRBENCH_LIGHTONOCR3_CACHE": str(tmp_path / "empty")}
    samples = {"a": ("page", (900, 420)), "b": ("empty", (300, 200))}
    _, records = run_adapter(tmp_path, samples, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "lightonocr3_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path, "1B")
    monkeypatch.setattr(fetch_model, "HERE", Path(env["OCRBENCH_LIGHTONOCR3_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main([str(tmp_path / "cache"), "1B"])
    assert capsys.readouterr().out.count("ok      ") == 2

    revision = loe.CHECKPOINTS["1B"]["revision"]

    def wrong_download(url, path):
        assert url.startswith(
            f"https://huggingface.co/lightonai/LightOnOCR-3-1B/resolve/{revision}/"
        )
        Path(path).write_text("wrong", encoding="utf-8")

    directory = tmp_path / "cache" / loe.pinned_dir("1B")
    (directory / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs from models-1B.sha256"):
        fetch_model.main([str(tmp_path / "cache"), "1B"])
    assert sorted(p.name for p in directory.iterdir()) == ["model.safetensors"]
    with pytest.raises(SystemExit, match="Usage"):
        fetch_model.main([str(tmp_path / "cache"), "2B"])


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def lightonocr3_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    for checkpoint in loe.CHECKPOINTS:
        weights = cache / loe.pinned_dir(checkpoint) / "model.safetensors"
        if not weights.is_file():
            pytest.fail(f"no weights at {weights}: run make lightonocr3-model first")
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
@pytest.mark.parametrize("preset", [path.name for path in PRESET_FILES])
def test_gpu_smoke_benchmark(tmp_path, lightonocr3_image, preset):
    cache = str(lightonocr3_image)
    config = PRESETS[preset]
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(next(path for path in PRESET_FILES if path.name == preset))
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "lightonocr3"]) == 0
    run = load_run(tmp_path / "runs" / "lightonocr3")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert records["page-1"]["status"] == "success"
    assert records["para-1"]["status"] == "success"
    assert records["blank-1"]["status"] in ("success", "truncated")
    extra = records["page-1"]["engine_extra"]
    environment = extra["environment"]
    assert environment["vllm"] == "0.30.0"
    assert environment["torch"] == "2.13.0+cu130"
    assert environment["gpu"]
    checkpoint = config["params"]["checkpoint"]
    lines = (ENGINE_DIR / f"models-{checkpoint}.sha256").read_text().split()
    weights = lines[lines.index("model.safetensors") - 1]
    assert extra["models"]["files"]["model.safetensors"] == weights
    assert extra["finish_reason"] == "stop"
    for sid in ("page-1", "para-1"):
        prediction = json.loads(
            (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
        )
        assert prediction["conversion"] == {"from": "markdown", "version": 1}
        assert "**" not in prediction["text"]  # markup converted on the host
        if config["params"]["mode"] == "grounding":
            assert prediction["reading_order"]
            assert all("polygon" in region for region in prediction["regions"])

    # Scoring and re-scoring read only stored outputs: no container runs.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["para-1"]["chars"]["rate"] < 0.1
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()
