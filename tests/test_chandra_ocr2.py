"""Chandra OCR 2 adapter (engines/chandra-ocr2): parameters, prompt, image size, layout blocks,
generation and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and
``transformers`` modules (tests/fixtures/chandra-ocr2/fake) that "generate" fixed outputs,
including a hand-written layout HTML page with a header, an image, an HTML table and a footer
(tests/fixtures/chandra-ocr2/page.html), whose image description is left out of the scored text,
and a newspaper page whose output never ends. They also
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
ENGINE_DIR = ROOT / "engines" / "chandra-ocr2"
FIXTURES = Path(__file__).parent / "fixtures" / "chandra-ocr2"
ADAPTER_PATH = ENGINE_DIR / "chandra_ocr2_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
# SHA-256 of OCR_LAYOUT_PROMPT in chandra-ocr 0.2.0 (chandra/prompts.py, wheel SHA-256
# 11a9fffaa1afa8fb3be470205d4f8f0fd601c007f76e77aee76a7a290b8a8ed4).
UPSTREAM_PROMPT_SHA256 = "025935f3e1de1acdfadd4c7d581ab17eb82e8caaffef7b64962621c80b7ca9a8"

_spec = importlib.util.spec_from_file_location("chandra_ocr2_engine", ADAPTER_PATH)
co = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(co)

# The text of every block of page.html, header and footer included, as converted on the host.
FIXTURE_LINES = [
    "Page 3 · Thursday",
    "The Daily Chronicle",
    "The quick brown fox jumps\nover the lazy dog near the river.",
    "A second paragraph with more text and bold and italic words.",
    "Year\tCopies\n1923\t2026",
    "3",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{co.REPOSITORY}@{co.REVISION}"
    assert co.PINNED_DIR.endswith(co.REVISION)
    names = CHECKSUMS[1::2]
    for name in ("model.safetensors", "generation_config.json", "chat_template.jinja", "LICENSE"):
        assert name in names
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_prompt_is_the_official_ocr_layout_one():
    assert hashlib.sha256(co.PROMPT.encode()).hexdigest() == UPSTREAM_PROMPT_SHA256
    assert co.PROMPT.startswith("OCR this image to HTML, arranged as layout blocks.")
    assert co.CONVERSATION[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": co.PROMPT},
    ]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"prompt_type": "ocr"}, "unknown parameters: prompt_type"),
        ({"device": "gpu:0"}, "device"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 100.0}, "max_new_tokens"),
        ({"max_new_tokens": 262144}, "less than the context"),
        ({"temperature": 0.7}, "unknown parameters: temperature"),  # greedy only
        ({"min_image_dim": -1}, "min_image_dim must be an integer of at least 0"),
        ({"min_image_dim": True}, "min_image_dim"),
        ({"model_dir": "/cache/models/x"}, "model_dir"),
        ({"model_dir": "models/../../etc"}, "model_dir"),
        ({"model_dir": ""}, "model_dir"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(co.AdapterError, match=message):
        co.parse_params(params)


def test_params_defaults():
    assert co.parse_params({"max_new_tokens": None}) == co.DEFAULT_PARAMS
    assert co.DEFAULT_PARAMS["model_dir"] == co.PINNED_DIR
    assert co.DEFAULT_PARAMS["max_new_tokens"] == 12384  # MAX_OUTPUT_TOKENS of chandra-ocr
    assert co.DEFAULT_PARAMS["min_image_dim"] == 1536  # MIN_IMAGE_DIM of chandra-ocr
    params = co.parse_params({"max_new_tokens": 100, "device": "cpu", "min_image_dim": 0})
    assert (params["max_new_tokens"], params["device"], params["min_image_dim"]) == (100, "cpu", 0)


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        # Values of chandra.model.util.scale_to_fit of chandra-ocr 0.2.0.
        ((2480, 3508), (2100, 2968)),  # A4 at 300 dpi: scaled down to at most 3072 x 2048 pixels
        ((5000, 7000), (2100, 2968)),
        ((1600, 1700), (1596, 1708)),  # only rounded to whole 28-pixel units
        ((900, 420), (896, 420)),
        ((100, 100), (224, 224)),  # scaled up to at least 1792 x 28 pixels
        ((3000, 100), (2996, 112)),
        ((20000, 10), (19992, 28)),  # at least one unit on each side
        ((3500, 1900), (3388, 1848)),
        ((100000, 200), (56084, 112)),
    ],
)
def test_fit_size_matches_scale_to_fit(size, expected):
    assert co.fit_size(*size) == expected
    w, h = expected
    assert (w % 28, h % 28) == (0, 0)
    assert w * h <= 3072 * 2048


@pytest.mark.parametrize(
    ("size", "min_dim", "expected"),
    [
        ((900, 420), 1536, (3304, 1540)),  # upscaled to a shorter side of 1536, then fitted
        ((300, 200), 1536, (2296, 1540)),
        ((1600, 1700), 1536, (1596, 1708)),  # both sides at least 1536: not upscaled
        ((1600, 1200), 1536, (2044, 1540)),
        ((900, 420), 0, (896, 420)),  # 0 turns the upscaling off
        ((2100, 2968), 1536, (2100, 2968)),  # already of the right size: the same image
    ],
)
def test_prepare_image_upscales_like_the_cli_and_fits(size, min_dim, expected):
    image = Image.new("RGB", size)
    prepared = co.prepare_image(image, min_dim)
    assert prepared.size == expected
    assert (prepared is image) == (size == expected)


def test_undecodable_image_is_an_adapter_error(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(co.AdapterError, match="cannot decode image x.png"):
        co.read_image(tmp_path / "x.png")
    Image.new("LA", (4, 3)).save(tmp_path / "la.png")
    assert co.read_image(tmp_path / "la.png").mode == "RGB"


def test_layout_blocks_keep_top_level_divs_and_scale_boxes():
    text = (
        '<div data-bbox="100 250 900 1000" data-label="Text"><p>a &amp; b</p>\n'
        "<div>nested</div></div>\n"
        '<br><div data-label="Page-Header" data-bbox="1 2 3">h</div>'
        '<div data-bbox="0 0 x 1" data-label="Image"><img alt="a > b"/></div>'
        "<div>no attributes</div>"
        '<div data-bbox="10 20 30 40" data-label="Text">cut <b>off'
    )
    blocks = co.layout_blocks(text, 2480, 3508)
    assert [b["label"] for b in blocks] == ["Text", "Page-Header", "Image", None, "Text"]
    assert blocks[0]["bbox"] == [100, 250, 900, 1000]  # the model's own box is kept
    assert blocks[0]["bbox_px"] == [248.0, 877.0, 2232.0, 3508.0]
    assert blocks[0]["content"] == "<p>a &amp; b</p>\n<div>nested</div>"  # inner HTML, as written
    # Boxes that are not four integers stay as written, without pixel boxes.
    assert [b["bbox"] for b in blocks[1:4]] == ["1 2 3", "0 0 x 1", None]
    assert not any("bbox_px" in b for b in blocks[1:4])
    assert blocks[2]["content"] == '<img alt="a > b"/>'
    assert [b["complete"] for b in blocks] == [True, True, True, True, False]
    assert blocks[4]["content"] == "cut <b>off"  # cut off: kept to the end of the output
    assert blocks[4]["bbox_px"] == pytest.approx([24.8, 70.16, 74.4, 140.32])
    # The whole block, tags included, as character offsets in the output.
    assert [text[slice(*b["span"])][:5] for b in blocks] == [*["<div "] * 3, "<div>", "<div "]
    assert text[slice(*blocks[0]["span"])].endswith("nested</div></div>")
    assert blocks[4]["span"][1] == len(text)
    assert co.layout_blocks("", 10, 10) == []
    assert co.layout_blocks("plain text", 10, 10) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('<div data-label="Text">a</div>', '<div data-label="Text">a</div>'),
        (
            '<div data-label="Text">a</div>\n<div data-label="Image"><img alt="x"/>A cat.</div >\n'
            '<div data-label="Figure"><div>chart</div></div><div data-label="Diagram">d</div>',
            '<div data-label="Text">a</div>\n\n<div data-label="Diagram">d</div>',
        ),
        ('t <div data-label="Image">cut off <b>', "t "),  # a cut-off image block too
        ("<p>no blocks</p>", "<p>no blocks</p>"),
    ],
)
def test_image_and_figure_blocks_are_left_out(text, expected):
    assert co.without_images(text, co.layout_blocks(text, 10, 10)) == expected


def model_cache(root, corrupt=False, extra=None):
    """A cache with fake model files at the pinned path, and checksums that match them; ``extra``
    adds a file that is not in the checksums."""
    directory = root / "cache" / co.PINNED_DIR
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
        "OCRBENCH_CHANDRA_OCR2_CACHE": str(root / "cache"),
        "OCRBENCH_CHANDRA_OCR2_CHECKSUMS": str(checksums),
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
            lines = [line.replace("\t", " ") for text in FIXTURE_LINES for line in text.split("\n")]
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
        "prose": (1200, 400),
        "blankpage": (1280, 400),
        "news": (2000, 3000),
        "empty": (960, 420),
    }
    run, records = run_adapter(tmp_path, sizes)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # only an end token: a genuine empty result
        "boom": ("failure", "engine-error"),
        "prose": ("success", None),  # no layout blocks: the text is scored as it is
        "blankpage": ("success", None),
        "news": ("truncated", None),  # stopped at max_new_tokens, scored as produced
        "empty": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: CUDA out of memory" in records["boom"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "output.html",
        "layout.json",
        "page.html",
        "result.json",
    }
    native = (page / "output" / "output.html").read_text(encoding="utf-8")
    assert native == (FIXTURES / "page.html").read_text(encoding="utf-8")  # unchanged
    layout = json.loads((page / "output" / "layout.json").read_text(encoding="utf-8"))
    assert layout["image"] == {"width": 900, "height": 420}
    assert [b["label"] for b in layout["blocks"]] == [
        "Page-Header",
        "Section-Header",
        "Text",
        "Image",
        "Text",
        "Table",
        "Page-Footer",
    ]
    # Boxes in original pixels: [0, 1000] of the width and height of the stored image.
    assert layout["blocks"][3]["bbox"] == [120, 280, 880, 540]
    assert layout["blocks"][3]["bbox_px"] == pytest.approx([108, 117.6, 792, 226.8])
    assert all(b["complete"] for b in layout["blocks"])
    assert "drawn in black ink" in layout["blocks"][3]["content"]  # kept in the native outputs
    # The primary output is the native one without the Image block.
    image_span = slice(*layout["blocks"][3]["span"])
    assert (page / "output" / "page.html").read_text(encoding="utf-8") == native.replace(
        native[image_span], ""
    )
    # Converted on the host: no HTML markup in the text, header and footer included, the image's
    # alt text and description left out, regions without positions.
    text = (page / "prediction.txt").read_text(encoding="utf-8")
    assert text == "\n\n".join(FIXTURE_LINES)
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"]["from"] == "html"
    types = [r["type"] for r in prediction["regions"]]
    assert types == ["text", "heading", "text", "text", "table", "text"]
    assert not any("polygon" in r or "lines" in r for r in prediction["regions"])
    assert "image" not in prediction

    extra = records["page"]["engine_extra"]
    assert extra["stop_reason"] == "end-token"
    assert extra["generated_tokens"] == len(native)  # one fake token per character
    assert (extra["blocks"], extra["image_blocks"]) == (7, 1)
    # Upscaled to a shorter side of 1536 and fitted to 28-pixel units (as chandra-ocr), then
    # resized to 32-pixel units by the processor.
    assert extra["scaled_image"] == {"width": 3304, "height": 1540}
    assert extra["model_image"] == {"width": 3296, "height": 1536}
    assert extra["image_tokens"] == 103 * 48  # 32 x 32 pixels each
    assert extra["prompt_tokens"] == 103 * 48 + 5
    assert extra["max_new_tokens"] == 12384
    assert extra["gpu_peak_memory_mb"] == 3072
    assert extra["environment"]["transformers"] == "5.19.0+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["verified_official_checkpoint"] is True
    assert (models["repository"], models["revision"]) == (co.REPOSITORY, co.REVISION)
    assert models["model_dir"] == co.PINNED_DIR
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert models["prompt"] == co.CONVERSATION
    assert models["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["params"] == co.DEFAULT_PARAMS

    prose = run.path / "samples" / "prose" / "attempt-1"
    assert (prose / "prediction.txt").read_text(encoding="utf-8") == "Sorry, this is plain text."
    assert records["prose"]["engine_extra"]["blocks"] == 0
    # A Blank-Page block has no text; <|endoftext|> also ends generation.
    blankpage = records["blankpage"]["engine_extra"]
    assert (blankpage["stop_reason"], blankpage["blocks"]) == ("end-token", 1)
    blankpage_dir = run.path / "samples" / "blankpage" / "attempt-1"
    assert (blankpage_dir / "prediction.txt").read_text(encoding="utf-8") == ""
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""
    assert records["blank"]["engine_extra"]["generated_tokens"] == 0

    news = records["news"]
    assert news["message"] == "generation stopped at max_new_tokens (12384 tokens)"
    assert news["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert news["engine_extra"]["generated_tokens"] == 12384
    assert news["engine_extra"]["scaled_image"] == {"width": 1988, "height": 2996}
    assert news["engine_extra"]["model_image"] == {"width": 1984, "height": 3008}
    news_dir = run.path / "samples" / "news" / "attempt-1"
    blocks = json.loads((news_dir / "output" / "layout.json").read_text())["blocks"]
    assert news["engine_extra"]["blocks"] == len(blocks)
    assert all(b["complete"] for b in blocks[:-1])  # only the last one can be cut off
    kept = (news_dir / "prediction.txt").read_text().split("\n\n")
    # Every block whose start tag was generated, the last one as far as it was generated.
    assert len(kept) == len(blocks)
    assert kept[:-1] == [_newspaper_line(n) for n in range(1, len(kept))]
    assert _newspaper_line(len(kept)).startswith(kept[-1])

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


def test_min_image_dim_cpu_and_context_limit(tmp_path):
    params = {"device": "cpu", "min_image_dim": 0}
    _, records = run_adapter(tmp_path, {"page": (900, 420)}, params, {"FAKE_TORCH_GPUS": "0"})
    extra = records["page"]["engine_extra"]
    assert extra["scaled_image"] == {"width": 896, "height": 420}  # not upscaled
    assert extra["model_image"] == {"width": 896, "height": 416}
    assert extra["image_tokens"] == 28 * 13
    assert "gpu" not in extra["environment"]
    assert "gpu_peak_memory_mb" not in extra
    # The 896-pixel image is not the fixture page's (3304 pixels): only an end token.
    assert records["page"]["status"] == "success"

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
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_CHANDRA_OCR2_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{co.REPOSITORY}/resolve/{co.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / co.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / co.PINNED_DIR).iterdir()) == [
        "model.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def chandra_ocr2_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / co.PINNED_DIR / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make chandra-ocr2-model first")
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
def test_gpu_smoke_benchmark(tmp_path, chandra_ocr2_image):
    cache = str(chandra_ocr2_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "chandra-ocr2"]) == 0
    run = load_run(tmp_path / "runs" / "chandra-ocr2")
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
    assert extra["blocks"] > 0
    assert extra["gpu_peak_memory_mb"] > 0
    layout = run.path / "samples" / "page-1" / "attempt-1" / "output" / "layout.json"
    blocks = json.loads(layout.read_text(encoding="utf-8"))["blocks"]
    assert all(len(b["bbox_px"]) == 4 for b in blocks)

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

    # A small max_new_tokens truncates the page; what was generated is scored.
    limited = tmp_path / "engine.toml"
    config = Path(engine).read_text(encoding="utf-8")
    limited.write_text(config.replace("max_new_tokens = 12384", "max_new_tokens = 64"), "utf-8")
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["message"] == "generation stopped at max_new_tokens (64 tokens)"
    assert page["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert page["engine_extra"]["generated_tokens"] == 64
    attempt = "samples/page-1/attempt-1/output/output.html"
    partial = (short_run.path / attempt).read_text(encoding="utf-8")
    assert (run.path / attempt).read_text(encoding="utf-8").startswith(partial)
    assert main(["evaluate", str(short_run.path)]) == 0
    scores = json.loads((short_run.path / "evaluations" / "1" / "scores.json").read_text())
    short_rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < short_rates["page-1"] <= 1  # scored as produced

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make chandra-ocr2-model" in r["message"] for r in records)
