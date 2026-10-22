"""Infinity-Parser2-Flash adapter (engines/infinityparser2): parameters, prompt, image size, layout
JSON parsing, generation and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and
``transformers`` modules (tests/fixtures/infinityparser2/fake) that "generate" fixed outputs,
including a hand-written layout JSON page with a header, a figure, an HTML table and a footer
(tests/fixtures/infinityparser2/page.json) and a newspaper page whose output never ends. They also
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
ENGINE_DIR = ROOT / "engines" / "infinityparser2"
FIXTURES = Path(__file__).parent / "fixtures" / "infinityparser2"
ADAPTER_PATH = ENGINE_DIR / "infinityparser2_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
# SHA-256 of PROMPT_DOC2JSON in infinity_parser2 0.4.0 (infinity_parser2/prompts.py), which is
# also the prompt of the model card of infly/Infinity-Parser2-Flash at revision
# cfddc4106b0abc4706f05575a164f7dd35d09c46.
UPSTREAM_PROMPT_SHA256 = "a1c4c81e3bce7b2eb6534231962187ed766af7d450867c61fd474a5e897fcb2c"

_spec = importlib.util.spec_from_file_location("infinityparser2_engine", ADAPTER_PATH)
ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ip)

# The text of every element of page.json, header and footer included, as converted on the host.
FIXTURE_LINES = [
    "Page 3 · Thursday",
    "The Daily Chronicle",
    "The quick brown fox jumps over the lazy dog near the river.",
    "A second paragraph with more text and a formula E = mc^2.",
    "Year\tCopies",
    "1923\t2026",
    "3",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{ip.REPOSITORY}@{ip.REVISION}"
    assert ip.PINNED_DIR.endswith(ip.REVISION)
    names = CHECKSUMS[1::2]
    assert "model.safetensors" in names
    assert "processor_config.json" in names
    assert "chat_template.jinja" in names
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_prompt_is_the_official_doc2json_one():
    assert hashlib.sha256(ip.PROMPT.encode()).hexdigest() == UPSTREAM_PROMPT_SHA256
    assert ip.PROMPT.startswith("\n- Extract layout information from the provided PDF image.")
    assert ip.CONVERSATION[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": ip.PROMPT},
    ]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"task": "doc2md"}, "unknown parameters: task"),
        ({"device": "gpu:0"}, "device"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 100.0}, "max_new_tokens"),
        ({"max_new_tokens": 262144}, "less than the context"),
        ({"temperature": 0.7}, "unknown parameters: temperature"),  # greedy only
        ({"max_pixels": 2047}, "max_pixels"),
        ({"max_pixels": 2048}, "max_pixels must be an integer of at least 204800"),
        ({"max_pixels": True}, "max_pixels"),
        ({"model_dir": "/cache/models/x"}, "model_dir"),
        ({"model_dir": "models/../../etc"}, "model_dir"),
        ({"model_dir": ""}, "model_dir"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(ip.AdapterError, match=message):
        ip.parse_params(params)


def test_params_defaults():
    assert ip.parse_params({"max_new_tokens": None}) == ip.DEFAULT_PARAMS
    assert ip.DEFAULT_PARAMS["model_dir"] == ip.PINNED_DIR
    assert ip.DEFAULT_PARAMS["max_pixels"] == 4096 * 4096  # as in infinity_parser2
    assert ip.DEFAULT_PARAMS["max_new_tokens"] == 32768
    assert ip.MIN_PIXELS == 2048
    params = ip.parse_params({"max_new_tokens": 100, "device": "cpu"})
    assert (params["max_new_tokens"], params["device"]) == (100, "cpu")


@pytest.mark.parametrize(
    ("size", "max_pixels", "expected"),
    [
        ((2480, 3508), 16777216, (2496, 3520)),  # A4 at 300 dpi: rounded, not scaled
        ((5000, 7000), 16777216, (3456, 4832)),  # scaled down to at most max_pixels
        ((900, 420), 16777216, (896, 416)),
        ((40, 30), 16777216, (64, 64)),  # scaled up to at least MIN_PIXELS
        ((10, 10), 16777216, (64, 64)),
        ((2000, 3000), 1_000_000, (800, 1216)),
    ],
)
def test_smart_resize_matches_qwen_vl_utils(size, max_pixels, expected):
    width, height = size
    assert ip.smart_resize(height, width, max_pixels)[::-1] == expected
    w, h = expected
    assert (w % 32, h % 32) == (0, 0)
    assert ip.MIN_PIXELS <= w * h <= max_pixels


def test_max_pixels_too_small_for_the_aspect_ratio_is_an_adapter_error():
    # qwen_vl_utils would round the short side down to zero units.
    with pytest.raises(ip.AdapterError, match="max_pixels of 2048 is too small for 2000 x 300"):
        ip.smart_resize(300, 2000, 2048)
    # The smallest accepted max_pixels keeps a unit on both sides at the largest aspect ratio.
    low = ip.parse_params({"max_pixels": 204800})["max_pixels"]
    height, width = ip.smart_resize(100, 20000, low)
    assert (height, width) == (32, 6400)
    assert height * width <= low


def test_extreme_aspect_ratio_is_an_adapter_error():
    with pytest.raises(ip.AdapterError, match="aspect ratio of 5000 x 20 pixels is above 200"):
        ip.smart_resize(20, 5000, ip.DEFAULT_PARAMS["max_pixels"])


def test_undecodable_image_is_an_adapter_error(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(ip.AdapterError, match="cannot decode image x.png"):
        ip.read_image(tmp_path / "x.png")
    Image.new("LA", (4, 3)).save(tmp_path / "la.png")
    assert ip.read_image(tmp_path / "la.png").mode == "RGB"


E1 = {"bbox": [0, 0, 500, 100], "category": "title", "text": "# A"}
E2 = {"bbox": [0, 100, 500, 200], "category": "text", "text": "b"}


@pytest.mark.parametrize(
    ("text", "elements", "rest", "complete"),
    [
        ("", [], "", True),
        ("  \n", [], "", True),
        ("[]", [], "", True),
        (json.dumps([E1, E2]), [E1, E2], "", True),
        (f"```json\n{json.dumps([E1, E2])}\n```", [E1, E2], "", True),  # fenced
        (f"```json\n{json.dumps([E1, E2])}", [E1, E2], "", True),  # unclosed fence
        (f"[{json.dumps(E1)},\n {json.dumps(E2)},]", [E1, E2], "", True),  # comma before ]
        # Truncated: the cut-off element is the rest, the complete ones are kept.
        (f'[{json.dumps(E1)}, {{"bbox": [1, 2', [E1], '{"bbox": [1, 2', False),
        (f"[{json.dumps(E1)}, ", [E1], "", False),  # dangling comma at the end
        (f"[{json.dumps(E1)}", [E1], "", False),  # no closing ]
        (f"[{json.dumps(E1)}\n", [E1], "", False),
        ("[", [], "", False),
        ('[{"bbox": [1, 2], "text": "cut', [], '{"bbox": [1, 2], "text": "cut', False),
        (f"[{json.dumps(E1)}] and more", [E1], "and more", False),
        (f"[{json.dumps(E1)} oops]", [E1], "oops]", False),
        # Adjacent elements without a comma: the array ends after the first one.
        (f"[{json.dumps(E1)} {json.dumps(E2)}]", [E1], f"{json.dumps(E2)}]", False),
        (f"[{json.dumps(E1)},, {json.dumps(E2)}]", [E1], f", {json.dumps(E2)}]", False),
        ("[,]", [], ",]", False),
    ],
)
def test_parse_layout_keeps_complete_elements(text, elements, rest, complete):
    assert ip.parse_layout(text) == (elements, rest, complete)


@pytest.mark.parametrize("text", ["Sorry.", '{"elements": []}', "```json\n{}\n```"])
def test_output_that_is_not_a_json_array_is_an_adapter_error(text):
    with pytest.raises(ip.AdapterError, match="not a layout JSON array"):
        ip.parse_layout(text)


def test_layout_elements_scale_boxes_to_original_pixels():
    elements = [
        {"bbox": [100, 250, 900, 1000], "category": "text", "text": "a"},
        {"bbox": [0, 0, 10], "category": "figure", "text": ""},  # not a box: kept, not scaled
        {"bbox": [0, 0, 1, "x"], "category": "figure", "text": None},
        {"category": "text", "text": "no box"},
    ]
    result = ip.layout_elements(elements, 2480, 3508)
    assert result[0]["bbox_px"] == [248.0, 877.0, 2232.0, 3508.0]
    assert result[0]["bbox"] == [100, 250, 900, 1000]  # the model's own box is kept
    assert not any("bbox_px" in e for e in result[1:])
    assert [e["text"] for e in result] == ["a", "", None, "no box"]


@pytest.mark.parametrize("element", [["x"], "text", {"text": 3}, {"text": ["a"]}])
def test_layout_element_without_text_is_an_adapter_error(element):
    with pytest.raises(ip.AdapterError, match="layout element 1 is not an object with a text"):
        ip.layout_elements([E1, element], 100, 100)


def model_cache(root, corrupt=False, extra=None):
    """A cache with fake model files at the pinned path, and checksums that match them; ``extra``
    adds a file that is not in the checksums."""
    directory = root / "cache" / ip.PINNED_DIR
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
        "OCRBENCH_INFINITYPARSER2_CACHE": str(root / "cache"),
        "OCRBENCH_INFINITYPARSER2_CHECKSUMS": str(checksums),
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
        "prose": (1200, 400),
        "cut": (1280, 400),
        "unclosed": (1344, 400),
        "news": (2000, 3000),
        "empty": (960, 420),
    }
    run, records = run_adapter(tmp_path, sizes)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # only an end token: a genuine empty result
        "boom": ("failure", "engine-error"),
        "prose": ("failure", "engine-error"),  # not layout JSON
        "cut": ("partial", None),  # ended inside an element: the complete ones are scored
        "unclosed": ("partial", None),  # ended without closing the array
        "news": ("truncated", None),  # stopped at max_new_tokens, scored as produced
        "empty": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: CUDA out of memory" in records["boom"]["message"]
    assert records["prose"]["message"] == (
        "the output is not a layout JSON array: 'Sorry, I cannot read this.'"
    )
    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "output.txt",
        "layout.json",
        "page.md",
        "result.json",
    }
    native = (page / "output" / "output.txt").read_text(encoding="utf-8")
    assert native == (FIXTURES / "page.json").read_text(encoding="utf-8")  # unchanged
    layout = json.loads((page / "output" / "layout.json").read_text(encoding="utf-8"))
    assert layout["image"] == {"width": 900, "height": 420}
    assert [e["category"] for e in layout["elements"]] == [
        "header",
        "title",
        "text",
        "figure",
        "text",
        "table",
        "footer",
    ]
    # Boxes in original pixels: [0, 1000] of the width and height of the stored image.
    assert layout["elements"][3]["bbox"] == [120, 280, 880, 540]
    assert layout["elements"][3]["bbox_px"] == pytest.approx([108, 117.6, 792, 226.8])
    # Every element's text in model order, header and footer included; the figure has none.
    markdown = (page / "output" / "page.md").read_text(encoding="utf-8")
    texts = [e["text"] for e in layout["elements"] if e["text"]]
    assert markdown == "\n\n".join(texts)
    # Converted on the host: no Markdown or HTML markup in the text, regions without positions.
    text = (page / "prediction.txt").read_text(encoding="utf-8")
    assert text == "\n\n".join([*FIXTURE_LINES[:4], "\n".join(FIXTURE_LINES[4:6]), "3"])
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"]["from"] == "markdown"
    types = [r["type"] for r in prediction["regions"]]
    assert types == ["text", "heading", "text", "text", "table", "text"]
    assert not any("polygon" in r or "lines" in r for r in prediction["regions"])
    assert "image" not in prediction

    extra = records["page"]["engine_extra"]
    assert extra["stop_reason"] == "end-token"
    assert extra["generated_tokens"] == len(native)  # one fake token per character
    assert extra["elements"] == 7
    assert extra["dropped_characters"] == 0
    assert extra["model_image"] == {"width": 896, "height": 416}  # whole 32-pixel units
    assert extra["image_tokens"] == 28 * 13  # 32 x 32 pixels each
    assert extra["prompt_tokens"] == 28 * 13 + 5
    assert extra["max_new_tokens"] == 32768
    assert extra["gpu_peak_memory_mb"] == 3072
    assert extra["environment"]["transformers"] == "5.19.0+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["verified_official_checkpoint"] is True
    assert (models["repository"], models["revision"]) == (ip.REPOSITORY, ip.REVISION)
    assert models["model_dir"] == ip.PINNED_DIR
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert models["prompt"] == ip.CONVERSATION
    assert models["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["params"] == ip.DEFAULT_PARAMS
    # A small image is only rounded to whole units: MIN_PIXELS is far smaller.
    assert records["blank"]["engine_extra"]["model_image"] == {"width": 288, "height": 192}
    # <|endoftext|>, the language model's own end token, also ends generation.
    assert records["empty"]["engine_extra"]["stop_reason"] == "end-token"
    assert records["empty"]["engine_extra"]["generated_tokens"] == 2
    prose = run.path / "samples" / "prose" / "attempt-1" / "output"
    assert (prose / "output.txt").read_text(encoding="utf-8") == "Sorry, I cannot read this."

    cut = records["cut"]
    assert cut["message"] == "malformed layout JSON after element 1: 14 characters dropped"
    assert cut["engine_extra"]["stop_reason"] == "end-token"
    assert cut["engine_extra"]["dropped_characters"] == len('{"bbox": [0, 5')
    cut_text = run.path / "samples" / "cut" / "attempt-1" / "prediction.txt"
    assert cut_text.read_text(encoding="utf-8") == "kept"
    unclosed = records["unclosed"]
    assert unclosed["message"] == "unclosed layout JSON array after element 1"
    assert unclosed["engine_extra"]["dropped_characters"] == 0
    unclosed_text = run.path / "samples" / "unclosed" / "attempt-1" / "prediction.txt"
    assert unclosed_text.read_text(encoding="utf-8") == "kept"

    news = records["news"]
    assert news["message"] == "generation stopped at max_new_tokens (32768 tokens)"
    assert news["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert news["engine_extra"]["generated_tokens"] == 32768
    assert news["engine_extra"]["model_image"] == {"width": 1984, "height": 3008}
    assert 0 < news["engine_extra"]["dropped_characters"] < 100  # the cut-off element
    news_text = (run.path / "samples" / "news" / "attempt-1" / "prediction.txt").read_text()
    kept = news_text.split("\n\n")
    assert kept == [_newspaper_line(n) for n in range(1, len(kept) + 1)]
    assert news["engine_extra"]["elements"] == len(kept)
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


def test_max_pixels_and_context_limit(tmp_path):
    params = {"device": "cpu", "max_pixels": 1_000_000}
    _, records = run_adapter(tmp_path, {"news": (2000, 3000)}, params, {"FAKE_TORCH_GPUS": "0"})
    extra = records["news"]["engine_extra"]
    # 2000 x 3000 pixels scaled to at most a million: 25 x 38 units of 32 pixels.
    assert extra["model_image"] == {"width": 800, "height": 1216}
    assert extra["image_tokens"] == 25 * 38
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


@pytest.mark.parametrize(
    ("content", "extra", "official"),
    [
        ("other weights", None, False),
        (None, None, True),
        # Extra files may change what transformers loads, for example the decoding defaults.
        (None, "generation_config.json", False),
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
        ({}, "generation_config.json", "not in models.sha256: ['generation_config.json']"),
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
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_INFINITYPARSER2_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{ip.REPOSITORY}/resolve/{ip.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / ip.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / ip.PINNED_DIR).iterdir()) == [
        "model.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def infinityparser2_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / ip.PINNED_DIR / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make infinityparser2-model first")
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
def test_gpu_smoke_benchmark(tmp_path, infinityparser2_image):
    cache = str(infinityparser2_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "infinityparser2"]) == 0
    run = load_run(tmp_path / "runs" / "infinityparser2")
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
    assert extra["elements"] > 0
    assert extra["gpu_peak_memory_mb"] > 0
    layout = run.path / "samples" / "page-1" / "attempt-1" / "output" / "layout.json"
    elements = json.loads(layout.read_text(encoding="utf-8"))["elements"]
    assert all(len(e["bbox_px"]) == 4 for e in elements)

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

    # A small max_new_tokens truncates the page; the complete elements are kept and scored.
    limited = tmp_path / "engine.toml"
    config = Path(engine).read_text(encoding="utf-8")
    limited.write_text(config.replace("max_new_tokens = 32768", "max_new_tokens = 64"), "utf-8")
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["message"] == "generation stopped at max_new_tokens (64 tokens)"
    assert page["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert page["engine_extra"]["generated_tokens"] == 64
    attempt = "samples/page-1/attempt-1/output/output.txt"
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
    assert all("make infinityparser2-model" in r["message"] for r in records)
