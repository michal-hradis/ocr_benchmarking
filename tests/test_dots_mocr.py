"""dots.mocr adapter (engines/dots-mocr): parameters, the prompt, the layout conversion to a
positioned prediction and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch`` and
``transformers`` modules (tests/fixtures/dots-mocr/fake) that "generate" fixed outputs, including
a hand-written layout list (tests/fixtures/dots-mocr/output.json), output that is not JSON, a list
cut off by an end token and a newspaper page whose list never ends. They also evaluate and report
that run. The opt-in GPU test builds the engine image and runs the real model from the weights
cache through ``ocrbench run``, ``evaluate`` and ``report``.
"""

import hashlib
import importlib.metadata
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
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import page_xml, smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "dots-mocr"
FIXTURES = Path(__file__).parent / "fixtures" / "dots-mocr"
ADAPTER_PATH = ENGINE_DIR / "dots_mocr_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
# SHA-256 of dict_promptmode_to_prompt["prompt_layout_all_en"] in dots_mocr/utils/prompts.py of
# github.com/rednote-hilab/dots.mocr at commit 23f3e5612fb8066d4034d5ecfc8f33a9243533eb.
UPSTREAM_PROMPT_SHA256 = "16ff71ac5d218f35e5b3db41240b6e70741498bc099db3fa922ce1ff972e3b2f"

_spec = importlib.util.spec_from_file_location("dots_mocr_engine", ADAPTER_PATH)
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)

# The region texts of tests/fixtures/dots-mocr/output.json in plain text, in output order.
FIXTURE_TEXTS = [
    "Page 3",
    "The Daily Chronicle",
    "The quick brown fox jumps over the lazy dog near the river.",
    "A list item about the harbour.",
    "E = mc^2",
    "Year\tCopies\n1923\t2026",
    "A caption with a broken box.",
    "3",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{dm.REPOSITORY}@{dm.REVISION}"
    assert dm.PINNED_DIR.endswith(dm.REVISION)
    assert "." not in dm.PINNED_DIR  # the remote code needs a directory name without periods
    names = CHECKSUMS[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    code = {"configuration_dots.py", "modeling_dots_ocr.py", "modeling_dots_vision.py"}
    assert code | {"chat_template.json", "preprocessor_config.json", "tokenizer.json"} <= set(names)
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_prompt_is_the_official_one():
    assert hashlib.sha256(dm.PROMPT.encode()).hexdigest() == UPSTREAM_PROMPT_SHA256
    assert dm.CONVERSATION == [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": dm.PROMPT}]}
    ]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "cpu"}, "device"),
        ({"device": "gpu:0"}, "device"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 100.0}, "max_new_tokens"),
        ({"max_new_tokens": 131072}, "below the context"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(dm.AdapterError, match=message):
        dm.parse_params(params)


def test_params_defaults():
    assert dm.parse_params({"max_new_tokens": None}) == dm.DEFAULT_PARAMS
    assert dm.DEFAULT_PARAMS == {"device": "cuda:0", "max_new_tokens": 24000}
    assert CONFIG["params"] == dm.DEFAULT_PARAMS


def test_flash_attn_import_is_provided_without_the_package(monkeypatch):
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    dm.install_flash_attn()
    from flash_attn import flash_attn_varlen_func

    assert flash_attn_varlen_func is dm.flash_attn_varlen_func
    assert importlib.util.find_spec("flash_attn") is not None
    with pytest.raises(importlib.metadata.PackageNotFoundError):  # not "available" to transformers
        importlib.metadata.version("flash_attn")


@pytest.mark.parametrize(
    ("text", "elements", "complete"),
    [
        ("[]", [], True),
        ('[{"a": 1}, {"b": 2}]', [{"a": 1}, {"b": 2}], True),
        ('{"a": 1}', [], False),  # not a list
        ('[{"a": 1}, {"b": 2', [{"a": 1}], False),  # cut off
        ('[{"a": 1},\n {"b": [1, 2]}, ', [{"a": 1}, {"b": [1, 2]}], False),
        ('  [{"a": "x]"}]  junk', [{"a": "x]"}], False),
        ("plain text", [], False),
        ("", [], False),
    ],
)
def test_layout_parsing_keeps_complete_elements(text, elements, complete):
    assert dm.parse_layout(text) == (elements, complete)


@pytest.mark.parametrize(
    ("category", "content", "text"),
    [
        ("Title", "# The Title", "The Title"),
        ("Section-header", "### **Part 2**", "Part 2"),
        ("Section-header", "#hashtag", "#hashtag"),  # not a heading mark
        ("Text", "# not a heading in Text", "# not a heading in Text"),
        ("Text", " a **bold** and *italic* word ", "a bold and *italic* word"),
        ("List-item", "- item", "item"),
        ("List-item", "• item", "item"),
        ("List-item", "2. item", "2. item"),  # the number is kept, as the host converter does
        ("Formula", "$$x^2$$", "x^2"),
        ("Formula", "\\[ x^2 \\]", "x^2"),
        ("Formula", "$x$", "x"),
        ("Formula", "x^2", "x^2"),
        ("Table", "<table><tr><td>a</td><td>b<br>c</td></tr></table>", "a\tb c"),
        ("Table", "plain", "plain"),
        ("Picture", "a drawing of a ship", ""),
    ],
)
def test_element_text_is_plain(category, content, text):
    assert dm.element_text(category, content) == text


def test_layout_to_prediction_maps_boxes_to_the_input_image():
    elements = [
        {"bbox": [0, 0, 1008, 700], "category": "Text", "text": "all"},
        {"bbox": [504, 350, 756, 525.5], "category": "Unknown", "text": "x"},
        {"bbox": [10, 10, 10, 20], "category": "Text", "text": "zero width"},
        {"bbox": [1, 2, 3], "category": "Picture"},
        {"bbox": [1, 2, 3, "4"], "text": "no category"},
        "not an element",
        {"bbox": [0, 0, 5, 5], "category": "Text", "text": "all"},
        {"bbox": [0, 0, 5, 5], "category": "Text", "text": "all"},  # repeated
    ]
    prediction, counts = dm.layout_to_prediction(elements, "s", (1000, 700), (1008, 700))
    regions = prediction["regions"]
    assert counts == {"skipped_elements": 1, "invalid_boxes": 3, "repeated_elements": 1}
    assert (
        [r["id"] for r in regions] == prediction["reading_order"] == [f"e{i}" for i in range(1, 8)]
    )
    assert regions[0]["polygon"] == [[0, 0], [1000, 0], [1000, 700], [0, 700]]
    s = 1000 / 1008
    assert regions[1]["polygon"] == [
        [504 * s, 350],
        [756 * s, 350],
        [756 * s, 525.5],
        [504 * s, 525.5],
    ]
    assert (regions[1]["type"], regions[1]["native_type"]) == ("other", "Unknown")
    assert "polygon" not in regions[2]  # no box is invented
    assert regions[2]["extra"] == {"bbox": [10, 10, 10, 20]}
    assert regions[3] == {
        "id": "e4",
        "type": "figure",
        "native_type": "Picture",
        "extra": {"bbox": [1, 2, 3]},
    }
    assert regions[4]["type"] == "other"
    assert "native_type" not in regions[4]
    assert prediction["text"] == "all\n\nx\n\nzero width\n\nno category\n\nall\n\nall"
    assert prediction["image"] == {"width": 1000, "height": 700}
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(1000, 700))
    assert standardized is not None
    assert issues == []


def model_cache(root, corrupt=False, extra_file=False):
    """A cache with fake model files at the pinned path, and checksums that match them."""
    directory = root / "cache" / dm.PINNED_DIR
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model-00001-of-00002.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "model-00001-of-00002.safetensors").write_text("changed", encoding="utf-8")
    if extra_file:
        (directory / "modeling_extra.py").write_text("print('hi')", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_DOTS_MOCR_CACHE": str(root / "cache"),
        "OCRBENCH_DOTS_MOCR_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000
# The fake model chooses its output by the gray level of the image (fake/README.md).
GRAY = {"page": 255, "blank": 250, "boom": 200, "nojson": 150, "news": 100, "cut": 50}


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
        "page": ("page", (1000, 700)),
        "blank": ("blank", (300, 200)),
        "boom": ("boom", (666, 400)),
        "nojson": ("nojson", (500, 500)),
        "cut": ("cut", (500, 500)),
        "news": ("news", (2000, 3000)),
        "big": ("big", (5000, 7000)),
        "page2": ("page", (1000, 700)),
    }
    run, records = run_adapter(tmp_path, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # an empty list: a genuine empty result
        "boom": ("failure", "engine-error"),
        "nojson": ("failure", "engine-error"),
        "cut": ("partial", None),
        "news": ("truncated", None),  # stopped at max_new_tokens, scored as produced
        "big": ("success", None),
        "page2": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: CUDA out of memory" in records["boom"]["message"]
    assert records["nojson"]["message"] == (
        "the output is not a JSON list of layout elements (output.json)"
    )
    nojson = run.path / "samples" / "nojson" / "attempt-1" / "output"
    assert (nojson / "output.json").read_text(encoding="utf-8") == "This page has no layout."

    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "output.json",
        "prediction.json",
        "result.json",
    }
    native = (page / "output" / "output.json").read_text(encoding="utf-8")
    assert native == (FIXTURES / "output.json").read_text(encoding="utf-8").strip()  # unchanged
    assert records["page"]["findings"] == []
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert "conversion" not in prediction  # written by the engine, not converted on the host
    assert prediction["image"] == {"width": 1000, "height": 700}
    regions = prediction["regions"]
    assert [r.get("native_type") for r in regions] == [e["category"] for e in json.loads(native)]
    assert [r["type"] for r in regions] == [
        "page-header",
        "heading",
        "text",
        "list-item",
        "formula",
        "figure",
        "table",
        "caption",
        "page-footer",
    ]
    assert [r["text"] for r in regions if "text" in r] == FIXTURE_TEXTS
    assert prediction["reading_order"] == [r["id"] for r in regions]
    # The model saw 1008 x 700 pixels (whole 28-pixel units); boxes are mapped back to 1000 x 700.
    s = 1000 / 1008
    assert regions[1]["polygon"] == [[56 * s, 84], [952 * s, 84], [952 * s, 140], [56 * s, 140]]
    assert "polygon" not in regions[7]  # the caption's box has x1 > x2: no box is invented
    text = (page / "prediction.txt").read_text(encoding="utf-8")
    assert text == "\n\n".join(FIXTURE_TEXTS)

    extra = records["page"]["engine_extra"]
    assert extra["stop_reason"] == "end-token"
    assert extra["generated_tokens"] == len(native)  # one fake token per character
    assert (extra["json_complete"], extra["elements"]) == (True, 9)
    assert extra["invalid_boxes"] == 1
    assert extra["skipped_elements"] == extra["repeated_elements"] == 0
    assert extra["source_image"] == {"width": 1000, "height": 700}
    assert extra["model_image"] == {"width": 1008, "height": 700}
    assert extra["image_tokens"] == (1008 // 28) * (700 // 28)
    assert extra["prompt_tokens"] == extra["image_tokens"] + len(dm.PROMPT) + 5
    assert extra["max_new_tokens"] == 24000
    assert extra["gpu_peak_memory_mb"] == 3072
    assert extra["environment"]["transformers"] == "4.57.6+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["repository"], models["revision"]) == (dm.REPOSITORY, dm.REVISION)
    assert set(models["files"]) == {"config.json", "model-00001-of-00002.safetensors"}
    assert models["prompt"] == dm.CONVERSATION
    assert models["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["params"] == dm.DEFAULT_PARAMS

    # A small image is enlarged to at least 3,136 pixels only if needed; a large one is reduced
    # to at most 11,289,600.
    assert records["blank"]["engine_extra"]["model_image"] == {"width": 308, "height": 196}
    big = records["big"]["engine_extra"]
    assert big["model_image"] == {"width": 2828, "height": 3948}
    assert big["model_image"]["width"] * big["model_image"]["height"] <= 11289600
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert json.loads((blank / "prediction.json").read_text())["text"] == ""
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""

    cut = records["cut"]
    assert cut["message"] == "the output is not a complete JSON list; 1 complete elements are used"
    assert cut["engine_extra"]["json_complete"] is False
    cut_dir = run.path / "samples" / "cut" / "attempt-1"
    assert (cut_dir / "prediction.txt").read_text(encoding="utf-8") == "Complete element."

    news = records["news"]
    assert news["message"].startswith("generation stopped at max_new_tokens (24000 of 24000 ")
    assert news["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert news["engine_extra"]["generated_tokens"] == 24000
    assert news["engine_extra"]["json_complete"] is False
    elements = news["engine_extra"]["elements"]
    assert news["message"].endswith(f"tokens); {elements} complete elements are used")
    assert news["engine_extra"]["repeated_elements"] == elements - 101  # the loop is reported
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
    assert news["scored_as"] == "prediction"
    assert news["flexible_units"]["reference"] == NEWSPAPER_LINES
    assert 0 < news["chars"]["rate"] < 1  # only the beginning of the page was transcribed
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


def test_context_limit(tmp_path):
    _, records = run_adapter(tmp_path, {"news": ("news", (2000, 3000))}, {"max_new_tokens": 131071})
    extra = records["news"]["engine_extra"]
    assert extra["max_new_tokens"] == 131072 - extra["prompt_tokens"]
    assert extra["stop_reason"] == "context"
    assert records["news"]["status"] == "truncated"
    assert "the model's context" in records["news"]["message"]


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "corrupt", "not matching models.sha256: ['model-00001-of-00002.safetensors']"),
        ({}, "extra", "model files not in models.sha256: ['modeling_extra.py']"),
        ({}, "missing", "does not exist; download the weights with make dots-mocr-model"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env in ("corrupt", "extra"):
        env = model_cache(tmp_path / env, corrupt=env == "corrupt", extra_file=env == "extra")
    elif env == "missing":
        env = {"OCRBENCH_DOTS_MOCR_CACHE": str(tmp_path / "empty")}
    samples = {"a": ("page", (900, 420)), "b": ("blank", (300, 200))}
    _, records = run_adapter(tmp_path, samples, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "dots_mocr_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_DOTS_MOCR_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{dm.REPOSITORY}/resolve/{dm.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / dm.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / dm.PINNED_DIR).iterdir()) == [
        "model-00001-of-00002.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def dots_mocr_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / dm.PINNED_DIR / "model-00002-of-00002.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make dots-mocr-model first")
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
def test_gpu_smoke_benchmark(tmp_path, dots_mocr_image):
    cache = str(dots_mocr_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "dots-mocr"]) == 0
    run = load_run(tmp_path / "runs" / "dots-mocr")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"]["transformers"] == "4.57.6"
    assert extra["environment"]["torch"] == "2.14.1+cu130"
    assert extra["environment"]["gpu"]
    name = "model-00001-of-00002.safetensors"
    assert extra["models"]["files"][name] == CHECKSUMS[CHECKSUMS.index(name) - 1]
    assert extra["models"]["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["stop_reason"] == "end-token"
    assert extra["json_complete"] is True
    assert extra["gpu_peak_memory_mb"] > 0
    for sid in ("page-1", "para-1"):
        prediction = json.loads(
            (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
        )
        assert prediction["reading_order"]
        assert all("polygon" in region for region in prediction["regions"])

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
    limited.write_text(config.replace("max_new_tokens = 24000", "max_new_tokens = 40"), "utf-8")
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["message"].startswith("generation stopped at max_new_tokens (40 of 40 tokens); ")
    assert page["engine_extra"]["json_complete"] is False
    assert main(["evaluate", str(short_run.path)]) == 0
    scores = json.loads((short_run.path / "evaluations" / "1" / "scores.json").read_text())
    short_rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < short_rates["page-1"] <= 1  # scored as produced

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make dots-mocr-model" in r["message"] for r in records)
