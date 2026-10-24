"""DeepSeek-OCR-2 adapter (engines/deepseek-ocr2): parameters, image views, prompt tokens, layout
blocks, Markdown and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch``,
``torchvision`` and ``transformers`` modules (tests/fixtures/deepseek-ocr2/fake) that "generate"
fixed outputs, including a hand-written page with layout tags, an image block and an HTML table
(tests/fixtures/deepseek-ocr2/output.md) and a newspaper page whose output never ends. They also
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
ENGINE_DIR = ROOT / "engines" / "deepseek-ocr2"
FIXTURES = Path(__file__).parent / "fixtures" / "deepseek-ocr2"
ADAPTER_PATH = ENGINE_DIR / "deepseek_ocr2_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()

_spec = importlib.util.spec_from_file_location("deepseek_ocr2_engine", ADAPTER_PATH)
dsk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dsk)

FIXTURE_LINES = [
    "The Daily Chronicle",
    "The quick brown fox jumps over the lazy dog near the river.",
    "A second paragraph with more text and a formula ( E = mc^2 ).",
    "Year\tCopies",
    "1923\t2026",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{dsk.REPOSITORY}@{dsk.REVISION}"
    assert dsk.PINNED_DIR.endswith(dsk.REVISION)
    names = CHECKSUMS[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == ["model-00001-of-000001.safetensors"]
    # The remote code is executed, so it is pinned with the weights.
    for name in ("modeling_deepseekocr2.py", "modeling_deepseekv2.py", "deepencoderv2.py"):
        assert name in names
    assert "LICENSE.txt" in names
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_prompt_is_the_official_one():
    # The model card's prompt after format_messages strips it; the official vLLM config's PROMPT.
    assert " <image>\n<|grounding|>Convert the document to markdown. ".strip() == dsk.PROMPT


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "gpu:0"}, "device"),
        ({"device": "cpu"}, "device"),  # the remote code needs CUDA
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 100.0}, "max_new_tokens"),
        ({"no_repeat_ngram_size": -1}, "no_repeat_ngram_size"),
        ({"no_repeat_ngram_size": True}, "no_repeat_ngram_size"),
        ({"model_dir": "/cache/models/x"}, "model_dir"),
        ({"model_dir": "models/../../etc"}, "model_dir"),
        ({"model_dir": ""}, "model_dir"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(dsk.AdapterError, match=message):
        dsk.parse_params(params)


def test_params_defaults():
    assert dsk.parse_params({"no_repeat_ngram_size": None}) == dsk.DEFAULT_PARAMS
    assert dsk.DEFAULT_PARAMS["model_dir"] == dsk.PINNED_DIR
    # infer() of the checkpoint's code, as the model card calls it.
    assert dsk.DEFAULT_PARAMS["max_new_tokens"] == 8192
    assert dsk.DEFAULT_PARAMS["no_repeat_ngram_size"] == 20
    assert dsk.parse_params({"no_repeat_ngram_size": 0})["no_repeat_ngram_size"] == 0


@pytest.mark.parametrize(
    ("size", "grid"),
    [
        ((2480, 3508), (2, 3)),  # an A4 page
        ((3508, 2480), (3, 2)),
        ((900, 420), (2, 1)),
        ((1000, 1000), (2, 2)),
        ((1000, 2000), (1, 2)),
        ((800, 6000), (1, 6)),
        ((6000, 800), (6, 1)),
        ((1000, 400), (2, 1)),  # 2.5 is as close to 2 x 1 as to 3 x 1: the smaller grid ...
        ((2500, 1000), (3, 1)),  # ... unless the image has enough pixels for the larger one
    ],
)
def test_tile_grid_is_the_official_one(size, grid):
    assert dsk.tile_grid(*size) == grid


def test_views_and_prompt_tokens(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(dsk.AdapterError, match="cannot decode image x.png"):
        dsk.read_image(tmp_path / "x.png")
    Image.new("LA", (900, 420)).save(tmp_path / "la.png")
    image = dsk.read_image(tmp_path / "la.png")
    assert (image.mode, image.size) == ("RGB", (900, 420))
    global_view, tiles, grid = dsk.views(image)
    assert global_view.size == (1024, 1024)
    assert global_view.getpixel((0, 0)) == (127, 127, 127)  # padded with the mean
    assert global_view.getpixel((512, 512)) == (0, 0, 0)
    assert (grid, [t.size for t in tiles]) == ((2, 1), [(768, 768)] * 2)
    # Up to 768 pixels on both sides: no tiles, only the global view.
    assert dsk.views(Image.new("RGB", (768, 500)))[1:] == ([], (1, 1))

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return [len(text)]

    ids, mask = dsk.prompt_ids(Tokenizer(), (1, 1))
    assert ids == [dsk.BOS, 0, *[dsk.IMAGE_TOKEN] * 257, len(dsk.PROMPT) - len("<image>")]
    assert mask == [False, False, *[True] * 257, False]
    ids, mask = dsk.prompt_ids(Tokenizer(), (2, 3))
    assert sum(mask) == ids.count(dsk.IMAGE_TOKEN) == 16 * 16 + 1 + 24 * 36
    assert len(ids) == sum(mask) + 3


def test_layout_blocks_and_markdown():
    text = (FIXTURES / "output.md").read_text(encoding="utf-8").strip()
    blocks = dsk.layout_blocks(text, 999, 1998)
    assert [b["label"] for b in blocks] == ["title", "text", "image", "text", "table"]
    assert blocks[0]["det"] == "[[96, 41, 904, 88]]"
    assert blocks[0]["boxes"] == [[96, 41, 904, 88]]
    assert blocks[0]["boxes_px"] == [[96.0, 82.0, 904.0, 176.0]]
    assert blocks[0]["text"] == "# The Daily Chronicle"
    assert blocks[2]["text"] == ""  # an image block has no text
    start, end = blocks[4]["span"]
    assert text[start:end] == "<|ref|>table<|/ref|><|det|>[[96, 580, 600, 700]]<|/det|>"
    assert blocks[4]["text"].startswith("<table>")

    page = dsk.markdown(text)
    assert "<|" not in page
    assert page.startswith("\n# The Daily Chronicle\n\n")
    assert "![](images/0.jpg)\n" in page

    # Boxes are read as literals only; anything else is kept as written, without a box.
    weird = (
        "<|ref|>text<|/ref|><|det|>[[1, 2, 3]]<|/det|>\nA\n"
        "<|ref|>text<|/ref|><|det|>__import__('os')<|/det|>\nB\n"
        "<|ref|>text<|/ref|><|det|>[[0, 0, 999, 999], [10, 10, 20, 20]]<|/det|>\nC := \\coloneqq"
    )
    blocks = dsk.layout_blocks(weird, 100, 100)
    assert [b["boxes"] for b in blocks[:2]] == [None, None]
    assert all("boxes_px" not in b for b in blocks[:2])
    assert blocks[1]["det"] == "__import__('os')"
    small = [1000 / 999, 1000 / 999, 2000 / 999, 2000 / 999]
    assert blocks[2]["boxes_px"] == [[0.0, 0.0, 100.0, 100.0], small]
    assert dsk.markdown(weird) == "\nA\n\nB\n\nC := :="
    assert dsk.markdown("No tags \\coloneqq") == "No tags \\coloneqq"  # as the official code


def model_cache(root, corrupt=False):
    """A cache with fake model files at the pinned path, and checksums that match them."""
    directory = root / "cache" / dsk.PINNED_DIR
    directory.mkdir(parents=True)
    lines = []
    for name in ("config.json", "model-00001-of-000001.safetensors"):
        (directory / name).write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(f'fake {name}'.encode()).hexdigest()}  {name}")
    if corrupt:
        (directory / "model-00001-of-000001.safetensors").write_text("changed", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_DEEPSEEK_OCR2_CACHE": str(root / "cache"),
        "OCRBENCH_DEEPSEEK_OCR2_CHECKSUMS": str(checksums),
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
            lines = [f"Line {n} of a long page." for n in range(1, NEWSPAPER_LINES + 1)]
            regions = [[(text, (10, n, 20, n + 1)) for n, text in enumerate(lines)]]
        elif kind == "page":
            lines = [line.replace("\t", " ") for line in FIXTURE_LINES]
            regions = [[(text, (10, 10, 20, 20))] for text in lines]
        else:
            regions = []
        page_xml(data / f"{sid}.xml", regions, size)
        entries.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, entries)


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
        "blank": ("success", None),  # an empty output: a genuine empty result
        "boom": ("failure", "engine-error"),
        "plain": ("success", None),  # without layout tags, the whole output is the text
        "news": ("truncated", None),  # stopped at the context, scored as produced
        "page2": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: CUDA out of memory" in records["boom"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "output.md",
        "layout.json",
        "page.md",
        "result.json",
    }
    native = (page / "output" / "output.md").read_text(encoding="utf-8")
    assert native == (FIXTURES / "output.md").read_text(encoding="utf-8")  # unchanged output
    markdown = (page / "output" / "page.md").read_text(encoding="utf-8")
    assert markdown == dsk.markdown(native.strip())
    layout = json.loads((page / "output" / "layout.json").read_text(encoding="utf-8"))
    assert layout["image"] == {"width": 900, "height": 420}
    assert layout["blocks"][0]["boxes_px"] == [[96 * 900 / 999, 41 * 420 / 999,
                                                904 * 900 / 999, 88 * 420 / 999]]  # fmt: skip
    # Converted on the host: no Markdown, HTML or layout markup in the text, no positions.
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
    assert (extra["blocks"], extra["image_blocks"], extra["invalid_boxes"]) == (5, 1, 0)
    assert extra["tiles"] == {"columns": 2, "rows": 1}
    assert extra["image_tokens"] == 16 * 16 + 1 + 12 * 2 * 12
    assert extra["prompt_tokens"] == extra["image_tokens"] + 1 + len(dsk.PROMPT) - len("<image>")
    assert extra["max_new_tokens"] == 8192 - extra["prompt_tokens"]
    assert extra["gpu_peak_memory_mb"] == 7168
    assert extra["environment"]["transformers"] == "4.46.3+fake"
    assert extra["environment"]["tokenizers"] == "0.20.3+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["verified_official_checkpoint"] is True
    assert (models["repository"], models["revision"]) == (dsk.REPOSITORY, dsk.REVISION)
    assert models["model_dir"] == dsk.PINNED_DIR
    assert set(models["files"]) == {"config.json", "model-00001-of-000001.safetensors"}
    assert models["prompt"] == dsk.PROMPT
    assert extra["params"] == dsk.DEFAULT_PARAMS
    # A small image has the global view only.
    blank = records["blank"]["engine_extra"]
    assert (blank["tiles"], blank["image_tokens"], blank["blocks"]) == (None, 257, 0)
    blank_dir = run.path / "samples" / "blank" / "attempt-1"
    assert (blank_dir / "output" / "page.md").read_text(encoding="utf-8") == ""
    assert (blank_dir / "prediction.txt").read_text(encoding="utf-8") == ""
    plain_text = run.path / "samples" / "plain" / "attempt-1" / "prediction.txt"
    assert plain_text.read_text(encoding="utf-8") == "Plain text without layout tags."

    news = records["news"]
    budget = 8192 - news["engine_extra"]["prompt_tokens"]
    assert news["message"] == f"generation stopped at the model's context ({budget} tokens)"
    assert news["engine_extra"]["stop_reason"] == "context"
    assert news["engine_extra"]["generated_tokens"] == budget
    assert news["engine_extra"]["tiles"] == {"columns": 2, "rows": 3}

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


def test_max_new_tokens_and_ngram_settings(tmp_path):
    params = {"device": "cuda:0", "max_new_tokens": 64, "no_repeat_ngram_size": 0}
    _, records = run_adapter(tmp_path, {"news": ("news", (2000, 3000))}, params)
    news = records["news"]
    assert news["status"] == "truncated"
    assert news["message"] == "generation stopped at max_new_tokens (64 tokens)"
    extra = news["engine_extra"]
    assert (extra["stop_reason"], extra["generated_tokens"]) == ("max_new_tokens", 64)
    assert extra["params"]["no_repeat_ngram_size"] == 0  # the fake model receives it


@pytest.mark.parametrize(("content", "official"), [("other weights", False), (None, True)])
def test_a_custom_model_dir_claims_the_official_checkpoint_only_when_verified(
    tmp_path, content, official
):
    custom = tmp_path / "cache" / "models" / "custom"
    custom.mkdir(parents=True)
    for name in ("config.json", "model-00001-of-000001.safetensors"):  # None: the official files
        (custom / name).write_text(content or f"fake {name}", encoding="utf-8")
    params = CONFIG["params"] | {"model_dir": "models/custom"}
    _, records = run_adapter(tmp_path, {"page": ("page", (900, 420))}, params)
    assert records["page"]["status"] == "success"
    models = records["page"]["engine_extra"]["models"]
    assert models["verified_official_checkpoint"] is official
    assert ("repository" in models, "revision" in models) == (official, official)
    assert models["model_dir"] == "models/custom"
    digest = hashlib.sha256((content or "fake model-00001-of-000001.safetensors").encode())
    assert models["files"]["model-00001-of-000001.safetensors"] == digest.hexdigest()


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({"model_dir": "models/other"}, {}, "does not exist; download the weights"),
        ({}, "corrupt", "not matching models.sha256: ['model-00001-of-000001.safetensors']"),
        ({}, "extra", "model files not in models.sha256: ['extra.py']"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "corrupt":
        env = model_cache(tmp_path / "corrupt", corrupt=True)
    elif env == "extra":  # code that the remote code could import
        env = model_cache(tmp_path / "extra")
        (tmp_path / "extra" / "cache" / dsk.PINNED_DIR / "extra.py").write_text("", "utf-8")
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
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_DEEPSEEK_OCR2_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{dsk.REPOSITORY}/resolve/{dsk.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / dsk.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / dsk.PINNED_DIR).iterdir()) == [
        "model-00001-of-000001.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def deepseek_ocr2_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / dsk.PINNED_DIR / "model-00001-of-000001.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make deepseek-ocr2-model first")
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
def test_gpu_smoke_benchmark(tmp_path, deepseek_ocr2_image):
    cache = str(deepseek_ocr2_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "deepseek-ocr2"]) == 0
    run = load_run(tmp_path / "runs" / "deepseek-ocr2")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    # The blank page is a success; the model may still write text on it.
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"]["transformers"] == "4.46.3"
    assert extra["environment"]["tokenizers"] == "0.20.3"
    assert extra["environment"]["torch"] == "2.14.1+cu130"
    assert extra["environment"]["gpu"]
    name = "model-00001-of-000001.safetensors"
    assert extra["models"]["files"][name] == CHECKSUMS[CHECKSUMS.index(name) - 1]
    assert extra["models"]["verified_official_checkpoint"] is True
    assert extra["stop_reason"] == "end-token"
    assert extra["blocks"] > 0
    assert extra["invalid_boxes"] == 0
    assert extra["gpu_peak_memory_mb"] > 0
    layout = run.path / "samples" / "page-1" / "attempt-1" / "output" / "layout.json"
    blocks = json.loads(layout.read_text(encoding="utf-8"))["blocks"]
    assert all(b["boxes_px"] for b in blocks)

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
    limited.write_text(config.replace("max_new_tokens = 8192", "max_new_tokens = 32"), "utf-8")
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["message"] == "generation stopped at max_new_tokens (32 tokens)"
    assert page["engine_extra"]["stop_reason"] == "max_new_tokens"
    assert page["engine_extra"]["generated_tokens"] == 32
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
    assert all("make deepseek-ocr2-model" in r["message"] for r in records)
