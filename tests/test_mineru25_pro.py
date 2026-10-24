"""MinerU2.5-Pro adapter (engines/mineru25-pro): block mapping, parameters, model, protocol.

Offline tests import the adapter and run it as a host process with fake ``mineru_vl_utils``,
``torch`` and ``transformers`` modules (tests/fixtures/mineru25-pro/fake). The fake client returns
committed results: real MinerU2.5-Pro-2605-1.2B output of mineru-vl-utils 2.0.7 for synthetic
pages rendered by tests/synthetic.py (page.json for PAGE, paragraph.json for PARAGRAPH,
columns.json for COLUMNS and newspaper.json for NEWSPAPER), and calls the model's ``generate`` for
the layout and for every block so that truncation can be detected. The opt-in GPU test builds the
engine image and runs the real model through ``ocrbench run``, ``evaluate`` and ``report``.
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
from ocrbench.issues import Severity
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import (
    COLUMNS,
    NEWSPAPER,
    PAGE,
    PARAGRAPH,
    page_xml,
    render,
    render_columns,
    smoke_dataset_with_columns,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "mineru25-pro"
FIXTURES = Path(__file__).parent / "fixtures" / "mineru25-pro"
ADAPTER_PATH = ENGINE_DIR / "mineru25_pro_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
NATIVE = {
    name: json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("page", "paragraph", "columns", "newspaper")
}
SIZES = {"page": (900, 420), "paragraph": (520, 120), "columns": COLUMNS[2]}
SIZES["newspaper"] = NEWSPAPER[2]

_spec = importlib.util.spec_from_file_location("mineru25_pro_engine", ADAPTER_PATH)
mineru = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mineru)


def rendered(name, directory):
    """The PAGE regions of a fixture's synthetic page, rendered as for the fixture."""
    if name in ("page", "paragraph"):
        paragraphs = PAGE if name == "page" else PARAGRAPH
        return render(directory / f"{name}.png", paragraphs, SIZES[name])
    heading, columns, size, font_size = {"columns": COLUMNS, "newspaper": NEWSPAPER}[name]
    return render_columns(directory / f"{name}.png", heading, columns, size, font_size)


def errors(prediction, size):
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=size)
    assert standardized is not None, [i.format() for i in issues]
    return [i for i in issues if i.severity is not Severity.INFO]


def block(kind, bbox=(0.1, 0.1, 0.5, 0.2), content=None, **extra):
    return {"type": kind, "bbox": list(bbox), "angle": 0, "content": content, **extra}


@pytest.mark.parametrize("name", ["page", "paragraph", "columns", "newspaper"])
def test_blocks_become_regions_with_the_recognized_text(tmp_path, name):
    """Recorded real behavior: every rendered paragraph is one block, in reading order (column by
    column on the newspaper page); the model keeps some line breaks inside a paragraph."""
    prediction = mineru.blocks_to_prediction(NATIVE[name], "s1", SIZES[name])
    assert errors(prediction, SIZES[name]) == []
    paragraphs = [" ".join(t for t, _ in region) for region in rendered(name, tmp_path)]
    assert [" ".join(r["text"].split()) for r in prediction["regions"]] == paragraphs
    assert " ".join(prediction["text"].split()) == " ".join(paragraphs)
    assert prediction["reading_order"] == [f"b{i + 1}" for i in range(len(NATIVE[name]))]
    for region, native in zip(prediction["regions"], NATIVE[name], strict=True):
        assert region["native_type"] == native["type"]
        assert "lines" not in region  # no line positions are invented
        assert region["extra"]["angle"] == 0
        assert region["extra"].get("merge_prev") == native.get("merge_prev")


@pytest.mark.parametrize("name", ["page", "columns", "newspaper"])
def test_region_boxes_cover_the_rendered_paragraphs(tmp_path, name):
    """Boxes are relative to the page; mapped to pixels, they cover the rendered text."""
    prediction = mineru.blocks_to_prediction(NATIVE[name], "s1", SIZES[name])
    width, height = SIZES[name]
    for region, lines in zip(prediction["regions"], rendered(name, tmp_path), strict=True):
        xs, ys = [x for x, _ in region["polygon"]], [y for _, y in region["polygon"]]
        assert len(region["polygon"]) == 4
        for _, (x0, y0, x1, y1) in lines:
            assert min(xs) - 0.01 * width <= x0 < x1 <= max(xs) + 0.01 * width
            assert min(ys) - 0.02 * height <= y0 < y1 <= max(ys) + 0.02 * height


def test_box_is_mapped_to_input_image_pixels():
    prediction = mineru.blocks_to_prediction(
        [block("text", (0.1, 0.25, 0.5, 0.75), "x")], "s1", (1000, 400)
    )
    assert prediction["regions"][0]["polygon"] == [
        [100.0, 100.0],
        [500.0, 100.0],
        [500.0, 300.0],
        [100.0, 300.0],
    ]
    assert prediction["image"] == {"width": 1000, "height": 400}


def test_region_types_follow_the_block_types():
    kinds = ["title", "doc_title", "paragraph_title", "text", "list_item", "list", "header"]
    kinds += ["footer", "page_number", "page_footnote", "aside_text", "ref_text", "code"]
    kinds += ["image_caption", "table_footnote", "table", "equation", "image", "chart"]
    kinds += ["image_block", "unknown"]
    prediction = mineru.blocks_to_prediction([block(k) for k in kinds], "s1", (100, 100))
    assert [r["type"] for r in prediction["regions"]] == [
        *["heading"] * 3,
        "text",
        "list-item",
        "other",  # a list container: its items are separate blocks
        "page-header",
        "page-footer",
        "page-number",
        "footnote",
        *["text"] * 3,
        "caption",
        "footnote",
        "table",
        "formula",
        *["figure"] * 3,
        "other",
    ]
    assert [r["native_type"] for r in prediction["regions"]] == kinds
    assert prediction["text"] == ""  # blocks without content stay regions without text
    assert not any("text" in r for r in prediction["regions"])


def test_tables_and_equations_become_plain_text():
    table = (
        "<table><tr><td>Jahr</td><td>Exemplare</td></tr>"
        "<tr><td>1784</td><td>2&amp;026<br>Stück</td></tr></table>"
    )
    blocks = [
        block("table", content=table),
        block("equation", content="\\[\nE = mc^2\n\\]"),
        block("equation", content="$$a+b$$"),
        block("text", content="  Inline \\(x^2\\) stays.  "),
        block("table", content="not a table"),
    ]
    prediction = mineru.blocks_to_prediction(blocks, "s1", (100, 100))
    assert [r["text"] for r in prediction["regions"]] == [
        "Jahr\tExemplare\n1784\t2&026 Stück",
        "E = mc^2",
        "a+b",
        "Inline \\(x^2\\) stays.",
        "not a table",
    ]
    assert errors(prediction, (100, 100)) == []


def test_continued_paragraphs_stay_separate_regions():
    """A text block that continues the previous one (merge_prev, joined in page.md) keeps its own
    box; the page text joins all regions with blank lines."""
    blocks = [
        block("text", (0.1, 0.1, 0.4, 0.9), "The first column ends in", merge_prev=False),
        block("text", (0.6, 0.1, 0.9, 0.9), "the second column.", merge_prev=True),
    ]
    prediction = mineru.blocks_to_prediction(blocks, "s1", (100, 100))
    assert prediction["text"] == "The first column ends in\n\nthe second column."
    assert [r["extra"]["merge_prev"] for r in prediction["regions"]] == [False, True]


def test_a_page_without_blocks_is_an_empty_prediction():
    prediction = mineru.blocks_to_prediction([], "s1", (300, 200))
    assert prediction["text"] == ""
    assert "regions" not in prediction
    assert errors(prediction, (300, 200)) == []


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "cpu"}, "device"),
        ({"device": "gpu:0"}, "device"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 10.0}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"image_analysis": True}, "unknown parameters: image_analysis"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(mineru.AdapterError, match=message):
        mineru.parse_params(params)


def test_params_defaults():
    assert mineru.parse_params({}) == {"device": "cuda:0", "max_new_tokens": None}
    assert mineru.parse_params({"device": "cuda:1", "max_new_tokens": 64}) == {
        "device": "cuda:1",
        "max_new_tokens": 64,
    }


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == f"{mineru.REPOSITORY}@{mineru.REVISION}"
    assert mineru.PINNED_DIR.endswith(mineru.REVISION)
    names = CHECKSUMS[1::2]
    assert {"model.safetensors", "config.json", "generation_config.json"} <= set(names)
    assert {"preprocessor_config.json", "chat_template.jinja", "tokenizer.json"} <= set(names)
    assert "README.md" not in names
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2])


def test_undecodable_image_is_an_adapter_error(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(mineru.AdapterError, match="cannot decode image x.png"):
        mineru.read_image(tmp_path / "x.png")
    Image.new("LA", (4, 3)).save(tmp_path / "la.png")
    assert mineru.read_image(tmp_path / "la.png").mode == "RGB"


def model_cache(root, corrupt=False, extra=None):
    """A cache with fake model files at the pinned path, and checksums that match them; ``extra``
    adds a file that is not in the checksums."""
    directory = root / "cache" / mineru.PINNED_DIR
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
        "OCRBENCH_MINERU25_PRO_CACHE": str(root / "cache"),
        "OCRBENCH_MINERU25_PRO_CHECKSUMS": str(checksums),
    }


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    heights = {1500: NEWSPAPER[2][1], 1000: COLUMNS[2][1], 520: 120, 2000: 3000}
    samples = []
    for sid, width in widths.items():
        size = (width, heights.get(width, 420))
        Image.new("L", size, 255).save(data / f"{sid}.png")
        page_xml(data / f"{sid}.xml", [[("Some reference text", (10, 10, 20, 20))]], size)
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def run_adapter(tmp_path, widths, params=None, env=None):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"])
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {"PYTHONPATH": str(FIXTURES / "fake"), **model_cache(tmp_path), **(env or {})}
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, widths),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in widths}


def test_adapter_follows_the_engine_protocol(tmp_path):
    widths = {"news": 1500, "blank": 300, "boom": 666, "page": 900, "loop": 2000}
    run, records = run_adapter(tmp_path, widths)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "news": ("success", None),
        "blank": ("success", None),  # no blocks: a genuine empty result
        "boom": ("failure", "engine-error"),
        "page": ("success", None),  # the adapter keeps going after failed samples
        "loop": ("truncated", None),  # a block ran to the end of the context
    }
    assert "RuntimeError: CUDA out of memory" in records["boom"]["message"]
    news = run.path / "samples" / "news" / "attempt-1"
    outputs = {o["path"] for o in records["news"]["outputs"]}
    assert outputs == {"mineru.json", "page.md", "prediction.json", "result.json"}
    assert json.loads((news / "output" / "mineru.json").read_text()) == NATIVE["newspaper"]
    expected = mineru.blocks_to_prediction(NATIVE["newspaper"], "news", SIZES["newspaper"])
    assert (news / "prediction.txt").read_text(encoding="utf-8") == expected["text"]
    page_md = (news / "output" / "page.md").read_text(encoding="utf-8")
    assert page_md == "\n\n".join(b["content"] for b in NATIVE["newspaper"])

    extra = records["news"]["engine_extra"]
    assert extra["blocks"] == extra["vl_calls"] == 7
    assert extra["layout"]["image_tokens"] == 37 * 37  # the page scaled to 1036 x 1036
    assert extra["layout"]["generated_tokens"] == 2 * 7
    assert extra["layout"]["stop_reason"] == "end-token"
    assert extra["generated_tokens"] == len(expected["text"].split())  # one per fake word
    assert extra["truncated_blocks"] == 0
    assert extra["max_new_tokens"] is None
    assert extra["image_tokens"] >= extra["max_image_tokens"] > 4
    assert extra["gpu_peak_memory_mb"] == 3072
    assert {"recognition_seconds", "other_seconds"} <= set(extra)
    assert extra["environment"]["mineru_vl_utils"] == "2.0.7+fake"
    assert extra["environment"]["transformers"] == "5.19.0+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["repository"], models["revision"]) == (mineru.REPOSITORY, mineru.REVISION)
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert extra["params"] == mineru.parse_params(CONFIG["params"])

    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""
    assert records["blank"]["engine_extra"]["vl_calls"] == 0
    assert records["blank"]["engine_extra"]["layout"]["stop_reason"] == "end-token"

    loop = records["loop"]
    assert loop["message"] == "1 of 1 blocks stopped without an end token"
    prompt = 5 + loop["engine_extra"]["image_tokens"]
    assert loop["engine_extra"]["generated_tokens"] == 8192 - prompt  # the whole context
    text = (run.path / "samples" / "loop" / "attempt-1" / "prediction.txt").read_text()
    assert text.startswith("Line 1 of a very long newspaper page. Line 2 of")  # kept as produced

    # The truncated page is scored as produced.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert (by_id["loop"]["status"], by_id["loop"]["scored_as"]) == ("truncated", "prediction")
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


def test_max_new_tokens_cuts_off_blocks_and_the_layout(tmp_path):
    params = CONFIG["params"] | {"max_new_tokens": 8}
    run, records = run_adapter(tmp_path, {"page": 900, "news": 1500}, params=params)
    page = records["page"]
    assert (page["status"], page["engine_status"]) == ("truncated", "truncated")
    assert page["message"] == "2 of 3 blocks stopped without an end token"
    assert page["engine_extra"]["max_new_tokens"] == 8
    assert page["engine_extra"]["max_generated_tokens"] == 8
    text = (run.path / "samples" / "page" / "attempt-1" / "prediction.txt").read_text()
    assert text == (
        "The Daily Chronicle\n\nThe quick brown fox jumps over the lazy\n\n"
        "A second paragraph and more text with numbers"
    )
    # The newspaper's layout needs 14 tokens: only the first 4 blocks are detected and read.
    news = records["news"]
    assert news["message"].startswith("layout detection stopped at max_new_tokens; ")
    assert news["engine_extra"]["layout"]["generated_tokens"] == 8
    assert news["engine_extra"]["blocks"] == news["engine_extra"]["vl_calls"] == 4


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "missing", "does not exist; download the weights with make mineru25-pro-model"),
        ({}, "corrupt", "not matching models.sha256: ['model.safetensors']"),
        ({}, "sub/x.json", "not in models.sha256: ['sub/x.json']"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "missing":
        env = {"OCRBENCH_MINERU25_PRO_CACHE": str(tmp_path / "empty")}
    elif env == "corrupt":
        env = model_cache(tmp_path / "corrupt", corrupt=True)
    elif isinstance(env, str):
        env = model_cache(tmp_path / "extra", extra=env)
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "mineru25_pro_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_MINERU25_PRO_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        prefix = f"https://huggingface.co/{mineru.REPOSITORY}/resolve/{mineru.REVISION}/"
        assert url.startswith(prefix)
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / mineru.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache" / mineru.PINNED_DIR).iterdir()) == [
        "model.safetensors"
    ]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def mineru25_pro_cache():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / mineru.PINNED_DIR / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make mineru25-pro-model first")
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
def test_gpu_smoke_benchmark(tmp_path, mineru25_pro_cache):
    cache = str(mineru25_pro_cache)
    manifest = smoke_dataset_with_columns(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "mineru"]) == 0
    run = load_run(tmp_path / "runs" / "mineru")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == dict.fromkeys(
        ("page-1", "para-1", "blank-1", "columns-1", "news-1"), "success"
    )
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"]["mineru_vl_utils"] == "2.0.7"
    assert extra["environment"]["transformers"] == "5.19.0"
    assert extra["environment"]["torch"] == "2.14.1+cu130"
    assert extra["environment"]["gpu"]
    assert (
        extra["models"]["files"]["model.safetensors"]
        == CHECKSUMS[CHECKSUMS.index("model.safetensors") - 1]
    )
    assert extra["layout"]["image_tokens"] == 37 * 37  # the page scaled to 1036 x 1036
    assert extra["vl_calls"] == extra["blocks"] == 3
    assert extra["truncated_blocks"] == 0
    assert extra["gpu_peak_memory_mb"] > 0
    for sid in ("columns-1", "news-1"):
        prediction = json.loads(
            (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
        )
        assert prediction["reading_order"]
        assert all("polygon" in region for region in prediction["regions"])
    blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
    assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure

    # Scoring and re-scoring read only stored outputs: no container runs.
    for _ in range(2):
        assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
    rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    for sid in ("page-1", "para-1", "columns-1", "news-1"):
        assert rates[sid] < 0.1, sid
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # A small max_new_tokens cuts generations off; the sample is truncated and scored as produced.
    limited = tmp_path / "engine.toml"
    config = Path(engine).read_text(encoding="utf-8")
    limited.write_text(config.replace('device = "cuda:0"', 'max_new_tokens = 8\ndevice = "cuda:0"'))
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["engine_extra"]["max_new_tokens"] == 8
    assert main(["evaluate", str(short_run.path)]) == 0
    scores = json.loads((short_run.path / "evaluations" / "1" / "scores.json").read_text())
    short_rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < short_rates["page-1"] <= 1  # scored as produced, not dropped

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make mineru25-pro-model" in r["message"] for r in records)
