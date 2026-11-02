"""MonkeyOCRv2 adapter (engines/monkeyocrv2): parameters, the server command (normal and DFlash),
the pinned checkpoints and source, the conversion of layout blocks to a positioned prediction with
region markup, and the engine protocol.

Offline tests import the adapter and run it as a host process with a fake upstream source tree
(tests/fixtures/monkeyocrv2/fake/source): a fake ``serve.py`` that serves fixed layout and block
outputs over the loopback interface, and stand-ins for the ``core_runner`` functions the adapter
uses, plus a fake ``torch`` module. The outputs include a hand-written layout
(tests/fixtures/monkeyocrv2/layout.txt, not real model output) with every block type, a newspaper
layout cut off at max_tokens, a block whose output repeats up to max_tokens, an output without
layout, an empty layout, an aborted request, failed layout and block requests and a server that
exits. They also evaluate and report that run, whose region markup is converted on the host. The
opt-in GPU test builds the engine image and runs both presets from the weights cache through
``ocrbench run``, ``evaluate`` and ``report``.
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
ENGINE_DIR = ROOT / "engines" / "monkeyocrv2"
FIXTURES = Path(__file__).parent / "fixtures" / "monkeyocrv2"
FAKE = FIXTURES / "fake"
ADAPTER_PATH = ENGINE_DIR / "monkeyocrv2_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
DFLASH = tomllib.loads(
    (ENGINE_DIR / "presets" / "b-parsing-dflash-nodewarp.toml").read_text(encoding="utf-8")
)

_spec = importlib.util.spec_from_file_location("monkeyocrv2_engine", ADAPTER_PATH)
moe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(moe)

# The layout of tests/fixtures/monkeyocrv2/layout.txt in [0, 1000]: (label, box) in model order;
# the last item has no valid box and is dropped by the parser.
LAYOUT = [
    ("Page-header", (100, 120, 900, 160)),
    ("Title", (100, 180, 900, 240)),
    ("Text", (100, 260, 900, 400)),
    ("Section-header", (100, 420, 900, 460)),
    ("Table", (100, 480, 600, 600)),
    ("Formula", (620, 480, 900, 540)),
    ("Picture", (620, 550, 900, 640)),
    ("Footnote", (100, 650, 900, 690)),
    ("Page-footer", (100, 940, 900, 980)),
    ("Text", (900, 920, 100, 700)),  # corners swapped
]
# The region texts after host conversion, in layout order; Picture and Footnote blocks are not
# recognized (upstream has no prompt for them) and have none.
FIXTURE_TEXTS = [
    "THE CHRONICLE",
    "The Daily Chronicle",
    "The quick brown fox jumps over\nthe lazy dog & the river.",
    "News",
    "Year\tCopies\n1923\t2O26",
    "E = mc^2",
    "Footer",
    "A block with its corners swapped.",
]


def test_pinned_checkpoints_and_source_are_the_configured_ones():
    target, draft = moe.CHECKPOINTS["B-Parsing"], moe.CHECKPOINTS["B-Parsing-DFlash"]
    assert CONFIG["model"] == DFLASH["model"] == f"{target['repository']}@{target['revision']}"
    assert (CONFIG["params"]["dflash"], DFLASH["params"]["dflash"]) == (False, True)
    assert CONFIG["image"] == DFLASH["image"]
    assert moe.pinned_dir("B-Parsing").endswith(target["revision"])
    assert moe.pinned_dir("B-Parsing-DFlash").endswith(draft["revision"])
    names = (ENGINE_DIR / "models-B-Parsing.sha256").read_text().split()
    assert {"model.safetensors", "configuration_monkeyocrv2.py"} <= set(names)
    assert not [n for n in names[1::2] if n.endswith(".pth")]  # no dewarping models
    weights = names[names.index("model.safetensors") - 1]
    assert weights.startswith("0267fdc991c9")  # the Hugging Face LFS object ID
    names = (ENGINE_DIR / "models-B-Parsing-DFlash.sha256").read_text().split()
    assert names[1::2] == ["config.json", "model.safetensors"]
    source = (ENGINE_DIR / "source.sha256").read_text().split()[1::2]
    assert {"parsing/serve.py", "parsing/core_runner.py", "LICENSE"} <= set(source)
    assert "parsing/modeling/modeling_monkeyocrv2_vllm.py" in source
    for path in ENGINE_DIR.glob("*.sha256"):
        assert all(len(digest) == 64 for digest in path.read_text().split()[::2])


def test_server_command_is_upstreams_serve_with_its_defaults():
    params = moe.parse_params({})
    command = moe.server_command(params, 1234)
    assert command[:2] == [sys.executable, "/opt/monkeyocrv2/parsing/serve.py"]
    options = " ".join(command[2:])
    target = (
        f"/cache/models/monkeyocrv2/MonkeyOCRv2-B-Parsing/{moe.CHECKPOINTS[moe.TARGET]['revision']}"
    )
    for expected in (
        f"--model-path {target}",
        "--host 127.0.0.1 --port 1234",
        "--served-model-name MonkeyOCRv2 --tensor-parallel-size 1",
        "--gpu-memory-utilization 0.5",
        "--max-model-len 16384 --max-num-seqs 128 --max-num-batched-tokens 16384",
    ):
        assert expected in options
    assert "--draft-model" not in command
    command = moe.server_command(moe.parse_params({"dflash": True}), 1234)
    draft = f"/cache/{moe.pinned_dir(moe.DRAFT)}"
    assert f"--draft-model {draft} --dflash-num-speculative-tokens 16" in " ".join(command)
    assert moe.server_env(params) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    assert moe.server_env(moe.parse_params({"device": "cuda:1"}))["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"dewarp": True}, "unknown parameters: dewarp"),
        ({"device": "cpu"}, "device"),
        ({"dflash": "yes"}, "dflash must be true or false"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 0.9}, r"gpu_memory_utilization must be in \(0, 0.6\]"),
        ({"startup_seconds": -1}, "startup_seconds"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(moe.AdapterError, match=message):
        moe.parse_params(params)


def test_params_defaults_and_pipeline():
    assert moe.DEFAULT_PARAMS == {
        "device": "cuda:0",
        "dflash": False,
        "gpu_memory_utilization": 0.5,
        "startup_seconds": 900,
    }
    assert moe.parse_params(CONFIG["params"]) == moe.DEFAULT_PARAMS
    assert moe.parse_params(DFLASH["params"]) == moe.DEFAULT_PARAMS | {"dflash": True}
    assert (moe.PIPELINE["dewarp"], moe.PIPELINE["retry_repeat"]) == (False, False)
    assert (moe.PIPELINE["max_pixels"], moe.PIPELINE["layout_min_pixels"]) == (1003520, 1003520)
    assert (moe.PIPELINE["layout_max_tokens"], moe.PIPELINE["recognition_max_tokens"]) == (
        4096,
        5000,
    )


def mapped(box, width, height):
    """A [0, 1000] box in pixels as upstream maps it: sorted, rounded and clipped."""
    x1, x2 = sorted(v * width / 1000 for v in (box[0], box[2]))
    y1, y2 = sorted(v * height / 1000 for v in (box[1], box[3]))
    return [round(x1), round(y1), round(x2), round(y2)]


def test_records_become_positioned_regions_with_markup():
    image = Image.new("RGB", (1000, 700), "white")
    records = [
        {"label": "Title", "bbox": [10, 20, 300, 60], "content": "# The **Daily** Chronicle"},
        {"label": "Picture", "bbox": [10, 70, 300, 200]},
        {
            "label": "Table",
            "bbox": [10, 210, 300, 300],
            "content": "<table><tr><td>a</td><td>b</td></tr></table>",
        },
        {"label": "Formula", "bbox": [10, 310, 300, 330], "content": "$$\nE = mc^2\n$$\n"},
        {"label": "Footnote", "bbox": [10, 340, 300, 360]},
        {"label": "Text", "bbox": [10, 370, 300, 400], "content": "  "},
        {"label": "Marginalia", "bbox": [10, 410, 300, 430], "content": "Note"},
    ]
    prediction = moe.records_to_prediction(records, "s", image)
    regions = prediction["regions"]
    assert [r["type"] for r in regions] == [
        "heading", "figure", "table", "formula", "footnote", "text", "other"
    ]  # fmt: skip
    assert [r["native_type"] for r in regions] == [r["label"] for r in records]
    assert regions[0]["polygon"] == [[10, 20], [300, 20], [300, 60], [10, 60]]
    assert all("polygon" in r for r in regions)  # every layout block has a box
    assert ["markup" in r for r in regions] == [True, False, True, True, False, False, True]
    assert prediction["reading_order"] == [r["id"] for r in regions]
    assert prediction["text"] == ""
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(1000, 700))
    assert issues == []
    assert [r.text for r in standardized.regions if r.text] == [
        "The Daily Chronicle", "a\tb", "E = mc^2", "Note"
    ]  # fmt: skip
    assert standardized.conversion == {"from": "markdown", "version": 1}
    empty = moe.records_to_prediction([], "s", image)
    assert "regions" not in empty
    assert empty["text"] == ""


# Engine protocol with the fake upstream source ------------------------------------------------


def page_image(gray, size, boxes=()):
    """A non-uniform page: a checkerboard, the fake server's selector gray level in the top-left
    corner and, for every layout box, the block number + 100 as a marker at its top-left corner,
    from which the fake server tells the crops apart."""
    width, height = size
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for x in range(0, width, 4):
        for y in range(0, height, 4):
            if (x + y) % 8 == 0:
                draw.rectangle((x, y, x + 3, y + 3), fill=255)
    draw.rectangle((0, 0, width // 20, height // 20), fill=gray)
    for number, box in enumerate(boxes, start=1):
        x, y = mapped(box, width, height)[:2]
        draw.rectangle((x, y, x + 4, y + 4), fill=100 + number)
    return image


def model_cache(root, corrupt=False, extra_file=False, without_draft=False):
    """A cache with fake checkpoint files at the pinned paths, and checksums that match them."""
    for checkpoint in moe.CHECKPOINTS:
        if without_draft and checkpoint == moe.DRAFT:
            (root / f"models-{checkpoint}.sha256").write_text("", encoding="utf-8")
            continue
        directory = root / "cache" / moe.pinned_dir(checkpoint)
        directory.mkdir(parents=True)
        lines = []
        for name in ("config.json", "model.safetensors"):
            (directory / name).write_text(f"fake {checkpoint} {name}", encoding="utf-8")
            digest = hashlib.sha256(f"fake {checkpoint} {name}".encode()).hexdigest()
            lines.append(f"{digest}  {name}")
        if corrupt and checkpoint == moe.TARGET:
            (directory / "config.json").write_text("changed", encoding="utf-8")
        if extra_file and checkpoint == moe.TARGET:
            (directory / "preprocessor1.pth").write_text("", encoding="utf-8")
        (root / f"models-{checkpoint}.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_MONKEYOCRV2_CACHE": str(root / "cache"),
        "OCRBENCH_MONKEYOCRV2_CHECKSUMS": str(root),
    }


NEWSPAPER_LINES = 2000
# The fake server chooses the layout by the gray level in the top-left corner (fake/README.md).
GRAY = {
    "page": 200, "news": 150, "rec500": 140, "repeat": 120, "garbage": 100, "abort": 80,
    "empty": 70, "boom": 50, "die": 20,
}  # fmt: skip
NEWS = [(f"Line {n}", (50, n * 9, 950, n * 9 + 8)) for n in range(1, 101)]


def _newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def make_dataset(root, samples):
    data = root / "data"
    data.mkdir()
    entries = []
    for sid, (kind, size) in samples.items():
        boxes = {"page": [box for _, box in LAYOUT], "news": [box for _, box in NEWS]}
        page_image(GRAY[kind], size, boxes.get(kind, ())).save(data / f"{sid}.png")
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


def run_adapter(tmp_path, samples, config=CONFIG, params=None, env=None):
    config = dict(config, params=params if params is not None else config["params"])
    config["timeouts"] = {"sample": 60, "startup": 30}
    del config["resources"]
    env = {
        "PYTHONPATH": str(FAKE),
        "OCRBENCH_MONKEYOCRV2_SOURCE": str(FAKE / "source"),
        "OCRBENCH_MONKEYOCRV2_LOG": str(tmp_path / "vllm-server.log"),
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
        "garbage": ("garbage", (500, 500)),
        "boom": ("boom", (666, 400)),
        "news": ("news", (2000, 3000)),
        "repeat": ("repeat", (400, 300)),
        "abort": ("abort", (300, 200)),
        "rec500": ("rec500", (300, 200)),
        "big": ("page", (2500, 1750)),
        "die": ("die", (500, 500)),
        "after": ("page", (1000, 700)),
    }
    run, records = run_adapter(tmp_path, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "empty": ("success", None),  # an empty layout is an empty page
        "garbage": ("success", None),  # an output without layout items: no blocks
        "boom": ("failure", "engine-error"),
        "news": ("truncated", None),  # the layout stopped at max_tokens, scored as parsed
        "repeat": ("truncated", None),  # a block stopped at max_tokens in a repetition loop
        "abort": ("failure", "engine-error"),
        "rec500": ("failure", "engine-error"),
        "big": ("success", None),  # the adapter keeps going after failed samples
        "die": ("failure", "engine-error"),
        "after": ("failure", "engine-error"),
    }
    assert records["empty"]["message"] == "the layout output has no blocks"
    assert records["garbage"]["message"] == "the layout output has no blocks"
    assert "request failed: HTTP 500" in records["boom"]["message"]
    assert "fake CUDA out of memory" in records["boom"]["message"]
    assert "fake recognition error" in records["rec500"]["message"]
    assert records["abort"]["message"] == "layout request failed (finish_reason 'abort')"
    assert records["after"]["message"].startswith("the vLLM server has exited")

    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "layout.txt", "blocks.json", "monkeyocrv2.json", "page.md", "prediction.json", "result.json"
    }  # fmt: skip
    native = (page / "output" / "layout.txt").read_text(encoding="utf-8")
    assert native == (FIXTURES / "layout.txt").read_text(encoding="utf-8")  # unchanged
    blocks = json.loads((page / "output" / "blocks.json").read_text())["blocks"]
    assert [(b["label"], b["box"]) for b in blocks] == [
        (label, list(map(float, box))) for label, box in LAYOUT
    ]
    assert [b["bbox"] for b in blocks] == [mapped(box, 1000, 700) for _, box in LAYOUT]
    assert blocks[1]["output"] == "The **Daily** Chronicle"  # the raw block output, unchanged
    assert blocks[1]["content"] == "# The **Daily** Chronicle"  # formatted as upstream does
    assert blocks[4]["otsl"].startswith("<fcel>Year")
    assert ["output" in b for b in blocks] == [True] * 6 + [False, False] + [True] * 2
    assert blocks[1]["request"]["finish_reason"] == "stop"
    upstream_record = json.loads((page / "output" / "monkeyocrv2.json").read_text())
    assert upstream_record["image_size"] == [1000, 700]
    assert len(upstream_record["layouts"]) == 10
    markdown = (page / "output" / "page.md").read_text(encoding="utf-8")
    assert markdown.startswith("# The **Daily** Chronicle\n\n")  # no page header or footer
    assert "Footer" not in markdown
    assert "<table>" in markdown
    assert records["page"]["findings"] == []
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": 1}  # converted on the host
    regions = prediction["regions"]
    assert [r["text"] for r in regions if "text" in r] == FIXTURE_TEXTS
    assert all("markup" not in r for r in regions)
    assert regions[9]["polygon"] == [[100, 490], [900, 490], [900, 644], [100, 644]]
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n\n".join(FIXTURE_TEXTS)

    extra = records["page"]["engine_extra"]
    assert extra["layout"]["finish_reason"] == "stop"
    assert extra["layout"]["image"] == [1197, 838]  # enlarged to about 1 MP
    assert extra["layout"]["generated_tokens"] == len(native)
    assert (extra["blocks"], extra["recognized_blocks"]) == (10, 8)
    assert (extra["truncated_blocks"], extra["repeated_blocks"]) == (0, 0)
    assert extra["labels"] == sorted({label for label, _ in LAYOUT})
    assert extra["recognition_generated_tokens"] > 0
    assert min(extra["layout_seconds"], extra["recognition_seconds"]) >= 0
    assert "spec_decode" not in extra
    environment = extra["environment"]
    assert (environment["vllm"], environment["transformers"]) == ("0.26.0+fake", "5.14.1+fake")
    assert (environment["gpu"], environment["gpu_capability"]) == ("Fake GPU", "8.9")
    assert environment["server_command"][1].endswith("parsing/serve.py")
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert list(models["checkpoints"]) == ["B-Parsing"]
    assert set(models["checkpoints"]["B-Parsing"]["files"]) == {"config.json", "model.safetensors"}
    assert models["source"] == {
        "repository": "Yuliang-Liu/MonkeyOCRv2",
        "commit": moe.SOURCE_COMMIT,
    }
    assert models["pipeline"]["dewarp"] is False
    assert extra["params"] == moe.DEFAULT_PARAMS

    big = records["big"]["engine_extra"]
    assert big["layout"]["image"] == [1197, 838]  # reduced to about 1 MP
    assert (run.path / "samples" / "big" / "attempt-1" / "prediction.txt").read_text(
        encoding="utf-8"
    ) == "\n\n".join(FIXTURE_TEXTS)
    empty = run.path / "samples" / "empty" / "attempt-1"
    assert (empty / "prediction.txt").read_text(encoding="utf-8") == ""
    garbage = run.path / "samples" / "garbage" / "attempt-1" / "output" / "layout.txt"
    assert garbage.read_text(encoding="utf-8") == "There is no layout here."  # kept
    abort = run.path / "samples" / "abort" / "attempt-1" / "output" / "layout.txt"
    assert abort.read_text(encoding="utf-8").startswith("[{'bbox'")  # kept on failure
    assert records["abort"]["engine_extra"]["layout"]["finish_reason"] == "abort"

    news = records["news"]
    assert news["message"].startswith("the layout output stopped at max_tokens (4096 tokens); ")
    parsed = news["engine_extra"]["blocks"]
    assert 80 < parsed < 100  # the complete items before the cut
    assert news["message"].endswith(f"{parsed} blocks were parsed from it")
    news_text = (run.path / "samples" / "news" / "attempt-1" / "prediction.txt").read_text()
    assert news_text.startswith(_newspaper_line(1))
    assert _newspaper_line(parsed) in news_text
    assert _newspaper_line(parsed + 1) not in news_text
    repeat = records["repeat"]
    assert repeat["message"] == "1 of 1 block outputs stopped at max_tokens (5000 tokens)"
    assert repeat["engine_extra"]["repeated_blocks"] == 1

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


def test_dflash_preset_records_speculative_decoding(tmp_path):
    run, records = run_adapter(tmp_path, {"page": ("page", (1000, 700))}, config=DFLASH)
    record = records["page"]
    assert record["status"] == "success"
    extra = record["engine_extra"]
    assert "--draft-model" in extra["environment"]["server_command"]
    assert list(extra["models"]["checkpoints"]) == ["B-Parsing", "B-Parsing-DFlash"]
    spec = extra["spec_decode"]
    assert spec["draft_tokens"] == 16 * spec["drafts"] > 0
    assert 0 < spec["accepted_tokens"] < spec["draft_tokens"]
    assert spec["acceptance_rate"] == round(spec["accepted_tokens"] / spec["draft_tokens"], 4)
    text = (run.path / "samples" / "page" / "attempt-1" / "prediction.txt").read_text()
    assert text == "\n\n".join(FIXTURE_TEXTS)


@pytest.mark.parametrize(
    ("config", "params", "env", "message"),
    [
        (CONFIG, {"psm": 3}, {}, "unknown parameters: psm"),
        (CONFIG, {}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        (CONFIG, {}, "corrupt", "not matching models-B-Parsing.sha256: ['config.json']"),
        (CONFIG, {}, "extra", "files not in models-B-Parsing.sha256: ['preprocessor1.pth']"),
        (CONFIG, {}, "missing", "does not exist; download the weights with make monkeyocrv2-model"),
        (
            DFLASH,
            None,
            "nodraft",
            "MonkeyOCRv2-B-Parsing-DFlash/5e87f46698d698e971039168c01c34e044445a9b does not exist",
        ),
        (CONFIG, {}, {"FAKE_VLLM_EXIT": "1"}, "vLLM exited with status 3 during start-up"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, config, params, env, message):
    if env in ("corrupt", "extra", "nodraft"):
        env = model_cache(
            tmp_path / env,
            corrupt=env == "corrupt",
            extra_file=env == "extra",
            without_draft=env == "nodraft",
        )
    elif env == "missing":
        env = {"OCRBENCH_MONKEYOCRV2_CACHE": str(tmp_path / "empty")}
    samples = {"a": ("page", (900, 420)), "b": ("empty", (300, 200))}
    _, records = run_adapter(tmp_path, samples, config, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_the_normal_preset_needs_no_draft(tmp_path):
    env = model_cache(tmp_path / "nodraft", without_draft=True)
    _, records = run_adapter(tmp_path, {"a": ("page", (1000, 700))}, env=env)
    assert records["a"]["status"] == "success"


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "monkeyocrv2_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "HERE", tmp_path)

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main([str(tmp_path / "cache")])
    assert capsys.readouterr().out.count("ok      ") == 4
    fetch_model.main([str(tmp_path / "cache"), "B-Parsing-DFlash"])
    assert capsys.readouterr().out.count("ok      ") == 2

    target = moe.CHECKPOINTS["B-Parsing"]
    urls = []

    def wrong_download(url, path):
        urls.append(url)
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / moe.pinned_dir("B-Parsing") / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(
        SystemExit, match="config.json: SHA-256 differs from models-B-Parsing.sha256"
    ):
        fetch_model.main([str(tmp_path / "cache"), "B-Parsing"])
    assert urls == [
        f"https://huggingface.co/{target['repository']}/resolve/{target['revision']}/config.json"
    ]
    assert sorted(p.name for p in (tmp_path / "cache" / moe.pinned_dir("B-Parsing")).iterdir()) == [
        "model.safetensors"
    ]
    (tmp_path / "source.sha256").write_text(
        f"{hashlib.sha256(b'wrong').hexdigest()}  parsing/serve.py\n", encoding="utf-8"
    )
    fetch_model.main(["--source", str(tmp_path / "src")])
    assert urls[-1] == (
        f"https://raw.githubusercontent.com/Yuliang-Liu/MonkeyOCRv2/{moe.SOURCE_COMMIT}/parsing/serve.py"
    )
    assert (tmp_path / "src" / "parsing" / "serve.py").read_text() == "wrong"
    with pytest.raises(SystemExit):
        fetch_model.main([str(tmp_path / "cache"), "S-Parsing"])


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def monkeyocrv2_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    for checkpoint in moe.CHECKPOINTS:
        weights = cache / moe.pinned_dir(checkpoint) / "model.safetensors"
        if not weights.is_file():
            pytest.fail(f"no weights at {weights}: run make monkeyocrv2-model first")
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
@pytest.mark.parametrize("preset", ["engine.toml", "presets/b-parsing-dflash-nodewarp.toml"])
def test_gpu_smoke_benchmark(tmp_path, monkeyocrv2_image, preset):
    cache = str(monkeyocrv2_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / preset)
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "monkeyocrv2"]) == 0
    run = load_run(tmp_path / "runs" / "monkeyocrv2")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success", "para-1": "success", "blank-1": "success"
    }  # fmt: skip
    extra = records["page-1"]["engine_extra"]
    environment = extra["environment"]
    assert (environment["vllm"], environment["torch"]) == ("0.26.0", "2.11.0+cu130")
    assert environment["gpu"]
    target = (ENGINE_DIR / "models-B-Parsing.sha256").read_text().split()
    weights = target[target.index("model.safetensors") - 1]
    assert extra["models"]["checkpoints"]["B-Parsing"]["files"]["model.safetensors"] == weights
    assert extra["layout"]["finish_reason"] == "stop"
    if "dflash" in preset:
        assert extra["spec_decode"]["accepted_tokens"] > 0
    for sid in ("page-1", "para-1"):
        prediction = json.loads(
            (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
        )
        assert prediction["conversion"] == {"from": "markdown", "version": 1}
        assert prediction["reading_order"]
        assert all("polygon" in region for region in prediction["regions"])

    # Scoring and re-scoring read only stored outputs: no container runs.
    assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
    by_id = {s["sample_id"]: s for s in scores["samples"]}
    assert by_id["page-1"]["chars"]["rate"] < 0.1
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()
