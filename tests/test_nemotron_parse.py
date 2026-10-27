"""Nemotron 3.5 Parse adapter (engines/nemotron-parse): parameters, the vLLM server command, the
block conversion to a positioned prediction with region markup, and the engine protocol.

Offline tests import the adapter and run it as a host process with a fake ``vllm`` executable that
serves fixed outputs over the loopback interface, and fake ``nemotron_parse`` and ``torch`` modules
(tests/fixtures/nemotron-parse/fake). The outputs include hand-written layout blocks
(tests/fixtures/nemotron-parse/output.txt), output without blocks, an empty output, a newspaper
page cut off at ``max_tokens`` in a repetition loop, a table crop cut off at ``max_tokens``, a
failed request and a server that exits. They also evaluate and report that run, whose region
markup is converted on the host. The opt-in GPU test builds the engine image and runs the real
model from the weights cache through ``ocrbench run``, ``evaluate`` and ``report``.
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
ENGINE_DIR = ROOT / "engines" / "nemotron-parse"
FIXTURES = Path(__file__).parent / "fixtures" / "nemotron-parse"
FAKE = FIXTURES / "fake"
ADAPTER_PATH = ENGINE_DIR / "nemotron_parse_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
DOUBLE = tomllib.loads((ENGINE_DIR / "presets" / "double.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text().split()
SOURCE = (ENGINE_DIR / "source.sha256").read_text().split()
# SHA-256 of PROMPT in nemotron_parse/inference.py of nemotron-parse 3.5.0.post7, the model card's
# prompt.
UPSTREAM_PROMPT_SHA256 = hashlib.sha256(
    b"</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
).hexdigest()

_spec = importlib.util.spec_from_file_location("nemotron_parse_engine", ADAPTER_PATH)
npe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(npe)

# The region texts of tests/fixtures/nemotron-parse/output.txt after host conversion, in output
# order; the Picture block has none. The fake does not render tables, so the table is the
# fixture's HTML.
FIXTURE_TEXTS = [
    "Page 3",
    "The Daily Chronicle",
    "The quick brown fox jumps over\nthe lazy dog & the river.",
    "First item",
    "Second item",
    "E = mc^2",
    "Year\tCopies\n1923\t2O26",
    "3",
]


def test_pinned_model_is_the_configured_one():
    assert CONFIG["model"] == DOUBLE["model"] == f"{npe.REPOSITORY}@{npe.REVISION}"
    assert CONFIG["image"] == DOUBLE["image"]
    assert npe.PINNED_DIR.endswith(npe.REVISION)
    names = CHECKSUMS[1::2]
    assert [n for n in names if n.endswith(".safetensors")] == ["model.safetensors"]
    assert {"config.json", "chat_template.jinja", "tokenizer.json", "LICENSE"} <= set(names)
    assert all(len(digest) == 64 for digest in CHECKSUMS[::2] + SOURCE[::2])
    # The weights' SHA-256 is the Hugging Face LFS object ID at the pinned revision.
    assert CHECKSUMS[CHECKSUMS.index("model.safetensors") - 1].startswith("81dd7609c589")
    # The package source: its build file, the plugin and the parser.
    sources = SOURCE[1::2]
    assert {"pyproject.toml", "vllm_tied_patch/plugin.py", "nemotron_parse/inference.py"} <= set(
        sources
    )


def test_server_command_has_the_serving_settings():
    params = npe.parse_params({})
    command = npe.server_command(params, 1234)
    directory = f"/cache/models/nemotron-parse/Nemotron-3.5-Parse/{npe.REVISION}"
    assert command[:7] == ["vllm", "serve", directory, "--host", "127.0.0.1", "--port", "1234"]
    options = " ".join(command[7:])
    for expected in (
        "--served-model-name nemotron-parse --trust-remote-code --dtype bfloat16",
        "--distributed-executor-backend uni --tensor-parallel-size 1 --max-model-len 16384",
        "--max-num-batched-tokens 16384 --no-enable-prefix-caching",
        '--limit-mm-per-prompt {"image": 1}',
        f"--chat-template {directory}/chat_template.jinja",
        "--max-num-seqs 16 --gpu-memory-utilization 0.6",
    ):
        assert expected in options
    assert "--attention-config" not in command  # vLLM chooses; FlashAttention 3 needs Hopper
    assert npe.server_env(params) == {
        "CUDA_VISIBLE_DEVICES": "0",
        "NP_VLLM_MTP": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    fa3 = npe.parse_params({"flash_attn_version": 3, "mtp": False, "device": "cuda:1"})
    assert npe.server_command(fa3, 1)[-2:] == ["--attention-config", '{"flash_attn_version": 3}']
    assert npe.server_env(fa3)["CUDA_VISIBLE_DEVICES"] == "1"
    assert npe.server_env(fa3)["NP_VLLM_MTP"] == "0"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"prompt": "OCR"}, "unknown parameters: prompt"),
        ({"device": "cpu"}, "device"),
        ({"mode": "triple"}, "mode must be 'single' or 'double'"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 100.0}, "max_tokens"),
        ({"max_tokens": 16384}, "below the 16384-token context"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 0.9}, r"gpu_memory_utilization must be in \(0, 0.6\]"),
        ({"max_num_seqs": 0}, "max_num_seqs"),
        ({"mtp": 1}, "mtp"),
        ({"flash_attn_version": 1}, "flash_attn_version"),
        ({"startup_seconds": -1}, "startup_seconds"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(npe.AdapterError, match=message):
        npe.parse_params(params)


def test_params_defaults():
    assert npe.parse_params({"max_tokens": None}) == npe.DEFAULT_PARAMS
    assert npe.DEFAULT_PARAMS == {
        "device": "cuda:0",
        "mode": "single",
        "max_tokens": 8744,
        "gpu_memory_utilization": 0.6,
        "max_num_seqs": 16,
        "mtp": True,
        "flash_attn_version": None,
        "startup_seconds": 900,
    }
    assert npe.parse_params(CONFIG["params"]) == npe.DEFAULT_PARAMS
    assert npe.parse_params(DOUBLE["params"]) == npe.DEFAULT_PARAMS | {"mode": "double"}


def test_greedy_sampling_and_guarded_merge():
    assert npe.FULLPAGE == {"temperature": 0.0, "top_k": 1, "repetition_penalty": 1.1}
    assert npe.CROP == {"temperature": 0.0, "top_k": 1, "repetition_penalty": 1.0}
    assert npe.MERGE_POLICY == "guarded"


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


def test_blocks_become_positioned_regions_with_markup(monkeypatch):
    monkeypatch.syspath_prepend(str(FAKE))
    from nemotron_parse import postprocess_document

    image = page_image(200, (1000, 700))
    raw = (FIXTURES / "output.txt").read_text(encoding="utf-8")
    document = postprocess_document(raw, image_size=image.size)
    prediction = npe.document_to_prediction(document, "s", image)
    regions = prediction["regions"]
    assert [r["native_type"] for r in regions] == [
        "Page-header",
        "Title",
        "Text",
        "List-item",
        "List-item",
        "Formula",
        "Picture",
        "Table",
        "Page-footer",
    ]
    assert [r["type"] for r in regions] == [
        "page-header",
        "heading",
        "text",
        "list-item",
        "list-item",
        "formula",
        "figure",
        "table",
        "page-footer",
    ]
    assert prediction["reading_order"] == [r["id"] for r in regions]
    corners = ([200, 56], [800, 56], [800, 98], [200, 98])  # content-normalized x 1000 and y 700
    assert regions[1]["polygon"] == [pytest.approx(corner) for corner in corners]
    assert regions[1]["markup"] == {"format": "markdown", "content": "The **Daily** Chronicle"}
    assert regions[-1]["polygon"][2] == [550, 700]  # y 1.02 is clipped to the image
    assert "markup" not in regions[6]  # the Picture block has no text
    assert prediction["text"] == ""
    standardized, issues = validate_prediction(prediction, sample_id="s", image_size=(1000, 700))
    assert issues == []
    assert [r.text for r in standardized.regions if r.text] == FIXTURE_TEXTS
    assert standardized.text == "\n\n".join(FIXTURE_TEXTS)
    assert standardized.conversion == {"from": "markdown", "version": 1}


def model_cache(root, corrupt=False, extra_file=False):
    """A cache with fake model files at the pinned path, and checksums that match them."""
    directory = root / "cache" / npe.PINNED_DIR
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
        "OCRBENCH_NEMOTRON_CACHE": str(root / "cache"),
        "OCRBENCH_NEMOTRON_CHECKSUMS": str(checksums),
    }


NEWSPAPER_LINES = 2000
# The fake server chooses its output by the gray level in the top-left corner (fake/README.md).
GRAY = {"page": 200, "crop": 120, "nohtml": 150, "news": 100, "loop": 80, "boom": 50, "die": 20}
GRAY["empty"] = 70


def make_dataset(root, samples):
    data = root / "data"
    data.mkdir()
    entries = []
    for sid, (kind, size) in samples.items():
        page_image(GRAY[kind], size).save(data / f"{sid}.png")
        if kind == "news":  # a newspaper page whose reference has 2,000 lines
            lines = [_newspaper_line(n) for n in range(1, NEWSPAPER_LINES + 1)]
            regions = [[(text, (10, n, 20, n + 1)) for n, text in enumerate(lines)]]
        elif kind in ("page", "crop"):
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
        "OCRBENCH_NEMOTRON_LOG": str(tmp_path / "vllm-server.log"),
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
        "nohtml": ("nohtml", (500, 500)),
        "boom": ("boom", (666, 400)),
        "news": ("news", (2000, 3000)),
        "loop": ("loop", (300, 200)),
        "big": ("page", (5000, 7000)),
        "page2": ("page", (1000, 700)),
        "die": ("die", (500, 500)),
        "after": ("page", (1000, 700)),
    }
    run, records = run_adapter(tmp_path, samples)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "empty": ("success", None),  # an empty output is an empty page
        "nohtml": ("failure", "engine-error"),
        "boom": ("failure", "engine-error"),
        "news": ("truncated", None),  # stopped at max_tokens, scored as produced
        "loop": ("truncated", None),  # no complete block before max_tokens: an empty prediction
        "big": ("success", None),
        "page2": ("success", None),  # the adapter keeps going after failed samples
        "die": ("failure", "engine-error"),
        "after": ("failure", "engine-error"),
    }
    assert records["empty"]["message"] == "the model returned an empty page"
    assert records["nohtml"]["message"] == "the output has no layout blocks (output.txt)"
    assert "fullpage request failed: HTTP 500" in records["boom"]["message"]
    assert "fake CUDA out of memory" in records["boom"]["message"]
    assert records["after"]["message"].startswith("the vLLM server has exited")

    page = run.path / "samples" / "page" / "attempt-1"
    assert {o["path"] for o in records["page"]["outputs"]} == {
        "output.txt",
        "nemotron.json",
        "page.md",
        "prediction.json",
        "result.json",
    }
    native = (page / "output" / "output.txt").read_text(encoding="utf-8")
    assert native == (FIXTURES / "output.txt").read_text(encoding="utf-8")  # unchanged
    stages = json.loads((page / "output" / "nemotron.json").read_text())["stages"]
    assert [(s["stage"], s["finish_reason"], s["raw_out"]) for s in stages] == [
        ("fullpage", "stop", native)
    ]
    assert records["page"]["findings"] == []
    engine_prediction = json.loads((page / "output" / "prediction.json").read_text())
    assert engine_prediction["text"] == ""
    assert "markup" in engine_prediction["regions"][0]
    prediction = json.loads((page / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": 1}  # converted on the host
    regions = prediction["regions"]
    assert [r["text"] for r in regions if "text" in r] == FIXTURE_TEXTS
    assert all("markup" not in r for r in regions)
    corners = ([400, 14], [600, 14], [600, 42], [400, 42])
    assert regions[0]["polygon"] == [pytest.approx(corner) for corner in corners]
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n\n".join(FIXTURE_TEXTS)

    extra = records["page"]["engine_extra"]
    assert (extra["mode"], extra["requests"], extra["crop_requests"]) == ("single", 1, 0)
    assert extra["stop_reasons"] == ["stop"]
    assert extra["generated_tokens"] == extra["fullpage_generated_tokens"] == len(native)
    assert extra["blocks"] == 9
    assert extra["source_image"] == {"width": 1000, "height": 700}
    assert extra["max_tokens"] == 8744
    environment = extra["environment"]
    assert (environment["vllm"], environment["nemotron-parse"]) == (
        "0.26.0+fake",
        "3.5.0.post7+fake",
    )
    assert (environment["gpu"], environment["gpu_capability"]) == ("Fake GPU", "8.9")
    assert environment["server_command"][:2] == ["vllm", "serve"]
    assert environment["server_env"]["NP_VLLM_MTP"] == "1"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["repository"], models["revision"]) == (npe.REPOSITORY, npe.REVISION)
    assert set(models["files"]) == {"config.json", "model.safetensors"}
    assert models["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert models["fullpage_sampling"] == npe.FULLPAGE
    assert models["merge_policy"] == "guarded"
    assert models["postprocessing"]["infer_headers"] is True
    assert extra["params"] == npe.DEFAULT_PARAMS

    empty = run.path / "samples" / "empty" / "attempt-1"
    assert (empty / "prediction.txt").read_text(encoding="utf-8") == ""
    assert (empty / "output" / "output.txt").read_text(encoding="utf-8") == ""  # kept
    nohtml = run.path / "samples" / "nohtml" / "attempt-1" / "output" / "output.txt"
    assert nohtml.read_text(encoding="utf-8") == "This page has no layout."  # kept on failure

    news = records["news"]
    assert news["message"] == "generation of the page stopped at max_tokens (8744 tokens)"
    assert news["engine_extra"]["stop_reasons"] == ["length"]
    news_text = (run.path / "samples" / "news" / "attempt-1" / "prediction.txt").read_text()
    assert news_text.startswith(_newspaper_line(1))
    assert _newspaper_line(100) in news_text

    loop = run.path / "samples" / "loop" / "attempt-1"
    assert (loop / "prediction.txt").read_text(encoding="utf-8") == ""
    assert records["loop"]["engine_extra"]["blocks"] == 0
    assert (loop / "output" / "output.txt").read_text(encoding="utf-8").endswith("ñññ")

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


def test_double_mode_refines_table_crops(tmp_path):
    samples = {"page": ("page", (1000, 700)), "crop": ("crop", (1000, 700))}
    run, records = run_adapter(tmp_path, samples, dict(CONFIG["params"], mode="double"))
    assert records["page"]["status"] == "success"
    extra = records["page"]["engine_extra"]
    assert (extra["mode"], extra["requests"], extra["crop_requests"]) == ("double", 2, 1)
    page = run.path / "samples" / "page" / "attempt-1"
    stages = json.loads((page / "output" / "nemotron.json").read_text())["stages"]
    assert [s["stage"] for s in stages] == ["fullpage", "crop"]
    assert stages[1]["image"] == {"width": 700, "height": 84}  # the table's box
    text = (page / "prediction.txt").read_text(encoding="utf-8")
    assert "1923\t2026" in text  # the crop's table replaces the page's
    assert "2O26" not in text

    # A crop cut off at max_tokens: the full page is used without refinement, and kept.
    crop = records["crop"]
    assert (crop["status"], crop["engine_extra"]["stop_reasons"]) == (
        "truncated",
        ["stop", "length"],
    )
    assert crop["message"] == (
        "generation of a Table/Chart crop stopped at max_tokens (40 tokens); "
        "the full page is used without crop refinement"
    )
    crop_dir = run.path / "samples" / "crop" / "attempt-1"
    assert "1923\t2O26" in (crop_dir / "prediction.txt").read_text(encoding="utf-8")
    stages = json.loads((crop_dir / "output" / "nemotron.json").read_text())["stages"]
    assert stages[1]["finish_reason"] == "length"


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "corrupt", "not matching models.sha256: ['model.safetensors']"),
        ({}, "extra", "model files not in models.sha256: ['extra.json']"),
        ({}, "missing", "does not exist; download the weights with make nemotron-parse-model"),
        ({}, {"FAKE_VLLM_EXIT": "1"}, "vLLM exited with status 3 during start-up"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env in ("corrupt", "extra"):
        env = model_cache(tmp_path / env, corrupt=env == "corrupt", extra_file=env == "extra")
    elif env == "missing":
        env = {"OCRBENCH_NEMOTRON_CACHE": str(tmp_path / "empty")}
    samples = {"a": ("page", (900, 420)), "b": ("empty", (300, 200))}
    _, records = run_adapter(tmp_path, samples, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "nemotron_parse_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "MODELS", Path(env["OCRBENCH_NEMOTRON_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main([str(tmp_path / "cache")])
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url.startswith(f"https://huggingface.co/{npe.REPOSITORY}/resolve/{npe.REVISION}/")
        Path(path).write_text("wrong", encoding="utf-8")

    (tmp_path / "cache" / npe.PINNED_DIR / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="config.json: SHA-256 differs from models.sha256"):
        fetch_model.main([str(tmp_path / "cache")])
    assert sorted(p.name for p in (tmp_path / "cache" / npe.PINNED_DIR).iterdir()) == [
        "model.safetensors"
    ]

    # The package source goes to the given directory, in subdirectories.
    source = tmp_path / "source.sha256"
    source.write_text(f"{hashlib.sha256(b'x').hexdigest()}  pkg/mod.py\n", encoding="utf-8")
    monkeypatch.setattr(fetch_model, "SOURCE", source)
    monkeypatch.setattr(
        fetch_model.urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(b"x")
    )
    fetch_model.main(["--source", str(tmp_path / "src")])
    assert (tmp_path / "src" / "pkg" / "mod.py").read_bytes() == b"x"


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def nemotron_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / npe.PINNED_DIR / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make nemotron-parse-model first")
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
@pytest.mark.parametrize("preset", ["engine.toml", "presets/double.toml"])
def test_gpu_smoke_benchmark(tmp_path, nemotron_image, preset):
    cache = str(nemotron_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / preset)
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "nemotron"]) == 0
    run = load_run(tmp_path / "runs" / "nemotron")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    # On the small blank page the model generates text in a repetition loop up to max_tokens
    # inside its first block: an empty truncated prediction.
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "truncated",
    }
    blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
    assert blank.read_text(encoding="utf-8") == ""
    extra = records["page-1"]["engine_extra"]
    environment = extra["environment"]
    assert (environment["vllm"], environment["nemotron-parse"]) == ("0.26.0", "3.5.0.post7")
    assert environment["torch"] == "2.11.0+cu130"
    assert environment["gpu"]
    assert (
        extra["models"]["files"]["model.safetensors"]
        == CHECKSUMS[CHECKSUMS.index("model.safetensors") - 1]
    )
    assert extra["models"]["prompt_sha256"] == UPSTREAM_PROMPT_SHA256
    assert extra["stop_reasons"][0] == "stop"
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
