"""docTR adapter (engines/doctr): result mapping, parameters, checkpoints, engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch``, ``numpy`` and
``doctr`` modules (tests/fixtures/doctr/fake). The fake returns committed results: real docTR
1.1.0 page exports (doctr.json) for synthetic pages rendered by tests/synthetic.py (page.json for
PAGE, columns.json for COLUMNS, newspaper.json for NEWSPAPER and blank.json for a blank page).
The opt-in GPU test builds the engine image and runs real docTR from the weights cache through
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
    page_xml,
    render,
    render_columns,
    smoke_dataset_with_columns,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "doctr"
FIXTURES = Path(__file__).parent / "fixtures" / "doctr"
ADAPTER_PATH = ENGINE_DIR / "doctr_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
DET2048 = tomllib.loads((ENGINE_DIR / "presets" / "det2048.toml").read_text(encoding="utf-8"))
CHECKSUMS = dict(
    reversed(line.split()) for line in (ENGINE_DIR / "models.sha256").read_text().splitlines()
)
NATIVE = {
    name: json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("page", "columns", "newspaper", "blank")
}
SIZES = {"page": (900, 420), "columns": COLUMNS[2], "newspaper": NEWSPAPER[2], "blank": (300, 200)}

_spec = importlib.util.spec_from_file_location("doctr_engine", ADAPTER_PATH)
dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dt)


def rendered(name, directory):
    """The PAGE regions of a fixture's synthetic page, rendered as for the fixture."""
    if name == "page":
        return render(directory / "page.png", PAGE, SIZES["page"])
    heading, columns, size, font_size = {"columns": COLUMNS, "newspaper": NEWSPAPER}[name]
    return render_columns(directory / f"{name}.png", heading, columns, size, font_size)


def errors(prediction, size):
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=size)
    assert standardized is not None, [i.format() for i in issues]
    return [i for i in issues if i.severity is not Severity.INFO]


def word(value, box):
    return {"value": value, "confidence": 0.9, "geometry": box, "objectness_score": 0.8}


def native(blocks, size=(900, 420)):
    """A minimal doctr.json: blocks of lines of ``(text, ((x0, y0), (x1, y1)))`` words."""
    return {
        "dimensions": [size[1], size[0]],
        "blocks": [
            {
                "geometry": [[0.0, 0.0], [1.0, 1.0]],
                "lines": [
                    {"geometry": [[0.0, 0.0], [1.0, 1.0]], "words": [word(*w) for w in line]}
                    for line in lines
                ],
                "artefacts": [],
            }
            for lines in blocks
        ],
        "layout": [],
        "tables": [],
    }


@pytest.mark.parametrize("name", ["page", "columns", "newspaper"])
def test_blocks_become_regions_in_doctr_reading_order(tmp_path, name):
    """docTR's reading order follows the rendered order, column by column."""
    prediction, counts = dt.result_to_prediction(NATIVE[name], "s1", SIZES[name])
    assert errors(prediction, SIZES[name]) == []
    regions = rendered(name, tmp_path)
    assert prediction["text"] == "\n\n".join("\n".join(t for t, _ in r) for r in regions)
    assert [[line["text"] for line in r["lines"]] for r in prediction["regions"]] == [
        [t for t, _ in r] for r in regions
    ]
    assert prediction["reading_order"] == [f"r{i + 1}" for i in range(len(regions))]
    assert {r["type"] for r in prediction["regions"]} == {"text"}
    lines = [line for r in prediction["regions"] for line in r["lines"]]
    assert [line["id"] for line in lines] == [f"l{i + 1}" for i in range(len(lines))]
    blocks = NATIVE[name]["blocks"]
    assert counts == {
        "blocks": len(blocks),
        "words": sum(len(line["words"]) for b in blocks for line in b["lines"]),
        "lines": len(lines),
        "regions": len(blocks),
    }


def test_relative_boxes_become_input_image_pixels(tmp_path):
    """docTR geometry is relative to the page; positions cover the rendered lines in pixels."""
    size = SIZES["newspaper"]
    prediction, _ = dt.result_to_prediction(NATIVE["newspaper"], "s1", size)
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    boxes = [b for region in rendered("newspaper", tmp_path) for _, b in region]
    for line, (x0, y0, x1, y1) in zip(lines, boxes, strict=True):
        xs, ys = [x for x, _ in line["polygon"]], [y for _, y in line["polygon"]]
        assert min(xs) - 5 <= x0 < x1 <= max(xs) + 5
        assert min(ys) - 5 <= y0 < y1 <= max(ys) + 5
        assert max(xs) - min(xs) < (x1 - x0) * 1.2
    for region, block in zip(prediction["regions"], NATIVE["newspaper"]["blocks"], strict=True):
        (bx0, by0), (bx1, by1) = block["geometry"]
        assert region["polygon"][0] == [round(bx0 * size[0], 2), round(by0 * size[1], 2)]
        assert region["polygon"][2] == [round(bx1 * size[0], 2), round(by1 * size[1], 2)]


def test_polygon():
    assert dt.polygon([[0.1, 0.25], [0.5, 0.75]], (200, 100)) == [
        [20.0, 25.0],
        [100.0, 25.0],
        [100.0, 75.0],
        [20.0, 75.0],
    ]
    rotated = [[0.1, 0.2], [0.5, 0.1], [0.6, 0.5], [0.2, 0.6]]
    assert dt.polygon(rotated, (100, 10)) == [[10.0, 2.0], [50.0, 1.0], [60.0, 5.0], [20.0, 6.0]]


def test_line_text_joins_its_words_and_blank_lines_are_dropped():
    box = ((0.1, 0.1), (0.2, 0.2))
    blocks = [
        [[("Hello", box), ("", box), ("world", box)], [(" ", box)]],
        [[("", box)]],  # no text: the block is not a region
        [[("Last", box)]],
    ]
    prediction, counts = dt.result_to_prediction(native(blocks), "s1", (900, 420))
    assert errors(prediction, (900, 420)) == []
    assert [[line["text"] for line in r["lines"]] for r in prediction["regions"]] == [
        ["Hello world"],
        ["Last"],
    ]
    assert [r["id"] for r in prediction["regions"]] == ["r1", "r2"]
    assert prediction["text"] == "Hello world\n\nLast"
    assert counts == {"blocks": 3, "words": 6, "lines": 2, "regions": 2}


def test_a_blank_page_is_an_empty_prediction():
    prediction, counts = dt.result_to_prediction(NATIVE["blank"], "s1", SIZES["blank"])
    assert errors(prediction, SIZES["blank"]) == []
    assert prediction["text"] == ""
    assert "regions" not in prediction
    assert "reading_order" not in prediction
    assert counts == {"blocks": 0, "words": 0, "lines": 0, "regions": 0}


def test_a_result_for_another_image_size_is_rejected():
    with pytest.raises(dt.AdapterError, match="docTR result is for a 900x420 image"):
        dt.result_to_prediction(native([]), "s1", (420, 900))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"psm": 3}, "unknown parameters: psm"),
        ({"device": "gpu:0"}, "device"),
        ({"device": "cuda"}, "device"),
        ({"device": 0}, "device"),
        ({"detection_size": 1000}, "detection_size"),
        ({"detection_size": 0}, "detection_size"),
        ({"detection_size": 2048.0}, "detection_size"),
        ({"detection_size": True}, "detection_size"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(dt.AdapterError, match=message):
        dt.parse_params(params)


def test_params_defaults():
    assert dt.parse_params({}) == dt.DEFAULT_PARAMS == {"device": "cuda:0", "detection_size": 1024}
    assert dt.parse_params({"device": None, "detection_size": None}) == dt.DEFAULT_PARAMS
    params = {"device": "cpu", "detection_size": 2048}
    assert dt.parse_params(params) == params
    assert CONFIG["params"] == dt.DEFAULT_PARAMS


def test_presets_differ_only_in_name_and_detection_size():
    assert (CONFIG["name"], DET2048["name"]) == ("doctr", "doctr-det2048")
    assert DET2048["params"] == {"device": "cuda:0", "detection_size": 2048}
    assert {**DET2048, "name": "doctr", "params": CONFIG["params"]} == CONFIG


def test_undecodable_image_is_an_adapter_error(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(dt.AdapterError, match="cannot decode image x.png"):
        dt.read_image(tmp_path / "x.png")
    Image.new("LA", (4, 3)).save(tmp_path / "la.png")
    assert dt.read_image(tmp_path / "la.png").mode == "RGB"


def test_checksums_cover_the_pinned_checkpoints():
    assert set(CHECKSUMS) == set(dt.CHECKPOINTS.values())
    for name, digest in CHECKSUMS.items():
        # docTR checks the first 8 hex digits of the SHA-256 against the file name
        assert name.removesuffix(".pt").rsplit("-", 1)[1] == digest[:8]
        assert len(digest) == 64
    for role, checkpoint in dt.CHECKPOINTS.items():
        assert checkpoint == f"models/{dt.ARCHS[role]}-{checkpoint.rsplit('-', 1)[1]}"
        assert dt.SOURCES[role].endswith("/" + checkpoint.removeprefix("models/"))
    assert CONFIG["model"] == "doctr-1.1.0:fast_base-688a8b34+crnn_vgg16_bn-0417f351"
    requirements = (ENGINE_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "python-doctr==1.1.0 " in requirements
    assert "opencv-python-headless==" in requirements
    assert "\nopencv-python==" not in requirements


# Engine protocol through the runner, with the fake modules ---------------------------------------


def model_cache(root, corrupt=None):
    """A cache with fake checkpoint files, and checksums that match them."""
    cache = root / "cache"
    lines = []
    for name in dt.CHECKPOINTS.values():
        path = cache / dt.MODELS_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fake {name}", encoding="utf-8")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    if corrupt:
        (cache / dt.MODELS_DIR / corrupt).write_text("changed", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"OCRBENCH_DOCTR_CACHE": str(cache), "OCRBENCH_DOCTR_CHECKSUMS": str(checksums)}


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    samples = []
    heights = {width: SIZES[name][1] for name, width in (("newspaper", 1500), ("blank", 300))}
    for sid, width in widths.items():
        Image.new("L", (width, heights.get(width, 420)), 255).save(data / f"{sid}.png")
        page_xml(data / f"{sid}.xml", [], (width, heights.get(width, 420)))
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
    widths = {"news": 1500, "blank": 300, "boom": 666, "empty": 500, "page": 900}
    run, records = run_adapter(tmp_path, widths)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "news": ("success", None),
        "blank": ("success", None),
        "boom": ("failure", "engine-error"),
        "empty": ("success", None),  # nothing detected: a genuine empty result
        "page": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: CUDA out of memory (fake)" in records["boom"]["message"]
    news = run.path / "samples" / "news" / "attempt-1"
    outputs = {o["path"] for o in records["news"]["outputs"]}
    assert outputs == {"doctr.json", "prediction.json", "result.json"}
    assert json.loads((news / "output" / "doctr.json").read_text()) == NATIVE["newspaper"]
    expected, counts = dt.result_to_prediction(NATIVE["newspaper"], "news", SIZES["newspaper"])
    assert (news / "prediction.txt").read_text(encoding="utf-8") == expected["text"]
    extra = records["news"]["engine_extra"]
    assert {k: extra[k] for k in counts} == counts
    assert extra["doctr_seconds"] >= 0
    environment = extra["environment"]
    assert environment["doctr"] == "1.1.0+fake"
    assert environment["detection_input"] == [1024, 1024]  # docTR's own pre-processor
    assert (environment["gpu"], environment["gpu_memory_mb"]) == ("Fake GPU", 24 << 10)
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert (models["archs"], models["checkpoints"]) == (dt.ARCHS, dt.CHECKPOINTS)
    assert models["sources"] == dt.SOURCES
    assert set(models["files"]) == set(dt.CHECKPOINTS.values())
    assert all(len(digest) == 64 for digest in models["files"].values())
    assert extra["params"] == dt.DEFAULT_PARAMS
    for sid in ("blank", "empty"):
        path = run.path / "samples" / sid / "attempt-1" / "prediction.txt"
        assert path.read_text(encoding="utf-8") == ""

    assert main(["evaluate", str(run.path)]) == 0
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


def test_cpu_device_and_detection_size_reach_doctr(tmp_path):
    params = {"device": "cpu", "detection_size": 2048}
    _, records = run_adapter(tmp_path, {"page": 900}, params, {"FAKE_TORCH_GPUS": "0"})
    assert records["page"]["status"] == "success"
    extra = records["page"]["engine_extra"]
    assert extra["params"] == params
    assert extra["environment"]["detection_input"] == [2048, 2048]
    assert "gpu" not in extra["environment"]


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "no-cache", "does not exist; download the weights with make doctr-model"),
        ({}, "corrupt", "not matching models.sha256: ['models/fast_base-688a8b34.pt']"),
        ({}, "missing", "cannot load docTR: OSError: no checkpoint at"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "no-cache":
        env = {"OCRBENCH_DOCTR_CACHE": str(tmp_path / "nowhere")}
    elif env == "corrupt":
        env = model_cache(tmp_path / "corrupt", "models/fast_base-688a8b34.pt")
    elif env == "missing":
        # Not pinned in this checksum file, so only docTR's loading notices it is missing.
        env = model_cache(tmp_path / "partial")
        checksums = Path(env["OCRBENCH_DOCTR_CHECKSUMS"])
        lines = checksums.read_text().splitlines()
        checksums.write_text("\n".join(x for x in lines if "crnn" not in x))
        (Path(env["OCRBENCH_DOCTR_CACHE"]) / dt.MODELS_DIR / dt.CHECKPOINTS["recognition"]).unlink()
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "doctr_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_DOCTR_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    def wrong_download(url, path):
        assert url == dt.SOURCES["recognition"]
        Path(path).write_text("wrong", encoding="utf-8")

    models = tmp_path / "cache" / dt.MODELS_DIR / "models"
    (models / "crnn_vgg16_bn-0417f351.pt").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="crnn_vgg16_bn-0417f351.pt: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in models.iterdir()) == ["fast_base-688a8b34.pt"]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def doctr_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / dt.MODELS_DIR / dt.CHECKPOINTS["recognition"]
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make doctr-model first")
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
def test_gpu_smoke_benchmark(tmp_path, doctr_image):
    cache = str(doctr_image)
    manifest = smoke_dataset_with_columns(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    preset = str(ENGINE_DIR / "presets" / "det2048.toml")
    paths = []
    for run_id, config, size in (("doctr", engine, 1024), ("det2048", preset, 2048)):
        run_args = [*args[:3], config, *args[4:], "--cache", cache, "--run-id", run_id]
        assert main(run_args) == 0
        run = load_run(tmp_path / "runs" / run_id)
        paths.append(str(run.path))
        records = {s.id: latest_record(run, s.id) for s in run.samples}
        assert {sid: r["status"] for sid, r in records.items()} == dict.fromkeys(
            ("page-1", "para-1", "blank-1", "columns-1", "news-1"), "success"
        )
        environment = records["page-1"]["engine_extra"]["environment"]
        assert environment["detection_input"] == [size, size]
        assert environment["doctr"] == "1.1.0"
        assert environment["torch"] == "2.14.1+cu130"
        assert environment["gpu"]
        assert records["page-1"]["engine_extra"]["models"]["files"] == CHECKSUMS
        for sid in ("columns-1", "news-1"):
            prediction = json.loads(
                (run.path / "samples" / sid / "attempt-1" / "prediction.json").read_text()
            )
            assert prediction["reading_order"]
            assert all("polygon" in line for r in prediction["regions"] for line in r["lines"])
        blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
        assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure

        # Scoring and re-scoring read only stored outputs: no container runs.
        for _ in range(2):
            assert main(["evaluate", str(run.path)]) == 0
        scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
        rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
        for sid in ("page-1", "para-1", "columns-1", "news-1"):
            assert rates[sid] < 0.1, (run_id, sid)  # ordered CER: columns one after another
    report = tmp_path / "report"
    assert main(["report", *paths, "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make doctr-model" in r["message"] for r in records)
