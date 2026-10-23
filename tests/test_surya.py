"""Surya OCR adapter (engines/surya): result mapping, parameters, checkpoints, engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch``,
``transformers`` and ``surya`` modules (tests/fixtures/surya/fake). The fake returns committed
results: real Surya 0.17.1 output (surya.json, without the per-character boxes) for synthetic
pages rendered by tests/synthetic.py (page.json for PAGE, columns.json for COLUMNS,
newspaper.json for NEWSPAPER and blank.json for a blank page). The opt-in GPU test builds the
engine image and runs real Surya from the weights cache through ``ocrbench run``, ``evaluate``
and ``report``.
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
ENGINE_DIR = ROOT / "engines" / "surya"
FIXTURES = Path(__file__).parent / "fixtures" / "surya"
ADAPTER_PATH = ENGINE_DIR / "surya_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = dict(
    reversed(line.split()) for line in (ENGINE_DIR / "models.sha256").read_text().splitlines()
)
NATIVE = {
    name: json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("page", "columns", "newspaper", "blank")
}
SIZES = {"page": (900, 420), "columns": COLUMNS[2], "newspaper": NEWSPAPER[2], "blank": (300, 200)}

_spec = importlib.util.spec_from_file_location("surya_engine", ADAPTER_PATH)
su = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(su)


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


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def native(blocks, lines, size=(900, 420)):
    """A minimal surya.json: layout blocks ``(label, bbox)``, lines ``(text, box)``.

    Blocks are listed in reverse, so their ``position`` alone gives the reading order.
    """
    image_bbox = [0, 0, *size]
    polygons = [box(*b) for _, b in lines]
    layout = [
        {"polygon": box(*b), "label": label, "position": i, "confidence": 0.9}
        for i, (label, b) in enumerate(blocks)
    ]
    return {
        "detection": {"bboxes": [{"polygon": p, "confidence": 1.0} for p in polygons]}
        | {"image_bbox": image_bbox},
        "recognition": {
            "text_lines": [
                {"polygon": p, "text": text, "confidence": 0.9, "chars": []}
                for p, (text, _) in zip(polygons, lines, strict=True)
            ],
            "image_bbox": image_bbox,
        },
        "layout": {"bboxes": layout[::-1], "image_bbox": image_bbox, "sliced": False},
    }


@pytest.mark.parametrize("name", ["page", "columns", "newspaper"])
def test_blocks_become_regions_in_reading_order_with_their_lines(tmp_path, name):
    """The page text follows the rendered reading order, column by column."""
    prediction, counts = su.result_to_prediction(NATIVE[name], "s1", SIZES[name])
    assert errors(prediction, SIZES[name]) == []
    regions = rendered(name, tmp_path)
    assert prediction["text"] == "\n\n".join("\n".join(t for t, _ in r) for r in regions)
    assert [[line["text"] for line in r["lines"]] for r in prediction["regions"]] == [
        [t for t, _ in r] for r in regions
    ]
    blocks = NATIVE[name]["layout"]["bboxes"]
    assert prediction["reading_order"] == [f"r{i + 1}" for i in range(len(blocks))]
    for region, block in zip(prediction["regions"], blocks, strict=True):
        assert region["native_type"] == block["label"]
        assert region["polygon"] == block["polygon"]
        assert region["extra"] == {"position": block["position"]}
    assert [r["type"] for r in prediction["regions"]] == ["heading"] + ["text"] * (len(blocks) - 1)
    lines = NATIVE[name]["recognition"]["text_lines"]
    assert counts == {
        "detected_lines": len(lines),
        "lines": len(lines),
        "unassigned_lines": 0,
        "layout_blocks": len(blocks),
        "regions": len(blocks),
    }
    for region in prediction["regions"]:
        for line in region["lines"]:
            native_line = lines[int(line["id"].removeprefix("l")) - 1]
            assert line["polygon"] == native_line["polygon"]  # input-image pixels, as is
            assert line["confidence"] == round(native_line["confidence"], 4)


@pytest.mark.parametrize("name", ["columns", "newspaper"])
def test_columns_are_not_interleaved_like_the_detection_order(tmp_path, name):
    """Surya detects side-by-side lines row by row; the layout reading order undoes that."""
    first_column = [t for t, _ in rendered(name, tmp_path)[1]]
    texts = [line["text"] for line in NATIVE[name]["recognition"]["text_lines"]]
    assert texts[1:3] != first_column[:2]  # the native line order interleaves the columns
    prediction, _ = su.result_to_prediction(NATIVE[name], "s1", SIZES[name])
    assert [line["text"] for line in prediction["regions"][1]["lines"]] == first_column


def test_line_polygons_cover_the_rendered_lines_in_input_image_pixels(tmp_path):
    prediction, _ = su.result_to_prediction(NATIVE["newspaper"], "s1", SIZES["newspaper"])
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    boxes = [b for region in rendered("newspaper", tmp_path) for _, b in region]
    for line, (x0, y0, x1, y1) in zip(lines, boxes, strict=True):
        xs, ys = [x for x, _ in line["polygon"]], [y for _, y in line["polygon"]]
        assert min(xs) - 5 <= x0 < x1 <= max(xs) + 5
        assert min(ys) - 5 <= y0 < y1 <= max(ys) + 5
        assert max(xs) - min(xs) < (x1 - x0) * 1.2


def test_inline_formatting_is_removed_from_the_text_but_kept_natively():
    heading = NATIVE["newspaper"]["recognition"]["text_lines"][0]["text"]
    assert heading == "<b>Evening Gazette</b>"  # what Surya wrote
    prediction, _ = su.result_to_prediction(NATIVE["newspaper"], "s1", SIZES["newspaper"])
    assert prediction["regions"][0]["lines"][0]["text"] == "Evening Gazette"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<b>Bold</b> and <i>italic</i>", "Bold and italic"),
        ("x<sup>2</sup> H<sub>2</sub>O", "x2 H2O"),
        ("<u>a</u><del>b</del><mark>c</mark><s>d</s><small>e</small>", "abcde"),
        ("one<br>two<br/>three<BR />four", "one two three four"),
        ("Fish &amp; chips &lt;3", "Fish & chips <3"),
        ("<math>x^2</math> <bold>", "<math>x^2</math> <bold>"),  # other tags are text
    ],
)
def test_plain_text(text, expected):
    assert su.plain_text(text) == expected


def test_region_types_follow_the_layout_labels():
    labels = ["SectionHeader", "ListItem", "PageFooter", "Equation", "Picture", "Form", "Code"]
    blocks = [(label, (10, 10 + 40 * i, 800, 40 + 40 * i)) for i, label in enumerate(labels)]
    prediction, _ = su.result_to_prediction(native(blocks, []), "s1", (900, 420))
    assert [r["type"] for r in prediction["regions"]] == [
        "heading",
        "list-item",
        "page-footer",
        "formula",
        "figure",
        "other",
        "other",
    ]
    assert [r["native_type"] for r in prediction["regions"]] == labels
    assert all("lines" not in r for r in prediction["regions"])
    assert prediction["text"] == ""


def test_each_line_goes_to_the_block_that_contains_most_of_it():
    blocks = [("Text", (0, 0, 100, 100)), ("Text", (100, 0, 200, 100)), ("Text", (0, 0, 200, 100))]
    lines = [
        ((10, 10, 90, 20)),  # inside the first and the third: tie, the earlier one
        ((60, 30, 180, 40)),  # inside the third only completely
        ((150, 50, 300, 60)),  # a third inside the second, a third inside the third
        ((10, 200, 90, 210)),  # outside every block
    ]
    assert su.assign_lines([list(b) for b in lines], [list(b) for _, b in blocks]) == [
        0,
        2,
        None,
        None,
    ]
    assert su.assign_lines([[10, 10, 90, 10]], [[0, 0, 100, 100]]) == [None]  # no area
    assert su.assign_lines([[0, 0, 10, 10]], [[0, 0, 10, 5]]) == [0]  # half inside is enough


def test_lines_outside_every_block_follow_in_their_own_region():
    blocks = [("Text", (0, 300, 900, 400)), ("SectionHeader", (0, 0, 900, 50))]
    lines = [
        ("Below the layout", (10, 410, 300, 418)),
        ("Second paragraph", (10, 310, 300, 330)),
        ("Heading", (10, 10, 300, 40)),
        ("In the margin", (10, 100, 300, 120)),
        ("   ", (10, 340, 300, 360)),  # blank text: not a line
    ]
    prediction, counts = su.result_to_prediction(native(blocks, lines), "s1", (900, 420))
    assert errors(prediction, (900, 420)) == []
    assert [[line["text"] for line in r.get("lines", [])] for r in prediction["regions"]] == [
        ["Second paragraph"],
        ["Heading"],
        ["In the margin", "Below the layout"],
    ]
    assert [r["extra"] for r in prediction["regions"]] == [
        {"position": 0},
        {"position": 1},
        {"grouping": "unassigned"},
    ]
    unassigned = prediction["regions"][2]
    assert unassigned["type"] == "text"
    assert "polygon" not in unassigned
    assert [line["id"] for line in unassigned["lines"]] == ["l4", "l1"]  # native line numbers
    assert prediction["text"] == "Second paragraph\n\nHeading\n\nIn the margin\nBelow the layout"
    assert counts == {
        "detected_lines": 5,
        "lines": 4,
        "unassigned_lines": 2,
        "layout_blocks": 2,
        "regions": 3,
    }


def test_lines_of_a_block_are_ordered_by_visual_line_then_left_to_right():
    lines = [
        ("second", (10, 50, 200, 70)),
        ("right", (400, 12, 600, 32)),
        ("left", (10, 10, 200, 30)),
    ]
    prediction, _ = su.result_to_prediction(
        native([("Text", (0, 0, 900, 100))], lines), "s1", (900, 420)
    )
    assert [line["text"] for line in prediction["regions"][0]["lines"]] == [
        "left",
        "right",
        "second",
    ]


def test_a_blank_page_is_an_empty_prediction():
    prediction, counts = su.result_to_prediction(NATIVE["blank"], "s1", SIZES["blank"])
    assert errors(prediction, SIZES["blank"]) == []
    assert prediction["text"] == ""
    # Surya's layout model reports the blank page as one picture; the region has no text.
    assert [(r["native_type"], "lines" in r) for r in prediction["regions"]] == [("Picture", False)]
    assert counts["detected_lines"] == counts["lines"] == 0
    empty = native([], [])
    prediction, _ = su.result_to_prediction(empty, "s1", (900, 420))
    assert prediction["text"] == ""
    assert "regions" not in prediction


def test_inconsistent_results_are_rejected():
    result = native([], [("a", (10, 10, 50, 20))])
    result["recognition"]["text_lines"] = []
    with pytest.raises(su.AdapterError, match="recognized 0 of 1 detected lines"):
        su.result_to_prediction(result, "s1", (900, 420))
    for name in ("detection", "recognition", "layout"):
        result = native([], [])
        result[name]["image_bbox"] = [0, 0, 450, 210]
        with pytest.raises(su.AdapterError, match=f"{name} result is for a 450x210 image"):
            su.result_to_prediction(result, "s1", (900, 420))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"psm": 3}, "unknown parameters: psm"),
        ({"device": "gpu:0"}, "device"),
        ({"device": "cuda"}, "device"),
        ({"math_mode": "no"}, "math_mode"),
        ({"math_mode": 0}, "math_mode"),
        ({"recognition_batch_size": 0}, "recognition_batch_size"),
        ({"recognition_batch_size": 8.0}, "recognition_batch_size"),
        ({"recognition_batch_size": True}, "recognition_batch_size"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(su.AdapterError, match=message):
        su.parse_params(params)


def test_params_defaults():
    assert su.parse_params({}) == su.DEFAULT_PARAMS
    assert su.parse_params({"device": None, "math_mode": None}) == su.DEFAULT_PARAMS
    assert su.DEFAULT_PARAMS == {
        "device": "cuda:0",
        "math_mode": False,
        "recognition_batch_size": None,
    }
    params = su.parse_params({"device": "cpu", "math_mode": True, "recognition_batch_size": 32})
    assert params == {"device": "cpu", "math_mode": True, "recognition_batch_size": 32}
    assert CONFIG["params"] == {"device": "cuda:0", "math_mode": False}


def test_undecodable_image_is_an_adapter_error(tmp_path):
    (tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(su.AdapterError, match="cannot decode image x.png"):
        su.read_image(tmp_path / "x.png")
    Image.new("LA", (4, 3)).save(tmp_path / "la.png")
    assert su.read_image(tmp_path / "la.png").mode == "RGB"


def test_checksums_cover_the_pinned_checkpoints():
    assert CONFIG["model"] == "surya-0.17.1:" + "+".join(su.CHECKPOINTS.values())
    for checkpoint in su.CHECKPOINTS.values():
        assert f"{checkpoint}/config.json" in CHECKSUMS
        assert f"{checkpoint}/model.safetensors" in CHECKSUMS
    for name in ("recognition", "layout"):  # the tokenizer files of the foundation models
        for file in ("vocab_math.json", "specials_dict.json", "processor_config.json"):
            assert f"{su.CHECKPOINTS[name]}/{file}" in CHECKSUMS
    assert all(name.split("/", 2)[2] != "training_args.bin" for name in CHECKSUMS)
    assert all(len(digest) == 64 for digest in CHECKSUMS.values())
    requirements = (ENGINE_DIR / "requirements.txt").read_text(encoding="utf-8")
    assert "surya-ocr==0.17.1 " in requirements
    assert "transformers==4.57.6 " in requirements


# Engine protocol through the runner, with the fake modules ---------------------------------------


def model_cache(root, corrupt=None):
    """A cache with fake checkpoint files, and checksums that match them."""
    cache = root / "cache"
    lines = []
    for checkpoint in su.CHECKPOINTS.values():
        for file in ("config.json", "model.safetensors"):
            path = cache / su.MODELS_DIR / checkpoint / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fake {checkpoint} {file}", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {checkpoint}/{file}")
    if corrupt:
        (cache / su.MODELS_DIR / corrupt).write_text("changed", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"OCRBENCH_SURYA_CACHE": str(cache), "OCRBENCH_SURYA_CHECKSUMS": str(checksums)}


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
    assert outputs == {"surya.json", "prediction.json", "result.json"}
    assert json.loads((news / "output" / "surya.json").read_text()) == NATIVE["newspaper"]
    expected, counts = su.result_to_prediction(NATIVE["newspaper"], "news", SIZES["newspaper"])
    assert (news / "prediction.txt").read_text(encoding="utf-8") == expected["text"]
    extra = records["news"]["engine_extra"]
    assert {k: extra[k] for k in counts} == counts
    assert extra["surya_seconds"] >= 0
    environment = extra["environment"]
    assert (environment["surya"], environment["transformers"]) == ("0.17.1+fake", "4.57.6+fake")
    assert (environment["gpu"], environment["gpu_memory_mb"]) == ("Fake GPU", 24 << 10)
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["checkpoints"] == su.CHECKPOINTS
    assert models["source"] == "https://models.datalab.to"
    assert len(models["files"]) == 6
    assert all(len(digest) == 64 for digest in models["files"].values())
    assert extra["params"] == su.DEFAULT_PARAMS
    for sid in ("blank", "empty"):
        path = run.path / "samples" / sid / "attempt-1" / "prediction.txt"
        assert path.read_text(encoding="utf-8") == ""

    assert main(["evaluate", str(run.path)]) == 0
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()


def test_cpu_device_and_parameters_reach_surya(tmp_path):
    params = {"device": "cpu", "math_mode": True, "recognition_batch_size": 16}
    _, records = run_adapter(tmp_path, {"page": 900}, params, {"FAKE_TORCH_GPUS": "0"})
    assert records["page"]["status"] == "success"
    extra = records["page"]["engine_extra"]
    assert extra["params"] == params
    assert "gpu" not in extra["environment"]


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({}, "no-cache", "does not exist; download the weights with make surya-model"),
        ({}, "corrupt", "not matching models.sha256: ['layout/2025_09_23/model.safetensors']"),
        ({}, "missing-config", "cannot load Surya: OSError: no checkpoint at"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "no-cache":
        env = {"OCRBENCH_SURYA_CACHE": str(tmp_path / "nowhere")}
    elif env == "corrupt":
        env = model_cache(tmp_path / "corrupt", "layout/2025_09_23/model.safetensors")
    elif env == "missing-config":
        # Not pinned in this checksum file, so only Surya's loading notices it is missing.
        env = model_cache(tmp_path / "partial")
        checksums = Path(env["OCRBENCH_SURYA_CHECKSUMS"])
        lines = checksums.read_text().splitlines()
        checksums.write_text("\n".join(x for x in lines if "detection/2025_05_07/config" not in x))
        config = Path(env["OCRBENCH_SURYA_CACHE"]) / su.MODELS_DIR / "text_detection"
        (config / "2025_05_07" / "config.json").unlink()
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params, env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    monkeypatch.delitem(sys.modules, "surya_engine", raising=False)
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_SURYA_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 6

    def wrong_download(url, path):
        assert url == "https://models.datalab.to/layout/2025_09_23/config.json"
        Path(path).write_text("wrong", encoding="utf-8")

    layout = tmp_path / "cache" / su.MODELS_DIR / "layout" / "2025_09_23"
    (layout / "config.json").unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="layout/2025_09_23/config.json: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    assert sorted(p.name for p in layout.iterdir()) == ["model.safetensors"]


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def surya_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / su.MODELS_DIR / su.CHECKPOINTS["recognition"] / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make surya-model first")
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
def test_gpu_smoke_benchmark(tmp_path, surya_image):
    cache = str(surya_image)
    manifest = smoke_dataset_with_columns(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "surya"]) == 0
    run = load_run(tmp_path / "runs" / "surya")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == dict.fromkeys(
        ("page-1", "para-1", "blank-1", "columns-1", "news-1"), "success"
    )
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"]["surya"] == "0.17.1"
    assert extra["environment"]["transformers"] == "4.57.6"
    assert extra["environment"]["torch"] == "2.14.1+cu130"
    assert extra["environment"]["gpu"]
    assert extra["models"]["files"] == CHECKSUMS
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
        assert rates[sid] < 0.1, sid  # ordered CER: the columns are read one after another
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make surya-model" in r["message"] for r in records)
