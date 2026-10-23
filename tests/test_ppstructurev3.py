"""PP-StructureV3 adapter (engines/ppstructurev3): result mapping, parameters, engine protocol.

Offline tests import the adapter and run it as a host process with a fake ``paddleocr``
(tests/fixtures/ppstructurev3/fake) and the fake ``paddle``, ``paddlex`` and ``cv2`` modules of
tests/fixtures/paddleocr/fake. The fake returns committed results: real PP-StructureV3 output of
PaddleOCR 3.7.0 for synthetic pages rendered by tests/synthetic.py (page.json for PAGE,
columns.json for COLUMNS, newspaper.json for NEWSPAPER and blank.json for a blank page). The
opt-in GPU test builds the engine image and runs real PP-StructureV3 through ``ocrbench run``,
``evaluate`` and ``report``.
"""

import importlib.util
import json
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
    render,
    render_columns,
    smoke_dataset_with_columns,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "ppstructurev3"
FIXTURES = Path(__file__).parent / "fixtures" / "ppstructurev3"
FAKES = [FIXTURES / "fake", Path(__file__).parent / "fixtures" / "paddleocr" / "fake"]
ADAPTER_PATH = ENGINE_DIR / "ppstructurev3_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
PRESETS = {
    path.stem: tomllib.loads(path.read_text(encoding="utf-8"))
    for path in sorted((ENGINE_DIR / "presets").glob("*.toml"))
}
NATIVE = {
    name: json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("page", "columns", "newspaper", "blank")
}
SIZES = {"page": (900, 420), "columns": COLUMNS[2], "newspaper": NEWSPAPER[2], "blank": (300, 200)}

_spec = importlib.util.spec_from_file_location("ppstructurev3_engine", ADAPTER_PATH)
pps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pps)


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


def native(blocks, lines, size=(900, 420), orders=None):
    """A minimal PP-StructureV3 result: blocks ``(label, bbox)``, lines ``(text, box)``.

    ``orders`` are the blocks' ``block_order`` values, 1, 2, … by default.
    """
    poly = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for _, (x1, y1, x2, y2) in lines]
    ocr = {
        "rec_texts": [text for text, _ in lines],
        "rec_polys": poly,
        "rec_scores": [0.9] * len(lines),
        "rec_boxes": [list(box) for _, box in lines],
    }
    parsing = [
        {"block_label": label, "block_content": "", "block_bbox": list(bbox), "block_id": i}
        | {"block_order": i + 1 if orders is None else orders[i]}
        for i, (label, bbox) in enumerate(blocks)
    ]
    res = {"width": size[0], "height": size[1], "model_settings": {"use_doc_preprocessor": False}}
    return {"res": res | {"parsing_res_list": parsing, "overall_ocr_res": ocr}}


@pytest.mark.parametrize("name", ["page", "columns", "newspaper"])
def test_blocks_become_regions_in_reading_order_with_their_lines(tmp_path, name):
    """The page text follows the rendered reading order, column by column."""
    prediction, counts = pps.result_to_prediction(NATIVE[name], "s1", SIZES[name])
    assert errors(prediction, SIZES[name]) == []
    regions = rendered(name, tmp_path)
    assert prediction["text"] == "\n\n".join("\n".join(t for t, _ in r) for r in regions)
    assert [[line["text"] for line in r["lines"]] for r in prediction["regions"]] == [
        [t for t, _ in r] for r in regions
    ]
    blocks = NATIVE[name]["res"]["parsing_res_list"]
    assert prediction["reading_order"] == [f"b{i + 1}" for i in range(len(blocks))]
    for region, block in zip(prediction["regions"], blocks, strict=True):
        assert region["native_type"] == block["block_label"]
        x1, y1, x2, y2 = block["block_bbox"]
        assert region["polygon"] == [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        assert region["extra"] == {
            "block_id": block["block_id"],
            "block_order": block["block_order"],
        }
    ocr = NATIVE[name]["res"]["overall_ocr_res"]
    assert counts == {"lines": len(ocr["rec_texts"]), "assigned": len(ocr["rec_texts"])} | {
        "outside_blocks": 0
    }
    for region in prediction["regions"]:
        for line in region["lines"]:
            index = int(line["id"].removeprefix("l")) - 1
            assert line["text"] == ocr["rec_texts"][index]
            assert line["polygon"] == ocr["rec_polys"][index]  # input-image pixels, as is
            assert line["confidence"] == round(ocr["rec_scores"][index], 4)


@pytest.mark.parametrize("name", ["columns", "newspaper"])
def test_columns_are_not_interleaved_like_the_overall_ocr_order(tmp_path, name):
    """PP-OCR orders the lines of side-by-side columns row by row; the blocks undo that."""
    first_column = [t for t, _ in rendered(name, tmp_path)[1]]
    ocr_texts = NATIVE[name]["res"]["overall_ocr_res"]["rec_texts"]
    assert ocr_texts[1:3] != first_column[:2]  # the native line order interleaves the columns
    prediction, _ = pps.result_to_prediction(NATIVE[name], "s1", SIZES[name])
    assert [line["text"] for line in prediction["regions"][1]["lines"]] == first_column


def test_line_polygons_cover_the_rendered_lines_in_input_image_pixels(tmp_path):
    prediction, _ = pps.result_to_prediction(NATIVE["newspaper"], "s1", SIZES["newspaper"])
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    boxes = [box for region in rendered("newspaper", tmp_path) for _, box in region]
    for line, (x0, y0, x1, y1) in zip(lines, boxes, strict=True):
        xs, ys = [x for x, _ in line["polygon"]], [y for _, y in line["polygon"]]
        assert min(xs) - 2 <= x0 < x1 <= max(xs) + 2  # tight detected boxes, within 2 px
        assert min(ys) - 2 <= y0 < y1 <= max(ys) + 2
        assert max(xs) - min(xs) < (x1 - x0) * 1.1


def test_region_types_follow_the_block_labels():
    labels = ["doc_title", "paragraph_title", "text", "header", "footer", "number", "image"]
    labels += ["figure_title", "footnote", "table", "formula", "aside_text", "seal", "mystery"]
    blocks = [(label, (10 * i, 0, 10 * i + 8, 8)) for i, label in enumerate(labels)]
    prediction, _ = pps.result_to_prediction(native(blocks, []), "s1", (900, 420))
    types = [region["type"] for region in prediction["regions"]]
    assert types == [
        *("heading", "heading", "text", "page-header", "page-footer", "page-number", "figure"),
        *("caption", "footnote", "table", "formula", "text", "figure", "other"),
    ]
    assert all("lines" not in region for region in prediction["regions"])  # no fabricated text
    assert prediction["text"] == ""
    assert errors(prediction, (900, 420)) == []


def test_each_line_goes_to_the_block_it_overlaps_most():
    blocks = [
        ("text", (0, 0, 100, 100)),
        ("text", (100, 0, 200, 100)),
        ("image", (0, 200, 99, 299)),
    ]
    lines = [
        ("left", (10, 10, 90, 20)),
        ("mostly right", (90, 30, 190, 40)),  # 10 px in the left block, 90 in the right
        ("touching", (97, 50, 103, 60)),  # 3 px into each block: matches neither
        ("tie", (80, 70, 120, 80)),  # 20 px in each block: the earlier block
        ("outside", (300, 300, 400, 310)),
        ("  ", (10, 85, 90, 95)),  # blank text is dropped
    ]
    prediction, counts = pps.result_to_prediction(native(blocks, lines), "s1", (900, 420))
    got = {r["id"]: [line["text"] for line in r.get("lines", [])] for r in prediction["regions"]}
    assert got == {"b1": ["left", "tie"], "b2": ["mostly right"], "b3": []}
    assert counts == {"lines": 5, "assigned": 3, "outside_blocks": 2}
    assert prediction["text"] == "left\ntie\n\nmostly right"
    assert errors(prediction, (900, 420)) == []


def test_lines_of_a_block_are_ordered_by_visual_line_then_left_to_right():
    blocks = [("text", (0, 0, 400, 100))]
    lines = [
        ("second", (10, 40, 100, 60)),
        ("first-b", (210, 11, 300, 31)),  # on the first visual line, right of first-a
        ("first-a", (10, 10, 200, 30)),
        ("third", (10, 70, 100, 90)),
    ]
    prediction, _ = pps.result_to_prediction(native(blocks, lines), "s1", (900, 420))
    texts = [line["text"] for line in prediction["regions"][0]["lines"]]
    assert texts == ["first-a", "first-b", "second", "third"]


def test_blocks_without_block_order_keep_their_place_in_the_reading_order():
    """PaddleX numbers only some labels; a figure title (null) still sits where it was sorted."""
    blocks = [
        ("text", (0, 0, 400, 50)),
        ("image", (0, 60, 400, 200)),
        ("figure_title", (0, 210, 400, 240)),
        ("text", (0, 250, 400, 300)),
    ]
    lines = [
        ("Before the figure.", (10, 10, 300, 40)),
        ("Fig. 1: A harbour.", (10, 212, 300, 238)),
        ("After the figure.", (10, 260, 300, 290)),
    ]
    raw = native(blocks, lines, orders=[1, None, None, 2])
    prediction, counts = pps.result_to_prediction(raw, "s1", (900, 420))
    assert prediction["reading_order"] == ["b1", "b2", "b3", "b4"]
    assert prediction["text"] == "Before the figure.\n\nFig. 1: A harbour.\n\nAfter the figure."
    assert [r["extra"]["block_order"] for r in prediction["regions"]] == [1, None, None, 2]
    assert [r["type"] for r in prediction["regions"]] == ["text", "figure", "caption", "text"]
    assert counts["outside_blocks"] == 0
    assert errors(prediction, (900, 420)) == []


def test_a_blank_page_is_an_empty_prediction():
    prediction, counts = pps.result_to_prediction(NATIVE["blank"], "s1", SIZES["blank"])
    assert prediction["text"] == ""
    assert "regions" not in prediction
    assert "reading_order" not in prediction
    assert counts == {"lines": 0, "assigned": 0, "outside_blocks": 0}
    assert errors(prediction, SIZES["blank"]) == []


def test_preprocessing_runs_but_must_not_transform_the_image():
    """The real fixtures report the preprocessor as used, with both steps off."""
    assert NATIVE["page"]["res"]["model_settings"]["use_doc_preprocessor"] is True
    pps.result_to_prediction(NATIVE["page"], "s1", SIZES["page"])
    for step in pps.PREPROCESSING_STEPS:
        changed = json.loads(json.dumps(NATIVE["page"]))
        changed["res"]["doc_preprocessor_res"]["model_settings"][step] = True
        with pytest.raises(pps.AdapterError, match="preprocessing"):
            pps.result_to_prediction(changed, "s1", SIZES["page"])
    unknown = json.loads(json.dumps(NATIVE["page"]))
    del unknown["res"]["doc_preprocessor_res"]
    with pytest.raises(pps.AdapterError, match="preprocessing"):
        pps.result_to_prediction(unknown, "s1", SIZES["page"])


def test_inconsistent_results_are_rejected():
    broken = native([], [("a", (0, 0, 10, 10))])
    broken["res"]["overall_ocr_res"]["rec_scores"] = []
    with pytest.raises(pps.AdapterError, match="differ in length"):
        pps.result_to_prediction(broken, "s1", (900, 420))
    with pytest.raises(pps.AdapterError, match="900x420 image"):
        pps.result_to_prediction(native([], []), "s1", (901, 420))
    blocks = [("text", (0, 0, 10, 10)), ("image", (0, 20, 10, 30)), ("text", (0, 40, 10, 50))]
    with pytest.raises(pps.AdapterError, match="block_order does not increase"):
        pps.result_to_prediction(native(blocks, [], orders=[2, None, 1]), "s1", (900, 420))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lang": "en"}, "unknown parameters: lang"),
        ({"use_table_recognition": True}, "unknown parameters: use_table_recognition"),
        ({"device": "cpu"}, "device"),
        ({"ocr_models": "PP-OCRv5"}, "ocr_models must be 'PP-OCRv5-latin' or 'PP-OCRv6-medium'"),
        ({"ocr_models": None}, "ocr_models"),
        ({"layout_threshold": 0}, "layout_threshold"),
        ({"layout_threshold": 1.5}, "layout_threshold"),
        ({"text_det_limit_type": "both"}, "text_det_limit_type"),
        ({"text_det_limit_side_len": 0}, "text_det_limit_side_len"),
        ({"text_recognition_batch_size": 2.5}, "text_recognition_batch_size"),
        ({"text_det_thresh": "0.3"}, "text_det_thresh"),
        ({"text_det_box_thresh": float("nan")}, "text_det_box_thresh"),
        ({"text_det_unclip_ratio": 0}, "text_det_unclip_ratio"),
        ({"text_rec_score_thresh": 2}, "text_rec_score_thresh"),
        ({"text_rec_score_thresh": True}, "text_rec_score_thresh"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(pps.AdapterError, match=message):
        pps.parse_params(params)


def test_pipeline_uses_pinned_models_with_layout_and_ocr_only():
    params = pps.parse_params({"layout_threshold": 0.4, "text_det_limit_side_len": 1280})
    arguments = pps.pipeline_arguments(params)
    assert arguments["layout_detection_model_name"] == "PP-DocLayout_plus-L"
    assert arguments["region_detection_model_name"] == "PP-DocBlockLayout"
    assert arguments["text_detection_model_name"] == "PP-OCRv5_server_det"
    assert arguments["textline_orientation_model_name"] == "PP-LCNet_x1_0_textline_ori"
    assert arguments["text_recognition_model_name"] == "latin_PP-OCRv5_mobile_rec"
    assert arguments["text_recognition_model_dir"].endswith("/latin_PP-OCRv5_mobile_rec")
    assert "ocr_models" not in arguments
    assert arguments["use_doc_orientation_classify"] is False
    assert arguments["use_doc_unwarping"] is False
    assert arguments["use_textline_orientation"] is True
    assert arguments["use_region_detection"] is True
    for name in ("table", "formula", "seal", "chart"):
        assert arguments[f"use_{name}_recognition"] is False
    assert arguments["device"] == "gpu:0"
    assert arguments["layout_threshold"] == 0.4
    assert arguments["text_det_limit_side_len"] == 1280
    assert "text_det_thresh" not in arguments  # None keeps the pipeline default


def test_ocr_models_select_the_text_detection_and_recognition_models():
    arguments = pps.pipeline_arguments(pps.parse_params({"ocr_models": "PP-OCRv6-medium"}))
    assert arguments["text_detection_model_name"] == "PP-OCRv6_medium_det"
    assert arguments["text_detection_model_dir"].endswith("/PP-OCRv6_medium_det")
    assert arguments["text_recognition_model_name"] == "PP-OCRv6_medium_rec"
    assert arguments["layout_detection_model_name"] == "PP-DocLayout_plus-L"  # unchanged


def test_presets_differ_from_the_default_only_in_the_ocr_models():
    assert CONFIG["params"]["ocr_models"] == "PP-OCRv5-latin"
    assert set(PRESETS) == {"ocrv6"}
    preset = PRESETS["ocrv6"]
    assert preset["name"] == "ppstructurev3-ocrv6"
    assert preset["params"] == CONFIG["params"] | {"ocr_models": "PP-OCRv6-medium"}
    ignored = ("name", "model", "params")
    assert {k: v for k, v in preset.items() if k not in ignored} == {
        k: v for k, v in CONFIG.items() if k not in ignored
    }
    for config in (CONFIG, preset):  # the model field names the commits of the models used
        models = pps.models(pps.parse_params(config["params"]))
        dockerfile = (ENGINE_DIR / "Dockerfile").read_text(encoding="utf-8")
        for name in models.values():
            commit = dockerfile.split(f"{name}@", 1)[1][:8]
            assert commit in config["model"], (name, commit)
        parse_engine_config(config)


def test_model_checksums_cover_every_pinned_model():
    listed = (ENGINE_DIR / "models.sha256").read_text(encoding="utf-8").split()[1::2]
    files = ("inference.json", "inference.pdiparams", "inference.yml")
    names = [*pps.MODELS.values(), *(m for s in pps.OCR_MODELS.values() for m in s.values())]
    assert sorted(listed) == sorted(f"{m}/{f}" for m in names for f in files)
    dockerfile = (ENGINE_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert all(f"{m}@" in dockerfile for m in names)


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    samples = []
    heights = {1500: NEWSPAPER[2][1]}
    for sid, width in widths.items():
        Image.new("L", (width, heights.get(width, 420)), 255).save(data / f"{sid}.png")
        (data / f"{sid}.xml").write_text("<PcGts/>", encoding="utf-8")
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def run_adapter(tmp_path, widths, params=None, env=None):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"])
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {"PYTHONPATH": ":".join(str(path) for path in FAKES), **(env or {})}
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, widths),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in widths}


def test_adapter_follows_the_engine_protocol(tmp_path):
    widths = {"news": 1500, "blank": 300, "boom": 666, "oom": 777, "cols": 1000}
    run, records = run_adapter(tmp_path, widths)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "news": ("success", None),
        "blank": ("success", None),
        "boom": ("failure", "engine-error"),
        "oom": ("failure", "engine-error"),
        "cols": ("success", None),  # the pipeline keeps going after failed samples
    }
    assert "RuntimeError: fake pipeline error" in records["boom"]["message"]
    oom = records["oom"]["message"]
    assert oom.startswith("MemoryError: ResourceExhaustedError: Out of memory error on GPU 0")
    assert "C++ Traceback" not in oom
    news = run.path / "samples" / "news" / "attempt-1"
    outputs = {o["path"] for o in records["news"]["outputs"]}
    assert outputs == {"ppstructurev3.json", "prediction.json", "result.json"}
    assert json.loads((news / "output" / "ppstructurev3.json").read_text()) == NATIVE["newspaper"]
    expected, counts = pps.result_to_prediction(NATIVE["newspaper"], "news", SIZES["newspaper"])
    assert (news / "prediction.txt").read_text(encoding="utf-8") == expected["text"]
    extra = records["news"]["engine_extra"]
    assert extra["environment"]["paddleocr"] == "3.7.0-fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    assert extra["models"]["job_model"] == CONFIG["model"]
    assert extra["models"]["layout_detection"] == "PP-DocLayout_plus-L"
    assert extra["models"]["text_recognition"] == "latin_PP-OCRv5_mobile_rec"
    assert extra["params"]["ocr_models"] == "PP-OCRv5-latin"
    assert extra["params"]["device"] == "gpu:0"
    assert extra["blocks"] == 7
    assert extra["lines"] == counts
    assert extra["model_settings"]["use_table_recognition"] is False
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"device": "gpu:0", "psm": 3}, {}, "unknown parameters: psm"),
        ({"device": "gpu:0"}, {"FAKE_PADDLE_GPUS": "0"}, "0 CUDA devices are visible"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params=params, env=env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def ppstructurev3_image():
    subprocess.run(
        ["docker", "build", "-q", "-t", CONFIG["image"], str(ENGINE_DIR)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=True,
    )
    return CONFIG["image"]


@pytest.mark.docker
@pytest.mark.gpu
def test_gpu_smoke_benchmark(tmp_path, ppstructurev3_image):
    manifest = smoke_dataset_with_columns(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--run-id", "pps"]) == 0
    run = load_run(tmp_path / "runs" / "pps")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == dict.fromkeys(
        ("page-1", "para-1", "blank-1", "columns-1", "news-1"), "success"
    )
    environment = records["page-1"]["engine_extra"]["environment"]
    assert environment["paddleocr"] == "3.7.0"
    assert environment["paddlex"] == "3.7.2"
    assert environment["paddle"] == "3.4.0"
    assert environment["gpu"]
    assert json.loads((run.path / "run.json").read_text())["host"]["gpus"]
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
        assert rates[sid] < 0.1, sid  # ordered CER: the columns are read one after another
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # A GPU that is not visible is an explicit engine failure for every sample.
    bad = tmp_path / "bad.toml"
    bad.write_text(Path(engine).read_text().replace('"gpu:0"', '"gpu:7"'), encoding="utf-8")
    args[3] = str(bad)
    assert main([*args, "--run-id", "bad-gpu"]) == 0
    run = load_run(tmp_path / "runs" / "bad-gpu")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("gpu:7" in r["message"] for r in records)
