"""PaddleOCR-VL-1.6 adapter (engines/paddleocr-vl): result mapping, parameters, models, protocol.

Offline tests import the adapter and run it as a host process with a fake ``paddleocr``
(tests/fixtures/paddleocr-vl/fake) and the fake ``paddle``, ``paddlex`` and ``cv2`` modules of
tests/fixtures/paddleocr/fake. The fake returns committed results: real PaddleOCR-VL-1.6 output
of PaddleOCR 3.7.0 for synthetic pages rendered by tests/synthetic.py (page.json for PAGE,
columns.json for COLUMNS, newspaper.json for NEWSPAPER and blank.json for a blank page), and
calls the VL model's ``generate`` once per block so that truncation can be detected. The opt-in
GPU test builds the engine image and runs the real pipeline through ``ocrbench run``,
``evaluate`` and ``report``.
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
    render,
    render_columns,
    smoke_dataset_with_columns,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "paddleocr-vl"
FIXTURES = Path(__file__).parent / "fixtures" / "paddleocr-vl"
FAKES = [FIXTURES / "fake", Path(__file__).parent / "fixtures" / "paddleocr" / "fake"]
ADAPTER_PATH = ENGINE_DIR / "paddleocr_vl_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
CHECKSUMS = (ENGINE_DIR / "models.sha256").read_text(encoding="utf-8")
NATIVE = {
    name: json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("page", "columns", "newspaper", "blank")
}
SIZES = {"page": (900, 420), "columns": COLUMNS[2], "newspaper": NEWSPAPER[2], "blank": (300, 200)}

_spec = importlib.util.spec_from_file_location("paddleocr_vl_engine", ADAPTER_PATH)
vl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vl)


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


def native(blocks, size=(900, 420)):
    """A minimal PaddleOCR-VL result: blocks ``(label, bbox, content)``."""
    parsing = [
        {"block_label": label, "block_content": content, "block_bbox": list(bbox)}
        | {"block_id": i, "block_order": i + 1, "group_id": i}
        for i, (label, bbox, content) in enumerate(blocks)
    ]
    res = {"width": size[0], "height": size[1], "model_settings": {"use_doc_preprocessor": False}}
    return {"res": res | {"parsing_res_list": parsing}}


@pytest.mark.parametrize("name", ["page", "columns"])
def test_blocks_become_regions_with_the_generated_text(tmp_path, name):
    """The VL model reads each block as one paragraph: its lines joined by spaces."""
    prediction = vl.result_to_prediction(NATIVE[name], "s1", SIZES[name])
    assert errors(prediction, SIZES[name]) == []
    paragraphs = [" ".join(t for t, _ in region) for region in rendered(name, tmp_path)]
    assert [region["text"] for region in prediction["regions"]] == paragraphs
    assert prediction["text"] == "\n\n".join(paragraphs)
    blocks = NATIVE[name]["res"]["parsing_res_list"]
    assert prediction["reading_order"] == [f"b{i + 1}" for i in range(len(blocks))]
    for region, block in zip(prediction["regions"], blocks, strict=True):
        assert region["native_type"] == block["block_label"]
        assert region["polygon"] == block["block_polygon_points"]  # input-image pixels, as is
        assert "lines" not in region  # no line positions are invented
        assert region["extra"] == {
            "block_id": block["block_id"],
            "block_order": block["block_order"],
            "group_id": block["group_id"],
        }


def test_region_polygons_cover_the_rendered_paragraphs(tmp_path):
    prediction = vl.result_to_prediction(NATIVE["columns"], "s1", SIZES["columns"])
    for region, lines in zip(prediction["regions"], rendered("columns", tmp_path), strict=True):
        xs, ys = [x for x, _ in region["polygon"]], [y for _, y in region["polygon"]]
        for _, (x0, y0, x1, y1) in lines:
            assert min(xs) - 8 <= x0 < x1 <= max(xs) + 8
            assert min(ys) - 8 <= y0 < y1 <= max(ys) + 8


def test_newspaper_columns_follow_the_layout_models_order(tmp_path):
    """Recorded real behavior: PP-DocLayoutV3 orders the synthetic newspaper's blocks row by
    row across the columns; the adapter keeps that order and every block's text."""
    prediction = vl.result_to_prediction(NATIVE["newspaper"], "s1", SIZES["newspaper"])
    assert errors(prediction, SIZES["newspaper"]) == []
    regions = rendered("newspaper", tmp_path)
    paragraphs = [" ".join(t for t, _ in region) for region in regions]
    got = [" ".join(region["text"].split()) for region in prediction["regions"]]
    assert sorted(got) == sorted(paragraphs)
    assert got != paragraphs  # not column by column
    assert got[0] == "Evening Gazette"


def test_region_types_follow_the_block_labels():
    labels = ["doc_title", "paragraph_title", "text", "header", "footer", "number", "image"]
    labels += ["figure_title", "footnote", "table", "display_formula", "aside_text", "seal"]
    labels += ["vertical_text", "vision_footnote", "formula_number", "mystery"]
    blocks = [(label, (10 * i, 0, 10 * i + 8, 8), "") for i, label in enumerate(labels)]
    prediction = vl.result_to_prediction(native(blocks), "s1", (900, 420))
    types = [region["type"] for region in prediction["regions"]]
    assert types == [
        *("heading", "heading", "text", "page-header", "page-footer", "page-number", "figure"),
        *("caption", "footnote", "table", "formula", "text", "figure"),
        *("text", "footnote", "other", "other"),
    ]
    assert all("text" not in region for region in prediction["regions"])  # nothing invented
    assert prediction["text"] == ""
    assert errors(prediction, (900, 420)) == []


def test_tables_and_formulas_become_plain_text():
    table = (
        "<table><tr><th>Year</th><th>Ships</th></tr>"
        "<tr><td>1923</td><td>12 &amp; 3</td></tr><tr><td colspan=2>  total <br/>15 </td></tr>"
        "</table>"
    )
    blocks = [
        ("table", (0, 0, 100, 50), table),
        ("display_formula", (0, 60, 100, 80), " $$ E = mc^2 $$ "),
        ("inline_formula", (0, 90, 100, 100), "$x_1$"),
        ("formula_number", (0, 110, 100, 120), "(1)"),
        ("text", (0, 130, 100, 150), " Text with $a+b$ inline. \n"),
        ("table", (0, 160, 100, 170), "not a table"),
        ("image", (0, 180, 100, 200), ""),
    ]
    prediction = vl.result_to_prediction(native(blocks), "s1", (900, 420))
    texts = [region.get("text") for region in prediction["regions"]]
    assert texts == [
        "Year\tShips\n1923\t12 & 3\ntotal 15",
        "E = mc^2",
        "x_1",
        "(1)",
        "Text with $a+b$ inline.",
        "not a table",
        None,
    ]
    assert errors(prediction, (900, 420)) == []


def test_boxes_are_used_without_a_polygon():
    raw = native([("text", (5, 6, 50, 60), "a")])
    raw["res"]["parsing_res_list"][0]["block_polygon_points"] = None
    region = vl.result_to_prediction(raw, "s1", (900, 420))["regions"][0]
    assert region["polygon"] == [[5.0, 6.0], [50.0, 6.0], [50.0, 60.0], [5.0, 60.0]]
    poly = [[5, 6], [50, 6], [60, 40], [50, 60], [5, 60]]
    raw["res"]["parsing_res_list"][0]["block_polygon_points"] = poly
    region = vl.result_to_prediction(raw, "s1", (900, 420))["regions"][0]
    assert region["polygon"] == [[float(x), float(y)] for x, y in poly]


def test_a_blank_page_is_an_empty_prediction():
    prediction = vl.result_to_prediction(NATIVE["blank"], "s1", SIZES["blank"])
    assert prediction["text"] == ""
    assert "regions" not in prediction
    assert "reading_order" not in prediction
    assert errors(prediction, SIZES["blank"]) == []


def test_inconsistent_results_are_rejected():
    with pytest.raises(vl.AdapterError, match="900x420 image"):
        vl.result_to_prediction(native([]), "s1", (901, 420))
    preprocessed = native([])
    preprocessed["res"]["model_settings"]["use_doc_preprocessor"] = True
    with pytest.raises(vl.AdapterError, match="preprocessing"):
        vl.result_to_prediction(preprocessed, "s1", (900, 420))
    assert NATIVE["page"]["res"]["model_settings"]["use_doc_preprocessor"] is False  # real


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lang": "en"}, "unknown parameters: lang"),
        ({"use_layout_detection": False}, "unknown parameters: use_layout_detection"),
        ({"device": "cpu"}, "device"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 2.5}, "max_new_tokens"),
        ({"max_pixels": True}, "max_pixels"),
        ({"min_pixels": 0}, "min_pixels"),
        ({"min_pixels": 2000, "max_pixels": 1000}, "min_pixels must not be larger"),
        ({"layout_threshold": 0}, "layout_threshold"),
        ({"layout_threshold": float("nan")}, "layout_threshold"),
        ({"layout_threshold": "0.3"}, "layout_threshold"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(vl.AdapterError, match=message):
        vl.parse_params(params)


def test_pipeline_uses_the_pinned_models_without_preprocessing():
    params = vl.parse_params({"max_pixels": 2000000, "layout_threshold": 0.4})
    assert params["max_new_tokens"] == 4096
    arguments = vl.pipeline_arguments(params)
    assert arguments == {
        "pipeline_version": "v1.6",
        "device": "gpu:0",
        "layout_detection_model_name": "PP-DocLayoutV3",
        "layout_detection_model_dir": str(vl.CACHE / vl.model_dir("layout_detection")),
        "vl_rec_model_name": "PaddleOCR-VL-1.6-0.9B",
        "vl_rec_model_dir": str(vl.CACHE / vl.model_dir("vl_rec")),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
    }
    assert vl.predict_arguments(params) == {
        "max_new_tokens": 4096,
        "max_pixels": 2000000,
        "layout_threshold": 0.4,
    }  # min_pixels None keeps the pipeline default
    assert vl.parse_params({"max_new_tokens": None})["max_new_tokens"] == 4096


def test_configured_model_names_the_pinned_commits():
    assert CONFIG["params"] == {"device": "gpu:0", "max_new_tokens": 4096}
    for prefix, (_, repository, revision) in vl.MODELS.items():
        assert f"{repository.split('/', 1)[1]}@{revision[:8]}" in CONFIG["model"], prefix
    parse_engine_config(CONFIG)


def test_model_checksums_cover_both_pinned_directories():
    names = CHECKSUMS.split()[1::2]
    directories = {name.rsplit("/", 1)[0] for name in names}
    assert directories == {vl.model_dir(p).removeprefix(f"{vl.MODELS_DIR}/") for p in vl.MODELS}
    assert "PaddleOCR-VL-1.6/c5630abae1d940eafe0697512a0325494b02ab42/model.safetensors" in names
    assert "PP-DocLayoutV3/241f8bdfc77a7c7bee915a5057aaee58c235a8d3/inference.pdiparams" in names
    # Hugging Face's LFS hash of the VL weights at the pinned commit.
    digest = "85a479d506a11e724e7285d395c551be69f41dbc16b6342d3cacfb189aed71db"
    weights = f"{vl.model_dir('vl_rec')[len(vl.MODELS_DIR) + 1 :]}/model.safetensors"
    assert f"{digest}  {weights}" in CHECKSUMS


def model_cache(root, corrupt=False, extra=None):
    """A cache with fake files in both pinned directories and checksums that match them;
    ``extra`` adds a file that is not in the checksums."""
    lines = []
    for prefix in vl.MODELS:
        directory = root / "cache" / vl.model_dir(prefix)
        directory.mkdir(parents=True)
        name = "model.safetensors" if prefix == "vl_rec" else "inference.pdiparams"
        (directory / name).write_text(f"fake {prefix}", encoding="utf-8")
        relative = (directory / name).relative_to(root / "cache" / vl.MODELS_DIR).as_posix()
        lines.append(f"{hashlib.sha256(f'fake {prefix}'.encode()).hexdigest()}  {relative}")
        if corrupt:
            (directory / name).write_text("changed", encoding="utf-8")
    if extra:
        (root / "cache" / vl.model_dir("vl_rec") / extra).write_text("{}", encoding="utf-8")
    checksums = root / "models.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_PADDLEOCR_VL_CACHE": str(root / "cache"),
        "OCRBENCH_PADDLEOCR_VL_CHECKSUMS": str(checksums),
    }


def test_model_files_must_match_the_checksums_exactly(tmp_path, monkeypatch):
    env = model_cache(tmp_path)
    monkeypatch.setattr(vl, "CACHE", Path(env["OCRBENCH_PADDLEOCR_VL_CACHE"]))
    monkeypatch.setattr(vl, "CHECKSUMS", Path(env["OCRBENCH_PADDLEOCR_VL_CHECKSUMS"]))
    files = vl.check_models()
    layout, weights = (vl.model_dir(p)[len(vl.MODELS_DIR) + 1 :] for p in vl.MODELS)
    assert sorted(files) == [f"{layout}/inference.pdiparams", f"{weights}/model.safetensors"]
    (vl.CACHE / vl.model_dir("vl_rec") / "generation_config.json").write_text("{}")
    with pytest.raises(vl.AdapterError, match="not in models.sha256.*generation_config.json"):
        vl.check_models()
    (vl.CACHE / vl.model_dir("vl_rec") / "generation_config.json").unlink()
    (vl.CACHE / vl.model_dir("vl_rec") / "model.safetensors").write_text("changed")
    with pytest.raises(vl.AdapterError, match="not matching models.sha256.*model.safetensors"):
        vl.check_models()
    monkeypatch.setattr(vl, "CACHE", tmp_path / "nothing")
    with pytest.raises(vl.AdapterError, match="make paddleocr-vl-model"):
        vl.check_models()


def test_generations_record_tokens_and_cut_off_blocks():
    class Ids(list):
        def tolist(self):
            return [list(row) for row in self]

    class Model:
        def __init__(self):
            self.outputs = [Ids([[5, 6, 2, 0]]), Ids([[5, 6, 7]]), Ids([[2]])]

        def generate(self, inputs, **kwargs):
            assert kwargs == {"max_new_tokens": 3}
            return self.outputs.pop(0), None

    model = Model()
    generations = vl.Generations(model, eos_token_id=2)
    for _ in range(3):
        model.generate({}, max_new_tokens=3)
    assert generations.calls == [(2, True), (3, False), (0, True)]
    assert generations.summary(3) == {
        "vl_calls": 3,
        "generated_tokens": 5,
        "max_generated_tokens": 3,
        "truncated_blocks": 1,
        "max_new_tokens": 3,
    }


def test_fetch_model_keeps_verified_files(tmp_path, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "fetch_model", raising=False)  # each engine has its own
    sys.path.insert(0, str(ENGINE_DIR))
    try:
        import fetch_model
    finally:
        sys.path.remove(str(ENGINE_DIR))
    env = model_cache(tmp_path)
    monkeypatch.setattr(fetch_model, "CHECKSUMS", Path(env["OCRBENCH_PADDLEOCR_VL_CHECKSUMS"]))

    def no_download(url, path):
        raise AssertionError(f"downloaded {url}")

    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", no_download)
    fetch_model.main(tmp_path / "cache")
    assert capsys.readouterr().out.count("ok      ") == 2

    urls = []

    def wrong_download(url, path):
        urls.append(url)
        Path(path).write_text("wrong", encoding="utf-8")

    weights = tmp_path / "cache" / vl.model_dir("vl_rec") / "model.safetensors"
    weights.unlink()
    monkeypatch.setattr(fetch_model.urllib.request, "urlretrieve", wrong_download)
    with pytest.raises(SystemExit, match="model.safetensors: SHA-256 differs"):
        fetch_model.main(tmp_path / "cache")
    _, repository, revision = vl.MODELS["vl_rec"]
    assert urls == [f"https://huggingface.co/{repository}/resolve/{revision}/model.safetensors"]
    assert list(weights.parent.iterdir()) == []  # the wrong download is not kept


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    samples = []
    heights = {1500: NEWSPAPER[2][1], 1000: COLUMNS[2][1]}
    for sid, width in widths.items():
        Image.new("L", (width, heights.get(width, 420)), 255).save(data / f"{sid}.png")
        (data / f"{sid}.xml").write_text("<PcGts/>", encoding="utf-8")
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def run_adapter(tmp_path, widths, params=None, env=None, cache=True):
    config = dict(CONFIG, params=params if params is not None else CONFIG["params"])
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    paths = model_cache(tmp_path) if cache else {"OCRBENCH_PADDLEOCR_VL_CACHE": str(tmp_path)}
    paths.setdefault("OCRBENCH_PADDLEOCR_VL_CHECKSUMS", str(ENGINE_DIR / "models.sha256"))
    env = {"PYTHONPATH": ":".join(str(path) for path in FAKES), **paths, **(env or {})}
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, widths),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in widths}


def test_adapter_follows_the_engine_protocol(tmp_path):
    widths = {"news": 1500, "blank": 300, "boom": 666, "oom": 777, "page": 900}
    run, records = run_adapter(tmp_path, widths)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "news": ("success", None),
        "blank": ("success", None),
        "boom": ("failure", "engine-error"),
        "oom": ("failure", "engine-error"),
        "page": ("success", None),  # the pipeline keeps going after failed samples
    }
    assert "RuntimeError: fake pipeline error" in records["boom"]["message"]
    oom = records["oom"]["message"]
    assert oom.startswith("MemoryError: ResourceExhaustedError: Out of memory error on GPU 0")
    assert "C++ Traceback" not in oom
    news = run.path / "samples" / "news" / "attempt-1"
    outputs = {o["path"] for o in records["news"]["outputs"]}
    assert outputs == {"paddleocr_vl.json", "prediction.json", "result.json"}
    assert json.loads((news / "output" / "paddleocr_vl.json").read_text()) == NATIVE["newspaper"]
    expected = vl.result_to_prediction(NATIVE["newspaper"], "news", SIZES["newspaper"])
    assert (news / "prediction.txt").read_text(encoding="utf-8") == expected["text"]
    extra = records["news"]["engine_extra"]
    assert extra["environment"]["paddleocr"] == "3.7.0-fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    models = extra["models"]
    assert models["job_model"] == CONFIG["model"]
    assert models["pipeline_version"] == "v1.6"
    name, repository, revision = vl.MODELS["vl_rec"]
    assert models["vl_rec"] == {"name": name, "repository": repository, "revision": revision}
    assert len(models["files"]) == 2
    assert extra["params"] == vl.parse_params(CONFIG["params"])
    assert extra["blocks"] == 7
    assert extra["vl_calls"] == 7
    assert extra["truncated_blocks"] == 0
    assert extra["max_new_tokens"] == 4096
    assert extra["generated_tokens"] == len(expected["text"].split())  # one per fake word
    assert extra["model_settings"]["merge_layout_blocks"] is True
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""
    assert records["blank"]["engine_extra"]["vl_calls"] == 0  # a genuine empty result


def test_blocks_cut_off_at_max_new_tokens_make_the_sample_truncated(tmp_path):
    params = CONFIG["params"] | {"max_new_tokens": 4}
    run, records = run_adapter(tmp_path, {"page": 900, "blank": 300}, params=params)
    page = records["page"]
    assert (page["status"], page["engine_status"]) == ("truncated", "truncated")
    assert page["message"] == "2 of 3 blocks stopped at max_new_tokens (4 tokens)"
    assert page["engine_extra"]["truncated_blocks"] == 2
    text = (run.path / "samples" / "page" / "attempt-1" / "prediction.txt").read_text()
    assert text == "The Daily Chronicle\n\nThe quick brown fox\n\nA second paragraph and"
    assert records["blank"]["engine_status"] == "ok"


@pytest.mark.parametrize(
    ("params", "env", "cache", "message"),
    [
        ({"device": "gpu:0", "psm": 3}, {}, True, "unknown parameters: psm"),
        ({"device": "gpu:0"}, {"FAKE_PADDLE_GPUS": "0"}, True, "0 CUDA devices are visible"),
        ({"device": "gpu:0"}, {}, False, "make paddleocr-vl-model and run with --cache"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, cache, message):
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params=params, env=env, cache=cache)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def paddleocr_vl_cache():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    weights = cache / vl.model_dir("vl_rec") / "model.safetensors"
    if not weights.is_file():
        pytest.fail(f"no weights at {weights}: run make paddleocr-vl-model first")
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
def test_gpu_smoke_benchmark(tmp_path, paddleocr_vl_cache):
    cache = str(paddleocr_vl_cache)
    manifest = smoke_dataset_with_columns(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "vl"]) == 0
    run = load_run(tmp_path / "runs" / "vl")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == dict.fromkeys(
        ("page-1", "para-1", "blank-1", "columns-1", "news-1"), "success"
    )
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"]["paddleocr"] == "3.7.0"
    assert extra["environment"]["paddlex"] == "3.7.2"
    assert extra["environment"]["paddle"] == "3.4.0"
    assert extra["environment"]["gpu"]
    assert extra["truncated_blocks"] == 0
    assert extra["vl_calls"] == extra["blocks"] == 3
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
    for sid in ("page-1", "para-1", "columns-1"):
        assert rates[sid] < 0.1, sid
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # A small max_new_tokens cuts blocks off; the sample is truncated and scored as produced.
    limited = tmp_path / "engine.toml"
    config = Path(engine).read_text(encoding="utf-8")
    limited.write_text(config.replace("max_new_tokens = 4096", "max_new_tokens = 8"), "utf-8")
    short = [*args[:2], "--engine", str(limited), *args[4:]]
    assert main([*short, "--cache", cache, "--run-id", "truncated"]) == 0
    short_run = load_run(tmp_path / "runs" / "truncated")
    page = latest_record(short_run, "page-1")
    assert page["status"] == "truncated"
    assert page["engine_extra"]["truncated_blocks"] >= 1
    assert page["engine_extra"]["max_generated_tokens"] == 8
    assert main(["evaluate", str(short_run.path)]) == 0
    scores = json.loads((short_run.path / "evaluations" / "1" / "scores.json").read_text())
    short_rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < short_rates["page-1"] < 1  # scored as produced, not as empty

    # Without the cache mount, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    run = load_run(tmp_path / "runs" / "no-cache")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("make paddleocr-vl-model" in r["message"] for r in records)
