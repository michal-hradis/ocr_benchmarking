"""OCR-D/Calamari adapter (engines/ocrd-calamari): PAGE XML mapping, parameters and the protocol.

Offline tests import the adapter and run it as a host process with fake OCR-D, processor and
``tesserocr`` modules (tests/fixtures/ocrd-calamari/fake) that return committed pipeline output
(tests/fixtures/ocrd-calamari/page.xml, newspaper.xml and blank.xml: real output of the engine
image with the default gt4histocr preset for the synthetic page-1, news-1 and blank-1 pages). The
opt-in Docker test builds the engine image and runs the real pipeline on the CPU through
``ocrbench run``, ``evaluate`` and ``report``.
"""

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.issues import Severity
from ocrbench.pagexml import read_page_xml
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import smoke_dataset, smoke_dataset_with_columns, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "ocrd-calamari"
FIXTURES = Path(__file__).parent / "fixtures" / "ocrd-calamari"
PAGE_XML = (FIXTURES / "page.xml").read_text(encoding="utf-8")
NEWSPAPER_XML = (FIXTURES / "newspaper.xml").read_text(encoding="utf-8")
ADAPTER_PATH = ENGINE_DIR / "ocrd_calamari_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
PRESETS = {
    path.stem: tomllib.loads(path.read_text(encoding="utf-8"))
    for path in sorted((ENGINE_DIR / "presets").glob("*.toml"))
}

_spec = importlib.util.spec_from_file_location("ocrd_calamari_engine", ADAPTER_PATH)
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)

# The committed output for the synthetic page-1 (tests/synthetic.py PAGE): Calamari's historical
# model reads the modern sans-serif font imperfectly, including a long s.
FIXTURE_REGIONS = [
    ["The Daily Chronicle"],
    ["The quic bron fox ſumps over", "the lazy dggnnaa the rier."],
    ["A Second paragraph and more text", "vith numbers 1923 and 2026."],
]
PAGE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def errors(prediction, size=(900, 420)):
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=size)
    assert standardized is not None, [i.format() for i in issues]
    assert standardized.text == prediction["text"]  # page text agrees with the structure
    return [i for i in issues if i.severity is not Severity.INFO]


def page_xml(body, reading_order=""):
    return (
        f'<PcGts xmlns="{PAGE}"><Page imageFilename="x.png" imageWidth="900" imageHeight="420">'
        f"{reading_order}{body}</Page></PcGts>"
    )


def region(rid, lines, kind=None, coords="0,0 9,0 9,9", tag="TextRegion", inner=""):
    attributes = f' type="{kind}"' if kind else ""
    body = "".join(
        f'<TextLine id="{rid}-{n}"><Coords points="{coords}"/>'
        f'<TextEquiv conf="0.5"><Unicode>{text}</Unicode></TextEquiv></TextLine>'
        for n, text in enumerate(lines)
    )
    return f'<{tag} id="{rid}"{attributes}><Coords points="{coords}"/>{body}{inner}</{tag}>'


def test_page_xml_keeps_regions_lines_and_geometry():
    prediction = oc.page_xml_to_prediction(PAGE_XML, "s1")
    assert errors(prediction) == []
    assert prediction["image"] == {"width": 900, "height": 420}
    assert prediction["text"] == "\n\n".join("\n".join(r) for r in FIXTURE_REGIONS)
    regions = prediction["regions"]
    assert [[line["text"] for line in r["lines"]] for r in regions] == FIXTURE_REGIONS
    assert [r["extra"]["ocrd_id"] for r in regions] == ["region0000", "region0001", "region0002"]
    assert [r["id"] for r in regions] == prediction["reading_order"] == ["r1", "r2", "r3"]
    assert {r["type"] for r in regions} == {"text"}
    assert all("native_type" not in r for r in regions)  # Tesseract's blocks here have no type
    assert regions[1]["polygon"] == [[37, 113], [448, 113], [448, 187], [37, 187]]
    lines = [line for r in regions for line in r["lines"]]
    assert [line["id"] for line in lines] == ["l1", "l2", "l3", "l4", "l5"]
    assert lines[1]["polygon"] == [[37, 113], [448, 113], [448, 147], [37, 147]]
    assert lines[1]["extra"] == {"ocrd_id": "region0001_line0000"}
    assert all(0 < line["confidence"] <= 1 for line in lines)
    assert all("baseline" not in line for line in lines)  # Tesseract's segmentation has none


def test_newspaper_columns_follow_the_page_reading_order():
    prediction = oc.page_xml_to_prediction(NEWSPAPER_XML, "s1")
    assert errors(prediction, (1500, 620)) == []
    regions = prediction["regions"]
    assert len(regions) == 7  # the heading and two paragraphs in each of three columns
    # Column by column, top to bottom within a column: the left edge never decreases, and the
    # top resets only when the next column starts.
    lefts = [r["polygon"][0][0] for r in regions[1:]]
    tops = [r["polygon"][0][1] for r in regions[1:]]
    assert lefts == sorted(lefts)
    assert [tops[i] < tops[i + 1] for i in range(0, 6, 2)] == [True] * 3
    assert regions[1]["lines"][0]["text"] == "Rain fell over the city"
    assert regions[-1]["lines"][-1]["text"].startswith("until late")


def test_native_page_xml_is_readable_by_the_benchmark():
    page, issues = read_page_xml(FIXTURES / "page.xml")
    assert page is not None, [i.format() for i in issues]
    assert all(i.severity is Severity.INFO for i in issues)
    assert (page.width, page.height, page.image_filename) == (900, 420, "page-1.png")
    prediction = oc.page_xml_to_prediction(PAGE_XML, "s1")
    lines = [line for r in prediction["regions"] for line in r["lines"]]
    assert [line.text for line in page.reference_lines] == [line["text"] for line in lines]


def test_tables_types_order_and_omissions():
    cells = region("t_cell0001", ["cell b"]) + region("t_cell0000", ["cell a"])
    body = "".join(
        [
            region("a", ["first a", " "]),  # the blank line is dropped
            region("b", ["heading"], kind="heading"),
            region("t", [], tag="TableRegion", inner=cells),
            region("c", [" "]),  # no text: only in the native output
            region("d", ["pull-out"], kind="floating", coords="0,0 9,9"),  # too few points
            region("e", ["not in the reading order"], kind="caption"),
        ]
    )
    order = (
        '<ReadingOrder><OrderedGroup id="ro">'
        '<RegionRefIndexed index="0" regionRef="b"/>'
        '<OrderedGroupIndexed id="t_order" index="1" regionRef="t">'
        '<RegionRefIndexed index="0" regionRef="t_cell0000"/>'
        '<RegionRefIndexed index="1" regionRef="t_cell0001"/></OrderedGroupIndexed>'
        '<RegionRefIndexed index="3" regionRef="a"/><RegionRefIndexed index="2" regionRef="d"/>'
        '<RegionRefIndexed index="4" regionRef="c"/></OrderedGroup></ReadingOrder>'
    )
    prediction = oc.page_xml_to_prediction(page_xml(body, order), "s1")
    assert errors(prediction) == []
    got = [(r["type"], r.get("native_type"), r["extra"]["ocrd_id"]) for r in prediction["regions"]]
    assert got == [
        ("heading", "heading", "b"),
        ("text", None, "t_cell0000"),  # table cells in the table's own order
        ("text", None, "t_cell0001"),
        ("other", "floating", "d"),
        ("text", None, "a"),
        ("caption", "caption", "e"),  # unreferenced: after the reading order
    ]
    assert prediction["text"] == "heading\n\ncell a\n\ncell b\n\npull-out\n\nfirst a\n\n" + (
        "not in the reading order"
    )
    cell = prediction["regions"][1]
    assert cell["extra"]["parent"] == {"type": "TableRegion", "ocrd_id": "t"}
    assert "parent" not in prediction["regions"][0]["extra"]
    pull_out = prediction["regions"][3]
    assert "polygon" not in pull_out
    assert "polygon" not in pull_out["lines"][0]
    assert pull_out["lines"][0]["confidence"] == 0.5


def test_without_reading_order_regions_keep_document_order():
    body = region("b", ["two"]) + region("a", ["one"])
    prediction = oc.page_xml_to_prediction(page_xml(body), "s1")
    assert prediction["text"] == "two\n\none"
    assert prediction["reading_order"] == ["r1", "r2"]


def test_empty_and_malformed_page_xml():
    blank = (FIXTURES / "blank.xml").read_text(encoding="utf-8")
    empty = oc.page_xml_to_prediction(blank, "s1")
    assert empty["text"] == ""
    assert empty["image"] == {"width": 300, "height": 200}
    assert "regions" not in empty
    assert "reading_order" not in empty
    with pytest.raises(oc.AdapterError, match="without a Page"):
        oc.page_xml_to_prediction(f'<PcGts xmlns="{PAGE}"/>', "s1")
    with pytest.raises(oc.AdapterError, match="without a valid image size"):
        oc.page_xml_to_prediction(f'<PcGts xmlns="{PAGE}"><Page imageWidth="9"/></PcGts>', "s1")


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lang": "ces"}, "unknown parameters: lang"),
        ({"model": "qurator-gt4histocr-1.0"}, "model must be one of gt4histocr, "),
        ({"model": "../gt4histocr"}, "model must be one of"),
        ({"model": 3}, "model must be one of"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(oc.AdapterError, match=message):
        oc.parse_params(params)


def test_params_defaults_and_presets():
    assert oc.parse_params({"model": None}) == oc.DEFAULT_PARAMS
    assert CONFIG["params"] == oc.DEFAULT_PARAMS  # engine.toml spells out the gt4histocr preset
    configs = {"engine": CONFIG, **PRESETS}
    assert sorted(config["params"]["model"] for config in configs.values()) == sorted(oc.MODELS)
    for name, config in configs.items():
        parse_engine_config(config)
        assert config["model"] == f"calamari_models@2.2/{config['params']['model']}", name
        # The presets differ from engine.toml only in their name, model and model parameter.
        same = {k: v for k, v in config.items() if k not in {"name", "model", "params"}}
        assert same == {k: v for k, v in CONFIG.items() if k not in {"name", "model", "params"}}
    assert {config["name"] for config in configs.values()} == {
        "ocrd-calamari",
        "ocrd-calamari-fraktur19",
        "ocrd-calamari-antiqua",
    }


def test_image_downloads_exactly_the_checksummed_models():
    archives = [line.split()[1] for line in (ENGINE_DIR / "models.sha256").read_text().splitlines()]
    assert archives == [f"{model}.tar.gz" for model in oc.MODELS]
    dockerfile = (ENGINE_DIR / "Dockerfile").read_text(encoding="utf-8")
    downloads = re.search(r"for m in \(([^)]*)\)", dockerfile)[1]
    assert re.findall(r'"([^"]+)"', downloads) == list(oc.MODELS)


FOLD_FILES = ("{}.ckpt.json", "{}.ckpt/saved_model.pb", "{}.ckpt/variables/variables.index")


def make_models(root, models=oc.MODELS):
    """Placeholder ensembles with their file checksums; returns the adapter environment.

    The adapter reads the checksums next to itself, so a copy of it is placed in ``root/adapter``
    with the committed models.sha256 and a models.files.sha256 for the placeholders.
    """
    lines = []
    for model in models:
        for fold in range(5):
            for name in FOLD_FILES:
                path = root / "models" / model / name.format(fold)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{model} {path.name} {fold}", encoding="utf-8")
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append(f"{digest}  {path.relative_to(root / 'models').as_posix()}\n")
    adapter = root / "adapter"
    adapter.mkdir()
    shutil.copy(ADAPTER_PATH, adapter)
    shutil.copy(ENGINE_DIR / "models.sha256", adapter)
    (adapter / "models.files.sha256").write_text("".join(lines), encoding="utf-8")
    return {
        "OCRBENCH_OCRD_CALAMARI_MODELS": str(root / "models"),
        "OCRBENCH_OCRD_CALAMARI_WORK": str(root / "work"),
    }


def test_committed_file_checksums_cover_every_fold():
    files = (ENGINE_DIR / "models.files.sha256").read_text(encoding="utf-8").splitlines()
    names = [line.split(maxsplit=1)[1] for line in files]
    weights = ("variables/variables.data-00000-of-00001",)
    assert names == sorted(
        f"{model}/{name.format(fold)}"
        for model in oc.MODELS
        for fold in range(5)
        for name in (*FOLD_FILES, *(f"{{}}.ckpt/{w}" for w in weights))
    )
    assert oc.FILE_CHECKSUMS == ENGINE_DIR / "models.files.sha256"  # beside the adapter
    assert oc.CHECKSUMS == ENGINE_DIR / "models.sha256"


@pytest.fixture
def models(tmp_path, monkeypatch):
    """Placeholder gt4histocr ensemble with matching checksums, for ``oc.model_info``."""
    env = make_models(tmp_path, ["gt4histocr"])
    monkeypatch.setattr(oc, "MODELS_DIR", Path(env["OCRBENCH_OCRD_CALAMARI_MODELS"]))
    monkeypatch.setattr(oc, "FILE_CHECKSUMS", tmp_path / "adapter" / "models.files.sha256")
    return oc.MODELS_DIR / "gt4histocr"


def test_model_info_records_verified_checkpoints_and_archive(models):
    info = oc.model_info("gt4histocr")
    assert info["checkpoints"] == [f"{fold}.ckpt.json" for fold in range(5)]
    assert info["archive_sha256"] == (
        "d0b063fbaa1ca579a2873f1221bfe16c06f2e7c72ddf1d5997fdad3cd4bc5047"
    )
    assert info["files_verified"] == 15
    with pytest.raises(oc.AdapterError, match="no checksums for Calamari model antiqua"):
        oc.model_info("antiqua_historical")


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda d: (d / "4.ckpt.json").unlink(), "missing gt4histocr/4.ckpt.json"),
        (
            lambda d: shutil.rmtree(d / "2.ckpt"),
            "missing gt4histocr/2.ckpt/saved_model.pb, missing gt4histocr/2.ckpt/variables/",
        ),
        (lambda d: (d / "5.ckpt.json").write_text("{}"), "unexpected gt4histocr/5.ckpt.json"),
        (
            lambda d: (d / "0.ckpt" / "saved_model.pb").write_bytes(b"other weights"),
            "modified gt4histocr/0.ckpt/saved_model.pb",
        ),
    ],
)
def test_model_info_rejects_an_incomplete_or_modified_ensemble(models, change, problem):
    change(models)
    with pytest.raises(oc.AdapterError, match="differs from models.files.sha256") as error:
        oc.model_info("gt4histocr")
    assert problem in str(error.value)


# Image width -> size: the fake pipeline returns the committed output of that synthetic page.
HEIGHTS = {900: 420, 1500: 620, 300: 200}


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    samples = []
    for sid, width in widths.items():
        if width is None:
            (data / f"{sid}.png").write_bytes(b"not an image")
        else:
            Image.new("L", (width, HEIGHTS.get(width, 420)), 255).save(data / f"{sid}.png")
        (data / f"{sid}.xml").write_text("<PcGts/>", encoding="utf-8")
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def run_adapter(tmp_path, widths, params=None, env=None):
    config = dict(CONFIG, params=params or CONFIG["params"])
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {"PYTHONPATH": str(FIXTURES / "fake"), **make_models(tmp_path), **(env or {})}
    adapter = tmp_path / "adapter" / ADAPTER_PATH.name
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, widths),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(adapter)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in widths}


def test_adapter_follows_the_engine_protocol(tmp_path):
    widths = {
        "page": 900,
        "blank": 300,
        "boom": 666,
        "no-page": 555,
        "broken": None,
        "news": 1500,
    }
    run, records = run_adapter(tmp_path, widths)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # nothing found: a genuine empty result
        "boom": ("failure", "engine-error"),
        "no-page": ("failure", "engine-error"),
        "broken": ("failure", "input-error"),  # the runner rejects it before the engine
        "news": ("success", None),  # the adapter keeps going after failed samples
    }
    assert records["boom"]["message"] == "RuntimeError: fake segmentation error"
    assert records["no-page"]["message"] == "expected one PAGE file in OCR-D-OCR, found 0"
    page = run.path / "samples" / "page" / "attempt-1"
    outputs = {o["path"] for o in records["page"]["outputs"]}
    assert outputs == {"page.xml", "prediction.json", "result.json"}
    assert (page / "output" / "page.xml").read_text(encoding="utf-8") == PAGE_XML
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n\n".join(
        "\n".join(r) for r in FIXTURE_REGIONS
    )
    extra = records["page"]["engine_extra"]
    assert (extra["regions"], extra["lines"]) == (3, 5)
    assert set(extra["step_seconds"]) == {"binarize", "segment", "recognize"}
    environment = extra["environment"]
    assert set(environment) == {*oc.PACKAGES, "ocrd_calamari_commit", "tesseract", "device"}
    assert (environment["tesseract"], environment["device"]) == ("5.5.3", "cpu")
    calamari = extra["models"]["calamari"]
    assert calamari["name"] == "gt4histocr"
    assert calamari["path"].endswith("/models/gt4histocr")
    assert len(calamari["checkpoints"]) == 5
    assert len(calamari["archive_sha256"]) == 64
    assert calamari["files_verified"] == 15
    assert extra["models"]["job_model"] == CONFIG["model"]
    assert extra["params"] == CONFIG["params"]
    assert records["news"]["engine_extra"]["regions"] == 7
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""
    assert not any((tmp_path / "work").iterdir())  # every workspace was removed


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({"model": "fraktur"}, {}, "model must be one of"),
        ({}, {"OCRBENCH_OCRD_CALAMARI_MODELS": "/nonexistent"}, "no Calamari model directory"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    params = CONFIG["params"] | params
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params=params, env=env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


def test_overridden_model_directory_with_other_weights_fails_every_sample(tmp_path):
    custom = tmp_path / "custom" / "gt4histocr"
    custom.mkdir(parents=True)
    (custom / "0.ckpt.json").write_text("{}", encoding="utf-8")
    env = {"OCRBENCH_OCRD_CALAMARI_MODELS": str(custom.parent)}
    _, records = run_adapter(tmp_path, {"a": 900}, env=env)
    record = records["a"]
    assert (record["status"], record["reason"]) == ("failure", "engine-error")
    assert "differs from models.files.sha256: missing gt4histocr/0.ckpt/" in record["message"]
    assert record["engine_extra"]["models"]["calamari"] is None  # nothing attributed


# Opt-in Docker smoke benchmark (CPU) -----------------------------------------------------------


@pytest.fixture(scope="module")
def ocrd_calamari_image():
    subprocess.run(
        ["docker", "build", "-q", "-t", CONFIG["image"], str(ENGINE_DIR)],
        capture_output=True,
        text=True,
        timeout=3600,
        check=True,
    )
    return CONFIG["image"]


def run_engine(engine, manifest, runs, run_id):
    args = ["run", "start", "--engine", str(engine), "--dataset", str(manifest)]
    assert main([*args, "--runs-dir", str(runs), "--run-id", run_id]) == 0
    run = load_run(runs / run_id)
    return run, {s.id: latest_record(run, s.id) for s in run.samples}


@pytest.mark.docker
def test_docker_smoke_benchmark(tmp_path, ocrd_calamari_image):
    manifest = smoke_dataset_with_columns(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    runs = tmp_path / "runs"
    run, records = run_engine(ENGINE_DIR / "engine.toml", manifest, runs, "ocrd")
    assert {sid: r["status"] for sid, r in records.items()} == dict.fromkeys(
        ("page-1", "para-1", "blank-1", "columns-1", "news-1"), "success"
    )
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"] == {
        "ocrd": "3.13.3",
        "ocrd-cis": "0.2.0",
        "ocrd-tesserocr": "0.21.2",
        "ocrd-calamari": "2.1.0",
        "calamari-ocr": "2.3.1",
        "tensorflow": "2.15.1",
        "tesserocr": "2.12.0",
        "ocrd_calamari_commit": "4d66fa1b809abe124a8e632a0eda6a2c133d9f2d",
        "tesseract": "5.5.3",
        "device": "cpu",
    }
    assert extra["models"]["calamari"]["checkpoints"] == [f"{n}.ckpt.json" for n in range(5)]
    assert extra["models"]["calamari"]["files_verified"] == 20
    for sid in ("page-1", "news-1"):
        sample = run.path / "samples" / sid / "attempt-1"
        prediction = json.loads((sample / "prediction.json").read_text())
        assert prediction["reading_order"]
        assert all("polygon" in region for region in prediction["regions"])
        native = (sample / "output" / "page.xml").read_text(encoding="utf-8")
        assert re.search(r'<pc:Label value="300(\.0)?" type="dpi"/>', native)  # binarization
        page, _ = read_page_xml(sample / "output" / "page.xml")
        lines = [line["text"] for region in prediction["regions"] for line in region["lines"]]
        assert [line.text for line in page.reference_lines] == lines
    assert records["news-1"]["engine_extra"]["regions"] == 7
    blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
    assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure

    # Scoring and re-scoring read only stored outputs: no container runs.
    for _ in range(2):
        assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
    rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    # The historical model on a modern sans-serif font. Tesseract merges the narrow-gutter
    # two-column page into one block read across both columns, so columns-1 is not checked.
    for sid in ("page-1", "para-1", "news-1"):
        assert rates[sid] < 0.15, sid
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # Every preset loads its own ensemble.
    for name, preset in PRESETS.items():
        path = ENGINE_DIR / "presets" / f"{name}.toml"
        _, records = run_engine(path, smoke_dataset(tmp_path / name), runs, name)
        assert {r["status"] for r in records.values()} == {"success"}
        calamari = records["page-1"]["engine_extra"]["models"]["calamari"]
        assert calamari["name"] == preset["params"]["model"]
