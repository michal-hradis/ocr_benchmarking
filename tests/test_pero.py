"""PERO OCR adapter (engines/pero): PAGE XML mapping, bundle checks and the engine protocol.

Offline tests import the adapter and run it as a host process with fake ``torch``, ``cv2`` and
``pero_ocr`` modules (tests/fixtures/pero/fake) that return a committed result
(tests/fixtures/pero/page.xml and alto.xml, real pero-ocr output with the universal preset for a
synthetic 900x420 page) and with synthetic bundles in a temporary cache. The opt-in GPU test
builds the engine image and runs real PERO OCR with the user-supplied bundles through
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
from ocrbench.pagexml import read_page_xml
from ocrbench.prediction import validate_prediction
from ocrbench.runner import ProcessLauncher, create_run, execute, latest_record, load_run
from synthetic import smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "pero"
FIXTURES = Path(__file__).parent / "fixtures" / "pero"
PAGE_XML = (FIXTURES / "page.xml").read_text(encoding="utf-8")
ADAPTER_PATH = ENGINE_DIR / "pero_engine.py"
CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))

_spec = importlib.util.spec_from_file_location("pero_engine", ADAPTER_PATH)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)

FIXTURE_REGIONS = [
    ["The Daily Chronicle"],
    ["The quick brown fox jumps over", "the lazy dog near the river."],
    ["A second paragraph and more text", "with numbers 1923 and 2026."],
]
PAGE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
# The structure of the supplied universal preset's INIs (README.md), with placeholder files.
LAYOUT_INI = """[PAGE_PARSER]
RUN_LAYOUT_PARSER = yes
RUN_LINE_CROPPER = no
RUN_OCR = no
RUN_DECODER = no

[LAYOUT_PARSER_1]
METHOD = LAYOUT_CNN
MODEL_PATH = ./ParseNet.pt
USE_CPU = no

[LAYOUT_PARSER_2]
METHOD = REGION_SORTER_SMART
"""
OCR_INI = """[PAGE_PARSER]
RUN_LAYOUT_PARSER = no
RUN_LINE_CROPPER = yes
RUN_OCR = yes
RUN_DECODER = no

[LINE_CROPPER]
LINE_HEIGHT = 48

[OCR]
METHOD = pytorch_ocr-transformer
OCR_JSON = ./ocr_engine.json
"""
BUNDLES = {
    "layout": {"config.ini": LAYOUT_INI, "ParseNet.pt": "layout model"},
    "ocr": {
        "config.ini": OCR_INI,
        "ocr_engine.json": json.dumps({"checkpoint": "./checkpoint.pt", "exported": True}),
        "checkpoint.pt": "ocr model",
    },
}


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


def region(rid, lines, kind=None, coords="0,0 9,0 9,9"):
    attributes = f' type="{kind}"' if kind else ""
    body = "".join(
        f'<TextLine id="{rid}-{n}"><Coords points="{coords}"/><Baseline points="0,8 9,8"/>'
        f'<TextEquiv conf="0.5"><Unicode>{text}</Unicode></TextEquiv></TextLine>'
        for n, text in enumerate(lines)
    )
    return f'<TextRegion id="{rid}"{attributes}><Coords points="{coords}"/>{body}</TextRegion>'


def test_page_xml_keeps_pero_regions_lines_and_geometry():
    prediction = pe.page_xml_to_prediction(PAGE_XML, "s1", (900, 420))
    assert errors(prediction) == []
    assert prediction["image"] == {"width": 900, "height": 420}
    assert prediction["text"] == "\n\n".join("\n".join(r) for r in FIXTURE_REGIONS)
    regions = prediction["regions"]
    assert [[line["text"] for line in r["lines"]] for r in regions] == FIXTURE_REGIONS
    # PERO's region sorter put r002 (the heading) first; the file order is its reading order.
    assert [r["extra"]["pero_id"] for r in regions] == ["r002", "r001", "r000"]
    assert [r["id"] for r in regions] == prediction["reading_order"] == ["r1", "r2", "r3"]
    assert {r["type"] for r in regions} == {"text"}
    assert all("native_type" not in r for r in regions)
    assert regions[0]["polygon"] == [[292, 32], [35, 32], [35, 62], [292, 62], [292, 32]]
    lines = [line for r in regions for line in r["lines"]]
    assert [line["id"] for line in lines] == ["l1", "l2", "l3", "l4", "l5"]
    assert lines[0]["baseline"][0] == [35, 55]  # image pixels, as PERO wrote them
    assert all(len(line["polygon"]) >= 3 and len(line["baseline"]) >= 2 for line in lines)
    assert [line["confidence"] for line in lines] == [1.0] * 5
    assert lines[1]["extra"] == {"pero_id": "r001-l002"}


def test_native_page_xml_is_readable_by_the_benchmark():
    page, issues = read_page_xml(FIXTURES / "page.xml")
    assert page is not None, [i.format() for i in issues]
    assert all(i.severity is Severity.INFO for i in issues)
    assert (page.width, page.height, page.image_filename) == (900, 420, "page-1.png")
    prediction = pe.page_xml_to_prediction(PAGE_XML, "s1", (900, 420))
    lines = [line for r in prediction["regions"] for line in r["lines"]]
    assert [line.text for line in page.reference_lines] == [line["text"] for line in lines]


def test_reading_order_types_and_omissions():
    body = "".join(
        [
            region("a", ["first a", " "]),  # the blank line is dropped
            region("b", ["heading"], kind="heading"),
            region("c", [" "]),  # no text: only in the native outputs
            region("d", ["note"], kind="marginalia", coords="0,0 9,9"),  # too few points
        ]
    )
    order = (
        '<ReadingOrder><OrderedGroup id="ro">'
        '<RegionRefIndexed index="0" regionRef="b"/><RegionRefIndexed index="1" regionRef="d"/>'
        '<RegionRefIndexed index="2" regionRef="a"/></OrderedGroup></ReadingOrder>'
    )
    prediction = pe.page_xml_to_prediction(page_xml(body, order), "s1", (900, 420))
    assert errors(prediction) == []
    got = [(r["type"], r.get("native_type"), r["extra"]["pero_id"]) for r in prediction["regions"]]
    assert got == [("heading", "heading", "b"), ("other", "marginalia", "d"), ("text", None, "a")]
    assert prediction["text"] == "heading\n\nnote\n\nfirst a"
    note = prediction["regions"][1]
    assert "polygon" not in note
    assert "polygon" not in note["lines"][0]
    assert note["lines"][0]["baseline"] == [[0, 8], [9, 8]]
    assert note["lines"][0]["confidence"] == 0.5


def reading_order_text(order):
    body = "".join(region(rid, [rid]) for rid in "abcd")
    prediction = pe.page_xml_to_prediction(page_xml(body, order), "s1", (900, 420))
    assert errors(prediction) == []
    return prediction["text"].split("\n\n")


def test_reading_order_follows_index_not_xml_order():
    order = (
        '<ReadingOrder><OrderedGroup id="ro">'
        '<RegionRefIndexed index="1" regionRef="a"/><RegionRefIndexed index="0" regionRef="b"/>'
        '<RegionRefIndexed index="3" regionRef="c"/><RegionRefIndexed index="2" regionRef="d"/>'
        "</OrderedGroup></ReadingOrder>"
    )
    assert reading_order_text(order) == ["b", "a", "d", "c"]


def test_nested_reading_order_groups():
    order = (
        '<ReadingOrder><OrderedGroup id="ro">'
        '<OrderedGroupIndexed id="g2" index="1">'
        '<RegionRefIndexed index="1" regionRef="a"/><RegionRefIndexed index="0" regionRef="d"/>'
        "</OrderedGroupIndexed>"
        '<UnorderedGroupIndexed id="g1" index="0" regionRef="c">'
        '<RegionRef regionRef="b"/></UnorderedGroupIndexed>'
        "</OrderedGroup></ReadingOrder>"
    )
    # g1 first by index (its own region c, then b), then g2 by index (d, a).
    assert reading_order_text(order) == ["c", "b", "d", "a"]
    # Without integer indexes an ordered group keeps document order; unreferenced regions follow.
    order = (
        '<ReadingOrder><OrderedGroup id="ro"><RegionRefIndexed regionRef="c"/>'
        '<RegionRefIndexed index="0" regionRef="a"/></OrderedGroup></ReadingOrder>'
    )
    assert reading_order_text(order) == ["c", "a", "b", "d"]


def test_empty_and_malformed_page_xml():
    empty = pe.page_xml_to_prediction(page_xml(""), "s1", (900, 420))
    assert empty["text"] == ""
    assert "regions" not in empty
    assert "reading_order" not in empty
    with pytest.raises(pe.AdapterError, match="without a Page"):
        pe.page_xml_to_prediction(f'<PcGts xmlns="{PAGE}"/>', "s1", (9, 9))


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lang": "en"}, "unknown parameters: lang"),
        ({"device": "gpu:0"}, "device"),
        ({"device": "cuda"}, "device"),
        ({"layout_bundle": "../etc"}, "layout_bundle"),
        ({"ocr_bundle": "/cache/models/pero/x"}, "ocr_bundle"),
        ({"ocr_bundle": 3}, "ocr_bundle"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(pe.AdapterError, match=message):
        pe.parse_params(params)


def test_params_defaults():
    assert pe.parse_params({"device": None}) == pe.DEFAULT_PARAMS
    assert CONFIG["params"] == pe.DEFAULT_PARAMS  # engine.toml spells out the universal preset


def test_checksum_file_lists_the_universal_bundles():
    bundles = pe.read_checksums(ENGINE_DIR / "bundles.sha256")
    assert set(bundles) == {"layout_model", "universal.transformer.2025-03-19"}
    assert set(bundles["layout_model"]) == {"config.ini", "ParseNet_069500.pt"}
    assert set(bundles["universal.transformer.2025-03-19"]) == {
        "config.ini",
        "ocr_engine.json",
        "checkpoint_1420000.pt",
    }


def make_cache(root, bundles=BUNDLES, registered=None):
    """Bundles under <root>/cache/models/pero and a checksum file; returns the adapter env."""
    lines = []
    for bundle, files in bundles.items():
        directory = root / "cache" / pe.BUNDLES_DIR / bundle
        directory.mkdir(parents=True)
        for name, content in files.items():
            (directory / name).write_text(content, encoding="utf-8")
            digest = hashlib.sha256(content.encode()).hexdigest()
            lines.append(f"{digest}  {bundle}/{name}")
    checksums = root / "bundles.sha256"
    checksums.write_text("\n".join(registered or lines) + "\n", encoding="utf-8")
    return {"OCRBENCH_PERO_CACHE": str(root / "cache"), "OCRBENCH_PERO_CHECKSUMS": str(checksums)}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    env = make_cache(tmp_path)
    monkeypatch.setattr(pe, "CACHE", Path(env["OCRBENCH_PERO_CACHE"]))
    return pe.read_checksums(Path(env["OCRBENCH_PERO_CHECKSUMS"]))


def test_bundles_are_verified_against_their_checksums(cache):
    directory, files = pe.verify_bundle("ocr", cache)
    assert directory == pe.CACHE / "models" / "pero" / "ocr"
    assert set(files) == {"config.ini", "ocr_engine.json", "checkpoint.pt"}
    with pytest.raises(pe.AdapterError, match=r"bundle 'x' is not registered .*\(layout, ocr\)"):
        pe.verify_bundle("x", cache)
    (directory / "checkpoint.pt").write_text("changed", encoding="utf-8")
    with pytest.raises(pe.AdapterError, match="checkpoint.pt differs from bundles.sha256"):
        pe.verify_bundle("ocr", cache)
    (directory / "checkpoint.pt").unlink()
    with pytest.raises(pe.AdapterError, match="missing file .*checkpoint.pt"):
        pe.verify_bundle("ocr", cache)


def test_supplied_configurations_pass_their_checks(cache):
    for role, bundle in (("layout_bundle", "layout"), ("ocr_bundle", "ocr")):
        directory, files = pe.verify_bundle(bundle, cache)
        config = pe.read_config(directory, role)
        pe.check_references(config, directory, files, "cuda:0")
    directory, _ = pe.verify_bundle("ocr", cache)
    config = pe.read_config(directory, "ocr_bundle")
    references = pe.referenced_files(config, directory, "cuda:0")
    assert references == ["./ocr_engine.json", "checkpoint.pt"]


def write_ini(cache, bundle, text):
    directory = pe.CACHE / pe.BUNDLES_DIR / bundle
    (directory / "config.ini").write_text(text, encoding="utf-8")
    return directory


@pytest.mark.parametrize(
    ("role", "text", "message"),
    [
        ("layout_bundle", "[PAGE_PARSER\n", "not a valid INI file"),
        ("layout_bundle", "[OCR]\n", r"has no \[PAGE_PARSER\] section"),
        ("layout_bundle", "[PAGE_PARSER]\nRUN_LAYOUT_PARSER = maybe\n", "Not a boolean"),
        ("ocr_bundle", LAYOUT_INI, "RUN_LINE_CROPPER, RUN_OCR must be yes"),
        ("layout_bundle", OCR_INI, "RUN_LAYOUT_PARSER must be yes"),
        (
            "layout_bundle",
            LAYOUT_INI.replace("RUN_OCR = no", "RUN_OCR = yes"),
            "layout_bundle layout: RUN_OCR must be no$",
        ),
        (
            "ocr_bundle",
            OCR_INI.replace("RUN_DECODER = no", "RUN_DECODER = yes"),
            "RUN_DECODER must be no decoding with a language model is not supported",
        ),
    ],
)
def test_incompatible_configurations_are_rejected(cache, role, text, message):
    directory = write_ini(cache, "layout" if role == "layout_bundle" else "ocr", text)
    with pytest.raises(pe.AdapterError, match=message):
        pe.read_config(directory, role)


@pytest.mark.parametrize(
    ("bundle", "old", "new", "device", "message"),
    [
        ("layout", "./ParseNet.pt", "/opt/ParseNet.pt", "cuda:0", "is outside the bundle"),
        ("layout", "./ParseNet.pt", "../ocr/checkpoint.pt", "cuda:0", "is outside the bundle"),
        ("layout", "./ParseNet.pt", "./config.ini.bak", "cuda:0", "not a registered bundle file"),
        ("ocr", "./ocr_engine.json", "./missing.json", "cuda:0", "unreadable OCR_JSON"),
        ("ocr", "", "", "cpu", r"checkpoint.pt.cpu is not .*separate .cpu checkpoint"),
    ],
)
def test_missing_or_foreign_assets_are_rejected(cache, bundle, old, new, device, message):
    directory, files = pe.verify_bundle(bundle, cache)
    text = (directory / "config.ini").read_text(encoding="utf-8")
    write_ini(cache, bundle, text.replace(old, new) if old else text)
    config = pe.read_config(directory, f"{bundle}_bundle")
    with pytest.raises(pe.AdapterError, match=message):
        pe.check_references(config, directory, files, device)


def test_install_bundles_copies_and_checks_registered_files(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ENGINE_DIR))
    import install_bundles

    env = make_cache(tmp_path / "source")
    source = Path(env["OCRBENCH_PERO_CACHE"]) / pe.BUNDLES_DIR
    (source / "ocr" / "notes.txt").write_text("not registered", encoding="utf-8")
    monkeypatch.setattr(install_bundles, "CHECKSUMS", Path(env["OCRBENCH_PERO_CHECKSUMS"]))
    cache = tmp_path / "cache"
    install_bundles.main(source, cache)
    target = cache / pe.BUNDLES_DIR
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*.*")) == [
        "layout/ParseNet.pt",
        "layout/config.ini",
        "ocr/checkpoint.pt",
        "ocr/config.ini",
        "ocr/ocr_engine.json",
    ]
    install_bundles.main(source, cache)  # present and correct: kept
    (target / "ocr" / "checkpoint.pt").unlink()
    (source / "ocr" / "checkpoint.pt").write_text("changed", encoding="utf-8")
    with pytest.raises(SystemExit, match="ocr/checkpoint.pt: SHA-256 differs"):
        install_bundles.main(source, cache)
    assert not (target / "ocr" / "checkpoint.pt").exists()
    assert not (target / "ocr" / "checkpoint.pt.part").exists()


def make_dataset(root, widths):
    data = root / "data"
    data.mkdir()
    samples = []
    for sid, width in widths.items():
        if width is None:
            (data / f"{sid}.png").write_bytes(b"not an image")
        else:
            Image.new("L", (width, 420), 255).save(data / f"{sid}.png")
        (data / f"{sid}.xml").write_text("<PcGts/>", encoding="utf-8")
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


PARAMS = {"device": "cuda:0", "layout_bundle": "layout", "ocr_bundle": "ocr"}


def run_adapter(tmp_path, widths, params=PARAMS, env=None):
    config = dict(CONFIG, params=params)
    config["timeouts"] = {"sample": 20, "startup": 20}
    del config["resources"]
    env = {"PYTHONPATH": str(FIXTURES / "fake"), **make_cache(tmp_path), **(env or {})}
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, widths),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher([sys.executable, str(ADAPTER_PATH)], env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in widths}


def test_adapter_follows_the_engine_protocol(tmp_path):
    widths = {"page": 900, "blank": 300, "boom": 666, "broken": None, "alto": 555, "page2": 900}
    run, records = run_adapter(tmp_path, widths)
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),  # nothing found: a genuine empty result
        "boom": ("failure", "engine-error"),
        "broken": ("failure", "input-error"),  # the runner rejects it before the engine
        "alto": ("success", None),  # a failed ALTO export does not fail the sample
        "page2": ("success", None),  # the adapter keeps going after failed samples
    }
    assert "RuntimeError: fake layout error" in records["boom"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    outputs = {o["path"] for o in records["page"]["outputs"]}
    assert outputs == {"page.xml", "alto.xml", "prediction.json", "result.json"}
    assert (page / "output" / "page.xml").read_text(encoding="utf-8") == PAGE_XML
    assert (page / "prediction.txt").read_text(encoding="utf-8") == "\n\n".join(
        "\n".join(r) for r in FIXTURE_REGIONS
    )
    extra = records["page"]["engine_extra"]
    assert (extra["pero_regions"], extra["pero_lines"], extra["regions"], extra["lines"]) == (
        3,
        5,
        3,
        5,
    )
    assert "alto_error" not in extra
    assert extra["gpu_peak_memory_mb"] == 3072
    assert extra["environment"]["pero_ocr"] == "0.7.0+fake"
    assert extra["environment"]["gpu"] == "Fake GPU"
    bundles = extra["models"]["bundles"]
    assert bundles["layout_bundle"]["name"] == "layout"
    assert bundles["layout_bundle"]["path"].endswith("/models/pero/layout")
    assert set(bundles["ocr_bundle"]["files"]) == {"config.ini", "ocr_engine.json", "checkpoint.pt"}
    assert len(bundles["ocr_bundle"]["files"]["checkpoint.pt"]) == 64
    assert extra["models"]["job_model"] == CONFIG["model"]
    assert extra["params"] == PARAMS
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""
    alto = records["alto"]["engine_extra"]
    assert alto["alto_error"] == "ValueError: fake ALTO error"
    assert "alto.xml" not in {o["path"] for o in records["alto"]["outputs"]}


def test_cpu_device_needs_no_gpu(tmp_path):
    bundles = {
        "layout": BUNDLES["layout"],
        "ocr": BUNDLES["ocr"] | {"checkpoint.pt.cpu": "ocr model for the CPU"},
    }
    env = make_cache(tmp_path / "own", bundles) | {"FAKE_TORCH_GPUS": "0"}
    params = PARAMS | {"device": "cpu"}
    _, records = run_adapter(tmp_path, {"page": 900}, params=params, env=env)
    assert records["page"]["status"] == "success"
    assert "gpu" not in records["page"]["engine_extra"]["environment"]
    assert "gpu_peak_memory_mb" not in records["page"]["engine_extra"]


@pytest.mark.parametrize(
    ("params", "env", "message"),
    [
        ({"psm": 3}, {}, "unknown parameters: psm"),
        ({}, {"FAKE_TORCH_GPUS": "0"}, "cuda:0 requested but 0 CUDA devices are visible"),
        ({"ocr_bundle": "other"}, {}, "bundle 'other' is not registered"),
        ({"device": "cpu"}, {}, "the CPU needs a separate .cpu checkpoint"),
        ({}, "no-cache", "missing file /nonexistent/models/pero/layout/config.ini"),
        ({}, "no-checksums", "missing.sha256 does not exist"),
    ],
)
def test_setup_errors_fail_every_sample_explicitly(tmp_path, params, env, message):
    if env == "no-cache":
        env = {"OCRBENCH_PERO_CACHE": "/nonexistent"}
    elif env == "no-checksums":
        env = {"OCRBENCH_PERO_CHECKSUMS": str(tmp_path / "missing.sha256")}
    _, records = run_adapter(tmp_path, {"a": 900, "b": 300}, params=PARAMS | params, env=env)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


# Opt-in GPU smoke benchmark ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def pero_image():
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    for bundle, files in pe.read_checksums(ENGINE_DIR / "bundles.sha256").items():
        for name in files:
            if not (cache / pe.BUNDLES_DIR / bundle / name).is_file():
                pytest.fail(f"no PERO bundle {bundle} in {cache}: run make pero-bundles first")
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
def test_gpu_smoke_benchmark(tmp_path, pero_image):
    cache = str(pero_image)
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--cache", cache, "--run-id", "pero"]) == 0
    run = load_run(tmp_path / "runs" / "pero")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    # The blank page is a success either way; the universal preset finds one faint line on it.
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    extra = records["page-1"]["engine_extra"]
    assert extra["environment"]["pero_ocr"] == "0.7.0"
    assert extra["environment"]["pero_ocr_commit"] == "81b496e3084e294d2945e139fe60cf291010a83f"
    assert extra["environment"]["torch"] == "2.14.1+cu130"
    assert extra["environment"]["gpu"]
    assert extra["gpu_peak_memory_mb"] > 0
    assert "alto_error" not in extra
    checksums = pe.read_checksums(ENGINE_DIR / "bundles.sha256")
    bundles = extra["models"]["bundles"]
    assert bundles["layout_bundle"]["files"] == checksums["layout_model"]
    assert bundles["ocr_bundle"]["files"] == checksums["universal.transformer.2025-03-19"]
    sample = run.path / "samples" / "page-1" / "attempt-1"
    prediction = json.loads((sample / "prediction.json").read_text())
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    assert len(lines) == 5
    assert all(line["polygon"] and line["baseline"] for line in lines)
    page, _ = read_page_xml(sample / "output" / "page.xml")
    assert [line.text for line in page.reference_lines] == [line["text"] for line in lines]
    assert (sample / "output" / "alto.xml").read_text(encoding="utf-8").count("<TextLine") == 5

    # Scoring and re-scoring read only stored outputs: no container runs.
    for _ in range(2):
        assert main(["evaluate", str(run.path)]) == 0
    scores = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text())
    rates = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["page-1"] < 0.05
    assert rates["para-1"] < 0.05
    report = tmp_path / "report"
    assert main(["report", str(run.path), "-o", str(report), "--inspect"]) == 0
    assert (report / "index.html").is_file()

    # Without the cache mount, or with a GPU that is not visible, every sample fails explicitly.
    assert main([*args, "--run-id", "no-cache"]) == 0
    bad = tmp_path / "bad.toml"
    bad.write_text(Path(engine).read_text().replace('"cuda:0"', '"cuda:7"'), encoding="utf-8")
    bad_args = [*args[:2], "--engine", str(bad), *args[4:]]
    assert main([*bad_args, "--cache", cache, "--run-id", "bad-gpu"]) == 0
    for run_id, message in (("no-cache", "make pero-bundles"), ("bad-gpu", "cuda:7")):
        run = load_run(tmp_path / "runs" / run_id)
        records = [latest_record(run, s.id) for s in run.samples]
        assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
        assert all(message in r["message"] for r in records)
