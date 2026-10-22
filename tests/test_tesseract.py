"""Tesseract adapter (engines/tesseract): hOCR mapping, parameters and the engine protocol.

Offline tests import the adapter and run it as a host process with a fake ``tesseract``
executable on PATH that copies a committed hOCR (tests/fixtures/tesseract/page.hocr, real
Tesseract 5.3.0 output for a synthetic 900x400 page); the ``best`` presets read fake models from a
temporary cache. The opt-in Docker tests build the engine image and run real Tesseract through
``ocrbench run``, ``evaluate`` and ``report``, the best presets with ``make tesseract-best`` models.
"""

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
from synthetic import smoke_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engines" / "tesseract"
HOCR = Path(__file__).parent / "fixtures" / "tesseract" / "page.hocr"
ADAPTER_PATH = ENGINE_DIR / "tesseract_engine.py"
ADAPTER = [sys.executable, str(ADAPTER_PATH)]
PRESETS = sorted((ENGINE_DIR / "presets").glob("*.toml"))

_spec = importlib.util.spec_from_file_location("tesseract_engine", ADAPTER_PATH)
tess = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tess)
sys.modules["tesseract_engine"] = tess  # fetch_tessdata imports it by name
_spec = importlib.util.spec_from_file_location("fetch_tessdata", ENGINE_DIR / "fetch_tessdata.py")
fetch_tessdata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_tessdata)

FIXTURE_TEXT = (
    "The Daily Chronicle\n\n"
    "The quick brown fox jumps over\nthe lazy dog near the river.\n\n"
    "Asecond paragraph & more text\nwith <angle> brackets too."
)
EMPTY_HOCR = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<div class='ocr_page' id='page_1' title='image "x.png"; bbox 0 0 900 400; ppageno 0'></div>
</body></html>"""


def errors(prediction):
    standardized, issues = validate_prediction(prediction, sample_id="s1", image_size=(900, 400))
    assert standardized is not None, [i.format() for i in issues]
    return [i for i in issues if i.severity is not Severity.INFO]


def test_hocr_maps_paragraphs_to_regions_with_line_geometry():
    prediction = tess.hocr_to_prediction(HOCR.read_text(encoding="utf-8"), "s1")
    assert errors(prediction) == []
    assert prediction["image"] == {"width": 900, "height": 400}
    assert prediction["text"] == FIXTURE_TEXT
    regions = prediction["regions"]
    assert [r["id"] for r in regions] == ["par_1_1", "par_1_2", "par_1_3"]
    assert {(r["type"], r["native_type"]) for r in regions} == {("text", "paragraph")}
    assert regions[1]["polygon"] == [[41, 127], [444, 127], [444, 193], [41, 193]]
    assert regions[1]["extra"] == {"block": "block_1_2", "lang": "eng"}
    first = regions[0]["lines"][0]
    assert first["id"] == "line_1_1"
    assert first["polygon"] == [[41, 39], [390, 39], [390, 78], [41, 78]]
    # "baseline 0.003 -9" from the bottom-left corner (41, 78) of the line box.
    assert first["baseline"] == [[41, 69], [390, pytest.approx(69 + 0.003 * 349)]]
    assert first["confidence"] == pytest.approx((96 + 95 + 96) / 3, abs=0.01)
    assert "type" not in first


def test_empty_page_is_a_valid_empty_prediction():
    prediction = tess.hocr_to_prediction(EMPTY_HOCR, "s1")
    assert prediction["text"] == ""
    assert "regions" not in prediction
    assert errors(prediction) == []


def test_blank_words_lines_and_paragraphs_are_dropped_and_line_classes_kept():
    hocr = HOCR.read_text(encoding="utf-8")
    hocr = (
        hocr.replace(">The</span>", "> </span>", 1)
        .replace(">Daily<", "><")
        .replace(">Chronicle<", ">\t<")
    )  # par_1_1 has no text left
    hocr = hocr.replace("class='ocr_line' id='line_1_4'", "class='ocr_caption' id='line_1_4'")
    hocr = hocr.replace(">river.</span>", "><em>river.</em></span>")
    prediction = tess.hocr_to_prediction(hocr, "s1")
    assert errors(prediction) == []
    assert [r["id"] for r in prediction["regions"]] == ["par_1_2", "par_1_3"]
    assert prediction["regions"][1]["lines"][0]["type"] == "caption"
    assert prediction["text"] == FIXTURE_TEXT.split("\n\n", 1)[1]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lang": "eng; rm"}, "lang"),
        ({"lang": ""}, "lang"),
        ({"psm": 14}, "psm"),
        ({"oem": "1"}, "oem"),
        ({"psm": True}, "psm"),
        ({"dpi": 0}, "dpi"),
        ({"tessdata": "legacy"}, "tessdata must be one of fast, best"),
        ({"prompt": "x"}, "unknown parameters: prompt"),
    ],
)
def test_invalid_params_are_rejected(params, message):
    with pytest.raises(tess.AdapterError, match=message):
        tess.parse_params(params)


def test_command_uses_configured_language_and_modes():
    params = tess.parse_params({"lang": "ces+deu", "psm": 4, "dpi": 300})
    command = tess.tesseract_command(Path("/input/images/a.png"), Path("/output/a/page"), params)
    assert command == [
        "tesseract", "/input/images/a.png", "/output/a/page", "-l", "ces+deu",
        "--psm", "4", "--oem", "1", "--dpi", "300", "txt", "hocr", "tsv",
    ]  # fmt: skip


def test_best_command_reads_models_from_the_cache_and_configs_from_the_image():
    params = tess.parse_params({"lang": "deu_latf", "psm": 3, "tessdata": "best"})
    best = Path("/cache") / tess.BEST_DIR
    command = tess.tesseract_command(Path("/input/images/a.png"), Path("/o/page"), params, best)
    configs = "/usr/share/tesseract-ocr/5/tessdata/configs"
    assert command == [
        "tesseract", "/input/images/a.png", "/o/page", "-l", "deu_latf", "--psm", "3",
        "--oem", "1", "--tessdata-dir", str(best),
        f"{configs}/txt", f"{configs}/hocr", f"{configs}/tsv",
    ]  # fmt: skip


# Presets and the tessdata_best models ------------------------------------------------------------


def test_presets_are_valid_and_differ_only_in_models_or_segmentation():
    default = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))
    configs = {p.stem: tomllib.loads(p.read_text(encoding="utf-8")) for p in PRESETS}
    assert set(configs) == {"best-psm3", "best-psm11", "best-psm6", "best-deu-latf-psm3"}
    assert len({c["name"] for c in [default, *configs.values()]}) == 5
    for config in configs.values():
        parse_engine_config(config)
        params = tess.parse_params(config["params"])
        assert params["tessdata"] == "best"
        assert params["oem"] == 1
        models = {f"{lang}.traineddata" for lang in params["lang"].split("+")}
        assert models <= set(tess.read_checksums(BEST_SHA256))
        same = ("image", "engine_version", "env", "resources", "timeouts")
        assert {k: config[k] for k in same} == {k: default[k] for k in same}
    # The checkpoint variant and the page segmentation mode each change alone.
    best3 = configs["best-psm3"]["params"]
    assert {**default["params"], "tessdata": "best"} == best3
    assert {**best3, "psm": 11} == configs["best-psm11"]["params"]
    assert {**best3, "psm": 6} == configs["best-psm6"]["params"]
    assert {**best3, "lang": "deu_latf"} == configs["best-deu-latf-psm3"]["params"]


BEST_SHA256 = ENGINE_DIR / "tessdata_best.sha256"
FAKE_MODELS = {"eng": b"fake eng model", "deu_latf": b"fake deu_latf model"}


def make_cache(root, models=FAKE_MODELS):
    """Fake tessdata_best models in <root>/cache and their checksums; returns the adapter env."""
    directory = root / "cache" / tess.BEST_DIR
    directory.mkdir(parents=True)
    lines = []
    for lang, content in models.items():
        (directory / f"{lang}.traineddata").write_bytes(content)
        lines.append(f"{tess.hashlib.sha256(content).hexdigest()}  {lang}.traineddata")
    checksums = root / "tessdata_best.sha256"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "OCRBENCH_TESSERACT_CACHE": str(root / "cache"),
        "OCRBENCH_TESSDATA_BEST_CHECKSUMS": str(checksums),
    }


def test_best_models_are_checked_before_use(tmp_path, monkeypatch):
    env = make_cache(tmp_path)
    monkeypatch.setattr(tess, "CACHE", Path(env["OCRBENCH_TESSERACT_CACHE"]))
    monkeypatch.setattr(tess, "BEST_CHECKSUMS", Path(env["OCRBENCH_TESSDATA_BEST_CHECKSUMS"]))
    directory = tmp_path / "cache" / tess.BEST_DIR
    assert tess.tessdata_dir(tess.parse_params({"lang": "deu_latf+eng", "tessdata": "best"})) == (
        directory
    )
    assert tess.tessdata_dir(tess.parse_params({"lang": "xxx"})) is None  # fast: from the image
    with pytest.raises(tess.AdapterError, match="ces.traineddata is not a registered"):
        tess.tessdata_dir(tess.parse_params({"lang": "eng+ces", "tessdata": "best"}))
    (directory / "eng.traineddata").write_bytes(b"changed")
    with pytest.raises(tess.AdapterError, match="eng.traineddata differs from tessdata_best"):
        tess.tessdata_dir(tess.parse_params({"lang": "eng", "tessdata": "best"}))
    (directory / "eng.traineddata").unlink()
    with pytest.raises(tess.AdapterError, match="missing .*eng.traineddata .*make tesseract-best"):
        tess.tessdata_dir(tess.parse_params({"lang": "eng", "tessdata": "best"}))


def test_fetch_downloads_checks_and_keeps_models(tmp_path, monkeypatch):
    source = make_cache(tmp_path / "source")
    origin = Path(source["OCRBENCH_TESSERACT_CACHE"]) / tess.BEST_DIR
    checksums = Path(source["OCRBENCH_TESSDATA_BEST_CHECKSUMS"])
    monkeypatch.setattr(fetch_tessdata, "CHECKSUMS", checksums)
    urls = []

    def retrieve(url, path):
        urls.append(url)
        Path(path).write_bytes((origin / url.rsplit("/", 1)[1]).read_bytes())

    monkeypatch.setattr(fetch_tessdata.urllib.request, "urlretrieve", retrieve)
    cache = tmp_path / "cache"
    fetch_tessdata.main(cache)
    assert sorted(urls) == [
        f"https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/{tess.BEST_REVISION}/"
        f"{lang}.traineddata"
        for lang in ("deu_latf", "eng")
    ]
    target = cache / tess.BEST_DIR
    assert (target / "eng.traineddata").read_bytes() == FAKE_MODELS["eng"]
    fetch_tessdata.main(cache)  # present and correct: kept
    assert len(urls) == 2
    (target / "eng.traineddata").write_bytes(b"x")
    (origin / "eng.traineddata").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="eng.traineddata: SHA-256 differs"):
        fetch_tessdata.main(cache)
    assert not (target / "eng.traineddata.part").exists()


# Engine protocol with a fake tesseract executable ----------------------------------------------

FAKE_TESSERACT = """\
import os, sys
from pathlib import Path
if sys.argv[1:] == ["--version"]:
    print("tesseract 5.3.0-fake")
    sys.exit(0)
image, outbase = sys.argv[1], sys.argv[2]
Path(f"{outbase}.args").write_text("\\n".join(sys.argv[3:]))
if "bad" in image:
    print("Error in pixReadStream: Unknown format", file=sys.stderr)
    sys.exit(1)
hocr = Path(os.environ["FAKE_HOCR"]).read_text()
if "blank" in image:
    hocr = os.environ["FAKE_EMPTY_HOCR"]
for ext in ("txt", "tsv"):
    Path(f"{outbase}.{ext}").write_text("native " + ext)
if "nohocr" not in image:
    Path(f"{outbase}.hocr").write_text(hocr)
"""


def make_dataset(root, ids):
    data = root / "data"
    data.mkdir()
    samples = []
    for sid in ids:
        Image.new("L", (900, 400), 255).save(data / f"{sid}.png")
        (data / f"{sid}.xml").write_text("<PcGts/>", encoding="utf-8")
        samples.append({"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml"})
    return write_manifest(data, samples)


def run_adapter(tmp_path, ids, params=None, env=None, config_path=ENGINE_DIR / "engine.toml"):
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config["params"] = params if params is not None else config["params"]
    config["timeouts"] = {"sample": 20, "startup": 20}
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "tesseract"
    fake.write_text(f"#!{sys.executable}\n{FAKE_TESSERACT}", encoding="utf-8")
    fake.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "FAKE_HOCR": str(HOCR),
        "FAKE_EMPTY_HOCR": EMPTY_HOCR,
        **(env or {}),
    }
    run = create_run(
        parse_engine_config(config),
        make_dataset(tmp_path, ids),
        tmp_path / "runs",
        runtime="process",
    )
    execute(run, ProcessLauncher(ADAPTER, env), poll_interval=0.02)
    return run, {sid: latest_record(run, sid) for sid in ids}


def test_adapter_follows_the_engine_protocol(tmp_path):
    run, records = run_adapter(tmp_path, ["page", "blank", "bad", "nohocr"])
    got = {sid: (r["status"], r["reason"]) for sid, r in records.items()}
    assert got == {
        "page": ("success", None),
        "blank": ("success", None),
        "bad": ("failure", "engine-error"),
        "nohocr": ("failure", "engine-error"),
    }
    assert "Unknown format" in records["bad"]["message"]
    assert "no hOCR" in records["nohocr"]["message"]
    page = run.path / "samples" / "page" / "attempt-1"
    outputs = {o["path"] for o in records["page"]["outputs"]}
    assert {"page.txt", "page.hocr", "page.tsv", "prediction.json", "result.json"} <= outputs
    assert records["page"]["primary"] == "prediction.json"
    assert (page / "prediction.txt").read_text(encoding="utf-8") == FIXTURE_TEXT
    extra = records["page"]["engine_extra"]
    assert extra["tesseract"] == "tesseract 5.3.0-fake"
    assert extra["params"] == {
        "lang": "ces+deu+eng", "psm": 3, "oem": 1, "dpi": None, "tessdata": "fast"
    }  # fmt: skip
    assert "--tessdata-dir" not in extra["command"]
    assert records["page"]["engine_seconds"] >= 0
    blank = run.path / "samples" / "blank" / "attempt-1"
    assert (blank / "prediction.txt").read_text(encoding="utf-8") == ""


def test_invalid_params_fail_every_sample_explicitly(tmp_path):
    _, records = run_adapter(tmp_path, ["a", "b"], params={"psm": 99})
    assert {r["status"] for r in records.values()} == {"failure"}
    assert {r["reason"] for r in records.values()} == {"engine-error"}
    assert all("psm" in r["message"] for r in records.values())


def test_best_preset_runs_with_checked_models_from_the_cache(tmp_path):
    env = make_cache(tmp_path / "own")
    preset = ENGINE_DIR / "presets" / "best-deu-latf-psm3.toml"
    run, records = run_adapter(tmp_path, ["page", "blank"], env=env, config_path=preset)
    assert {r["status"] for r in records.values()} == {"success"}
    command = records["page"]["engine_extra"]["command"]
    directory = str(tmp_path / "own" / "cache" / tess.BEST_DIR)
    assert command[command.index("--tessdata-dir") + 1] == directory
    assert command[:2] == ["-l", "deu_latf"]


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ({"eng": b"x"}, "deu_latf.traineddata is not a registered tessdata_best model"),
        (None, "missing"),
    ],
)
def test_missing_best_models_fail_every_sample_explicitly(tmp_path, models, message):
    if models is None:
        env = make_cache(tmp_path / "own")
        env["OCRBENCH_TESSERACT_CACHE"] = str(tmp_path / "nonexistent")
    else:
        env = make_cache(tmp_path / "own", models)
    preset = ENGINE_DIR / "presets" / "best-deu-latf-psm3.toml"
    _, records = run_adapter(tmp_path, ["a", "b"], env=env, config_path=preset)
    assert {(r["status"], r["reason"]) for r in records.values()} == {("failure", "engine-error")}
    assert all(message in r["message"] for r in records.values())


# Opt-in Docker smoke benchmark -------------------------------------------------------------------

CONFIG = tomllib.loads((ENGINE_DIR / "engine.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tesseract_image():
    subprocess.run(
        ["docker", "build", "-q", "-t", CONFIG["image"], str(ENGINE_DIR)],
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    )
    return CONFIG["image"]


@pytest.mark.docker
def test_docker_smoke_benchmark(tmp_path, tesseract_image, capsys):
    manifest = smoke_dataset(tmp_path / "data")
    assert main(["dataset", "validate", str(manifest)]) == 0
    engine = str(ENGINE_DIR / "engine.toml")
    runs = str(tmp_path / "runs")
    args = ["run", "start", "--engine", engine, "--dataset", str(manifest), "--runs-dir", runs]
    assert main([*args, "--run-id", "tess"]) == 0
    run = load_run(tmp_path / "runs" / "tess")
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    assert {sid: r["status"] for sid, r in records.items()} == {
        "page-1": "success",
        "para-1": "success",
        "blank-1": "success",
    }
    assert records["page-1"]["engine_extra"]["tesseract"].startswith("tesseract 5.3.0")
    prediction = json.loads(
        (run.path / "samples" / "page-1" / "attempt-1" / "prediction.json").read_text()
    )
    assert len(prediction["regions"]) >= 2
    lines = [line for region in prediction["regions"] for line in region["lines"]]
    assert len(lines) == 5
    assert all("polygon" in line and "baseline" in line for line in lines)
    blank = run.path / "samples" / "blank-1" / "attempt-1" / "prediction.txt"
    assert blank.read_text(encoding="utf-8") == ""  # a genuine empty result, not a failure

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

    # A language without traineddata is an explicit engine failure for every sample.
    bad = tmp_path / "bad.toml"
    bad.write_text(
        (ENGINE_DIR / "engine.toml").read_text().replace('"ces+deu+eng"', '"xxx"'),
        encoding="utf-8",
    )
    args[3] = str(bad)
    assert main([*args, "--run-id", "bad-lang"]) == 0
    run = load_run(tmp_path / "runs" / "bad-lang")
    records = [latest_record(run, s.id) for s in run.samples]
    assert {(r["status"], r["reason"]) for r in records} == {("failure", "engine-error")}
    assert all("xxx" in r["message"] for r in records)


@pytest.mark.docker
def test_docker_best_presets(tmp_path, tesseract_image):
    cache = Path(os.environ.get("OCRBENCH_CACHE", Path.home() / ".cache" / "ocrbench"))
    for name in tess.read_checksums(BEST_SHA256):
        if not (cache / tess.BEST_DIR / name).is_file():
            pytest.fail(f"no tessdata_best {name} in {cache}: run make tesseract-best first")
    manifest = smoke_dataset(tmp_path / "data")
    runs = str(tmp_path / "runs")
    rates = {}
    for preset in PRESETS:
        args = ["run", "start", "--engine", str(preset), "--dataset", str(manifest)]
        args += ["--runs-dir", runs, "--cache", str(cache), "--run-id", preset.stem]
        assert main(args) == 0
        run = load_run(tmp_path / "runs" / preset.stem)
        records = {s.id: latest_record(run, s.id) for s in run.samples}
        assert {r["status"] for r in records.values()} == {"success"}, preset.stem
        assert "--tessdata-dir" in records["page-1"]["engine_extra"]["command"]
        assert main(["evaluate", str(run.path)]) == 0
        scores = json.loads((run.path / "evaluations" / "1" / "scores.json").read_text())
        rates[preset.stem] = {s["sample_id"]: s["chars"]["rate"] for s in scores["samples"]}
    assert rates["best-psm3"]["page-1"] < 0.1
    assert rates["best-psm6"]["para-1"] < 0.1
    assert main(["report", *(str(tmp_path / "runs" / p.stem) for p in PRESETS), "-o",
                 str(tmp_path / "report")]) == 0  # fmt: skip
