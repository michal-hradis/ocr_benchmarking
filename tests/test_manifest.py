"""Dataset manifests: structure, file checks, image/GT pairing and the CLI."""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.issues import Severity
from ocrbench.manifest import load_manifest, sha256_file, validate_dataset

FIXTURES = Path(__file__).parent / "fixtures" / "pagexml"
SIZES = {"columns_2019": (400, 300), "legacy_2010": (200, 100), "paragraph_2013": (120, 40)}


@pytest.fixture
def dataset(tmp_path):
    """A dataset directory with generated blank images next to the PAGE XML fixtures."""
    data = tmp_path / "data"
    data.mkdir()
    for stem, size in SIZES.items():
        shutil.copy(FIXTURES / f"{stem}.xml", data / f"{stem}.xml")
        Image.new("L", size, 255).save(data / f"{stem}.png")
    return data


def sample(stem, **extra):
    return {"id": stem, "image": f"data/{stem}.png", "gt": f"data/{stem}.xml", **extra}


def write_manifest(directory, samples, **top):
    manifest = {"format": "ocrbench.dataset-manifest", "version": 1, "name": "t", **top}
    manifest["samples"] = samples
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def codes(report, severity=Severity.ERROR):
    return sorted(i.code for i in report.issues if i.severity is severity)


def test_valid_dataset_with_page_and_paragraph_samples(dataset):
    path = write_manifest(
        dataset.parent,
        [
            sample("columns_2019", doc_type="newspaper", language=["deu"], year=1890),
            sample("legacy_2010", split="dev"),
            sample(
                "paragraph_2013",
                unit="paragraph",
                image_sha256=sha256_file(dataset / "paragraph_2013.png"),
                gt_sha256=sha256_file(dataset / "paragraph_2013.xml"),
            ),
        ],
    )
    report = validate_dataset(path)
    assert report.ok, [i.format() for i in report.issues]
    assert [s.sample_id for s in report.summaries] == list(SIZES)  # manifest order, stable ids
    columns = report.manifest.samples[0]
    assert (columns.doc_type, columns.language, columns.year) == ("newspaper", ("deu",), 1890)
    assert columns.image == (dataset / "columns_2019.png").resolve()  # relative to manifest
    assert report.summaries[2].unit == "paragraph"
    assert report.summaries[0].reference_lines == 7
    # PAGE imageFilename "legacy.tif" differs from the manifest image name: noted, not an error
    assert "image-filename-differs" in codes(report, Severity.INFO)


def test_size_mismatch_is_error(dataset):
    Image.new("L", (300, 400), 255).save(dataset / "columns_2019.png")  # swapped width/height
    report = validate_dataset(write_manifest(dataset.parent, [sample("columns_2019")]))
    assert not report.ok
    assert codes(report) == ["image-size-mismatch"]
    assert report.summaries == []


def test_missing_files_and_checksum_mismatch(dataset):
    (dataset / "legacy_2010.xml").unlink()
    report = validate_dataset(
        write_manifest(
            dataset.parent,
            [sample("legacy_2010"), sample("columns_2019", gt_sha256="0" * 64)],
        )
    )
    assert codes(report) == ["checksum-mismatch", "missing-file"]
    assert {i.sample_id for i in report.issues} == {"legacy_2010", "columns_2019"}


def test_unreadable_image_and_broken_gt(dataset):
    (dataset / "columns_2019.png").write_bytes(b"not an image")
    (dataset / "legacy_2010.xml").write_text("<PcGts>", encoding="utf-8")
    report = validate_dataset(
        write_manifest(dataset.parent, [sample("columns_2019"), sample("legacy_2010")])
    )
    assert codes(report) == ["image-unreadable", "xml-parse-error"]


def test_exif_orientation_warns(dataset):
    image = Image.new("RGB", SIZES["paragraph_2013"], "white")
    exif = Image.Exif()
    exif[0x0112] = 6
    image.save(dataset / "paragraph_2013.jpg", exif=exif)
    entry = sample("paragraph_2013", image="data/paragraph_2013.jpg")
    report = validate_dataset(write_manifest(dataset.parent, [entry]))
    assert report.ok
    assert codes(report, Severity.WARNING) == ["exif-orientation"]


def test_duplicates_are_errors(dataset):
    path = write_manifest(
        dataset.parent,
        [sample("columns_2019"), sample("columns_2019"), {**sample("legacy_2010"), "id": "other"}],
    )
    path_data = json.loads(path.read_text())
    path_data["samples"][2]["image"] = "data/columns_2019.png"
    path.write_text(json.dumps(path_data))
    manifest, issues = load_manifest(path)
    assert manifest is None
    assert sorted(i.code for i in issues) == [
        "duplicate-image",
        "duplicate-image",
        "duplicate-sample-id",
    ]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"id": "has space"}, "invalid-sample-id"),
        ({"id": None}, "invalid-sample-id"),
        ({"unit": "line"}, "invalid-field"),
        ({"gt": ""}, "invalid-field"),
        ({"language": "deu"}, "invalid-field"),
        ({"year": "1890"}, "invalid-field"),
        ({"image_sha256": "ABC"}, "invalid-field"),
        ({"doc_type": ""}, "invalid-field"),
        ({"spilt": "dev"}, "unknown-field"),
    ],
)
def test_invalid_sample_fields(tmp_path, change, code):
    manifest, issues = load_manifest(write_manifest(tmp_path, [{**sample("a"), **change}]))
    assert manifest is None
    assert [i.code for i in issues] == [code]


@pytest.mark.parametrize(
    ("top", "code"),
    [
        ({"version": 2}, "unsupported-version"),
        ({"format": "other"}, "invalid-manifest"),
        ({"name": ""}, "invalid-manifest"),
        ({"extra": 1}, "unknown-field"),
    ],
)
def test_invalid_manifest_header(tmp_path, top, code):
    manifest, issues = load_manifest(write_manifest(tmp_path, [sample("a")], **top))
    assert manifest is None
    assert [i.code for i in issues] == [code]


def test_empty_and_malformed_manifests(tmp_path):
    _, issues = load_manifest(write_manifest(tmp_path, []))
    assert [i.code for i in issues] == ["invalid-manifest"]
    bad = tmp_path / "bad.json"
    bad.write_text('{"samples": [', encoding="utf-8")
    _, issues = load_manifest(bad)
    assert [i.code for i in issues] == ["manifest-json-error"]
    assert "line 1" in issues[0].message
    _, issues = load_manifest(tmp_path / "missing.json")
    assert [i.code for i in issues] == ["manifest-unreadable"]


def test_cli_validate_and_text(dataset, capsys):
    path = write_manifest(dataset.parent, [sample("columns_2019"), sample("legacy_2010")])
    assert main(["dataset", "validate", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2/2 samples valid; 0 errors, 1 warnings" in out
    assert "line-without-text" in out
    assert main(["dataset", "text", str(path), "legacy_2010", "--ids"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "r1\tr1l1\touter line",
        "r1a\tr1al1\tnested line",
        "cell1\t-\tcell text",
    ]
    assert main(["dataset", "text", str(path), "nope"]) == 1


def test_cli_validate_fails_on_error(dataset, capsys):
    (dataset / "legacy_2010.png").unlink()
    path = write_manifest(dataset.parent, [sample("legacy_2010")])
    assert main(["dataset", "validate", str(path)]) == 1
    assert "error: [missing-file] legacy_2010" in capsys.readouterr().out
