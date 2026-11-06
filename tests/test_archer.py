"""ARCHER import (#135): pairing, validation, lossless sidecar, PAGE XML reference, text view.

``fixtures/archer/ref/synthetic.xml`` is a hand-written synthetic ARCHER file with four tasks:
an ordinary printed page (span tags, ``(YYY)``, image and noise blocks), two columns linked across
a figure (rotated box, a block ID starting with ``-``, an empty box), handwriting and a table.
"""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from lxml import etree
from PIL import Image

from ocrbench.archer import TEXT_VIEW, ArcherImportError, import_archer, text_view
from ocrbench.cli import main
from ocrbench.evaluate import reference_lines
from ocrbench.manifest import load_manifest, sha256_file, validate_dataset
from ocrbench.pagexml import read_page_xml
from ocrbench.runner import _job_entry

FIXTURE = Path(__file__).parent / "fixtures" / "archer" / "ref" / "synthetic.xml"
PAGE_SCHEMA = Path(__file__).parent / "fixtures" / "page-schema" / "pagecontent-2019-07-15.xsd"
SIZES = {"print.png": (200, 100), "columns.png": (200, 100), "hand.png": (100, 100)}
SIZES["table.png"] = (100, 100)
PAGE = "{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}"


@pytest.fixture
def root(tmp_path):
    """An ARCHER directory: ``ref/`` with the synthetic XML, ``src/`` with blank images."""
    (tmp_path / "ref").mkdir()
    (tmp_path / "src").mkdir()
    shutil.copy(FIXTURE, tmp_path / "ref" / "synthetic.xml")
    for name, size in SIZES.items():
        Image.new("L", size, 255).save(tmp_path / "src" / name)
    return tmp_path


def run_import(root, **kwargs):
    return import_archer(root / "src", root / "ref", root / "out", **kwargs)


def failure_codes(root, **kwargs):
    with pytest.raises(ArcherImportError) as raised:
        run_import(root, **kwargs)
    assert not (root / "out").exists(), "nothing is written when the import fails"
    return [i.code for i in raised.value.issues if i.severity == "error"]


def edit_reference(root, old, new, count=1):
    path = root / "ref" / "synthetic.xml"
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def test_text_view_removes_structure_tags_and_illegible_markers_only():
    view = text_view("Text", "(PR)A/RES/1 (E)\nPage 3(/PR) (HW)written(/HW)")
    assert view.text == "A/RES/1 (E)\nPage 3 written"
    assert view.illegible == 0
    assert text_view("Image", "caption-like text").text == ""
    assert text_view("Noise", None).text == ""
    # Non-textual handwritten additions (signatures, ticks, arrows) are never scored.
    assert text_view("Handwritten_annotations", "(HW)note(/HW)").text == ""


@pytest.mark.parametrize(
    ("raw", "scored"),
    [
        ("1(YYY)87", "1 87"),  # inside a word: never fused into the untranscribed "187"
        ("(YYY)ldiers trained", " ldiers trained"),
        ("www.(YYY).org.uk", "www. .org.uk"),
        ("before (YYY) after", "before   after"),  # standalone
        ("(YYY)", ""),
        ("(PR)(YYY)(/PR)", ""),
    ],
)
def test_illegible_markers_become_word_breaks(raw, scored):
    view = text_view("Text", raw)
    assert view.text == scored
    assert view.illegible == raw.count("(YYY)")


def test_text_view_flattens_table_rows_and_drops_rules():
    raw = "| No. | Name |\n|---|:--:|\n| 61 | Abdel Hadi |\n| 62 |  | a \\| b |\nnot a row"
    assert text_view("Table", raw).text == "No. Name\n\n61 Abdel Hadi\n62 a | b\nnot a row"
    assert text_view("Text", "| 61 | x |").text == "| 61 | x |", "only Table blocks"


def test_import_writes_a_valid_dataset_with_the_text_view(root):
    manifest_path, issues = run_import(root)
    report = validate_dataset(manifest_path)
    assert report.ok, [i.format() for i in report.issues]
    manifest, _ = load_manifest(manifest_path)
    assert manifest.name == "archer-en-text-v1"
    assert TEXT_VIEW in manifest.description
    assert "not official ARCHER" in manifest.description
    by_id = {s.id: s for s in manifest.samples}
    assert list(by_id) == ["syn-columns", "syn-hand", "syn-print", "syn-table"]
    assert by_id["syn-print"].image == (root / "src" / "print.png").resolve()
    assert by_id["syn-print"].language == ("EN",)
    assert by_id["syn-print"].year == 1998
    assert by_id["syn-hand"].year is None
    assert by_id["syn-print"].image_sha256 == sha256_file(root / "src" / "print.png")

    def lines(sample_id):
        return reference_lines(sample_id, by_id[sample_id].gt)

    # Annotation order, each text block once; image, noise and the tag-only block contribute none.
    assert lines("syn-print") == [
        "A/RES/1 (E)\nPage 3",
        "1.  The Commission welcomed\nthe report of 1 87.\n\n ",
    ]
    # Linked columns are not reordered; the block without a box keeps its text.
    assert lines("syn-columns") == [
        "second column continues\nthe story.",
        "First column starts\nthe story and",
        "Fig. 1 caption",
        "Box without position.",
    ]
    assert lines("syn-hand") == ["Printed line. OK!\n ldiers trained"], "not the signature"
    assert lines("syn-table") == ["No. Name\n\n61 Abdel Hadi\n62 a | b"]
    codes = {(i.code, i.location.rsplit("/", 1)[-1]) for i in issues}
    assert ("illegible-removed", "9body") in codes
    assert ("empty-text", "blank") in codes
    assert ("no-geometry", "unplaced") in codes
    assert ("text-not-scored", "scribble") in codes


def test_page_xml_keeps_geometry_without_inventing_it(root):
    run_import(root)
    tree = ET.parse(root / "out" / "page" / "syn-columns.xml").getroot()
    page = tree.find(f"{PAGE}Page")
    assert (page.get("imageWidth"), page.get("imageHeight")) == ("200", "100")
    regions = {r.get("id"): r for r in page if r.get("id")}
    assert list(regions) == ["b_-colB", "b_colA", "b_fig", "b_cap", "b_unplaced"]
    assert regions["b_fig"].tag == f"{PAGE}ImageRegion"
    assert regions["b_colA"].find(f"{PAGE}Coords").get("points") == "10,10 89,24 82,63 3,49"
    assert regions["b_colA"].get("orientation") == "-10"
    assert regions["b_-colB"].find(f"{PAGE}Coords").get("points") == "110,10 190,10 190,90 110,90"
    assert regions["b_unplaced"].find(f"{PAGE}Coords").get("points") == "0,0 0,0 0,0 0,0"
    assert "geometry:none" in regions["b_unplaced"].get("custom")
    assert "geometry" not in regions["b_colA"].get("custom")
    assert "id:-colB" in regions["b_-colB"].get("custom")
    relations = [
        (
            r.find(f"{PAGE}SourceRegionRef").get("regionRef"),
            r.find(f"{PAGE}TargetRegionRef").get("regionRef"),
        )
        for r in page.iter(f"{PAGE}Relation")
    ]
    assert relations == [("b_colA", "b_-colB"), ("b_fig", "b_cap"), ("b_colA", "b_unplaced")]
    assert [r.get("id") for r in page.iter(f"{PAGE}Relation")] == ["rel_1", "rel_2", "rel_3"]
    gt, issues = read_page_xml(root / "out" / "page" / "syn-columns.xml")
    assert gt.regions[-1].coords == ((0, 0),) * 4
    assert not [i for i in issues if i.severity != "info"]


def test_page_xml_is_valid_against_the_page_2019_schema(root):
    run_import(root)
    schema = etree.XMLSchema(etree.parse(str(PAGE_SCHEMA)))
    pages = sorted((root / "out" / "page").glob("*.xml"))
    assert len(pages) == 4
    for path in pages:
        document = etree.parse(str(path))
        assert schema.validate(document), (path.name, [str(e) for e in schema.error_log])
    hand = etree.parse(str(root / "out" / "page" / "syn-hand.xml")).getroot()
    [graphic] = hand.iter(f"{PAGE}GraphicRegion")
    assert graphic.get("type") == "handwritten-annotation"
    assert graphic.find(f"{PAGE}TextEquiv") is None


def test_sidecar_and_original_copy_are_lossless(root):
    run_import(root)
    out = root / "out"
    assert (out / "archer" / "ref" / "synthetic.xml").read_bytes() == FIXTURE.read_bytes()
    side = json.loads((out / "archer" / "syn-columns.json").read_text(encoding="utf-8"))
    assert side["text_view"] == TEXT_VIEW
    assert side["source"] == {
        "file": "archer/ref/synthetic.xml",
        "sha256": sha256_file(FIXTURE),
        "task_index": 1,
    }
    assert side["image"]["original_width"] == 200
    source = ET.parse(FIXTURE).getroot()[1]
    blocks = source.find("annotations").findall("block")
    assert [b["id"] for b in side["blocks"]] == [b.get("id") for b in blocks]
    assert [b["label"] for b in side["blocks"]] == [b.get("label") for b in blocks]
    assert [b["bbox"] for b in side["blocks"]] == [dict(b.find("bbox").attrib) for b in blocks]
    assert [b["text"] for b in side["blocks"]] == [b.findtext("text") for b in blocks]
    assert side["blocks"][-1]["polygon"] is None
    assert side["relations"] == [
        {"from": r.get("from"), "to": r.get("to")} for r in source.iter("relation")
    ]
    record = json.loads((out / "import.json").read_text(encoding="utf-8"))
    assert record["references"] == [
        {"file": "archer/ref/synthetic.xml", "sha256": sha256_file(FIXTURE)}
    ]


def test_reimport_is_byte_identical(root):
    run_import(root)
    first = {
        p.relative_to(root / "out"): p.read_bytes()
        for p in (root / "out").rglob("*")
        if p.is_file()
    }
    assert failure_codes_without_removal(root) == ["output-exists"]
    run_import(root, force=True)
    second = {
        p.relative_to(root / "out"): p.read_bytes()
        for p in (root / "out").rglob("*")
        if p.is_file()
    }
    assert first == second


def failure_codes_without_removal(root):
    with pytest.raises(ArcherImportError) as raised:
        run_import(root)
    return [i.code for i in raised.value.issues if i.severity == "error"]


def test_refuses_to_overwrite_a_foreign_directory(root):
    (root / "out").mkdir()
    (root / "out" / "keep.txt").write_text("x")
    with pytest.raises(ArcherImportError) as raised:
        run_import(root, force=True)
    assert [i.code for i in raised.value.issues if i.severity == "error"] == ["output-not-empty"]
    assert (root / "out" / "keep.txt").exists()


def test_force_never_deletes_files_the_import_did_not_write(root):
    run_import(root)
    (root / "out" / "notes.txt").write_text("mine")
    (root / "out" / "page" / "extra.xml").write_text("<x/>")
    with pytest.raises(ArcherImportError) as raised:
        run_import(root, force=True)
    [error] = [i for i in raised.value.issues if i.severity == "error"]
    assert error.code == "output-not-empty"
    assert "notes.txt" in error.message
    assert "page/extra.xml" in error.message
    assert (root / "out" / "notes.txt").read_text() == "mine"
    assert (root / "out" / "page" / "extra.xml").exists()


def test_force_rejects_an_import_record_without_a_file_list(root):
    run_import(root)
    record = json.loads((root / "out" / "import.json").read_text(encoding="utf-8"))
    del record["files"]
    (root / "out" / "import.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ArcherImportError) as raised:
        run_import(root, force=True)
    assert [i.code for i in raised.value.issues if i.severity == "error"] == ["output-not-empty"]
    assert (root / "out" / "manifest.json").exists()


def test_force_replaces_stale_generated_files_and_leaves_no_staging(root):
    run_import(root)
    edit_reference(root, '<task id="syn-table">', '<task id="syn-table2">')
    run_import(root, force=True)
    assert not (root / "out" / "page" / "syn-table.xml").exists()
    assert (root / "out" / "page" / "syn-table2.xml").exists()
    assert sorted(p.name for p in root.iterdir()) == ["out", "ref", "src"]
    record = json.loads((root / "out" / "import.json").read_text(encoding="utf-8"))
    on_disk = sorted(
        p.relative_to(root / "out").as_posix() for p in (root / "out").rglob("*") if p.is_file()
    )
    assert sorted(record["files"]) == on_disk


def test_an_empty_output_directory_is_used(root):
    (root / "out").mkdir()
    manifest_path, _ = run_import(root)
    assert manifest_path.is_file()


@pytest.mark.parametrize("name", ["../outside.png", "ABSOLUTE"])
def test_image_names_must_stay_inside_src(root, name):
    outside = root / "outside.png"
    Image.new("L", (100, 100), 255).save(outside)
    name = str(outside) if name == "ABSOLUTE" else name
    edit_reference(root, ">hand.png</image>", f">{name}</image>")
    edit_reference(
        root, "<source_name>hand.png</source_name>", f"<source_name>{name}</source_name>"
    )
    assert failure_codes(root) == ["image-outside-src"]


def test_missing_image_names_a_same_stem_candidate(root):
    (root / "src" / "hand.png").rename(root / "src" / "hand.jpg")
    with pytest.raises(ArcherImportError) as raised:
        run_import(root)
    [missing] = [i for i in raised.value.issues if i.code == "missing-image"]
    assert missing.sample_id == "syn-hand"
    assert "hand.jpg" in missing.message


def test_ambiguous_image_names(root):
    edit_reference(
        root, "<source_name>table.png</source_name>", "<source_name>hand.png</source_name>"
    )
    assert set(failure_codes(root)) == {"ambiguous-image"}


def test_source_name_mismatch_alone_is_a_warning(root):
    edit_reference(
        root, "<source_name>table.png</source_name>", "<source_name>gone.png</source_name>"
    )
    _, issues = run_import(root)
    assert "source-name-differs" in [i.code for i in issues]


def test_duplicate_task_ids_across_files(root):
    shutil.copy(FIXTURE, root / "ref" / "copy.xml")
    assert set(failure_codes(root)) >= {"duplicate-task-id"}


def test_image_reused_by_two_tasks(root):
    edit_reference(root, "table.png", "hand.png", count=2)
    assert set(failure_codes(root)) == {"image-reused"}


def test_image_size_must_match_declared_dimensions(root):
    Image.new("L", (101, 100), 255).save(root / "src" / "hand.png")
    assert failure_codes(root) == ["image-size-mismatch"]


@pytest.mark.parametrize("value", ["0", "-5", "12.5", ""])
def test_invalid_dimensions(root, value):
    edit_reference(
        root,
        'original_width="100" original_height="100">hand.png',
        f'original_width="{value}" original_height="100">hand.png',
    )
    assert failure_codes(root) == ["invalid-dimensions"]


def test_malformed_xml(root):
    (root / "ref" / "broken.xml").write_text("<dataset><task id='x'>", encoding="utf-8")
    assert failure_codes(root) == ["xml-parse-error"]


def test_not_an_archer_file(root):
    (root / "ref" / "other.xml").write_text("<PcGts/>", encoding="utf-8")
    assert failure_codes(root) == ["not-archer-xml"]


@pytest.mark.parametrize(
    ("old", "new", "code"),
    [
        ('label="Table"', 'label="Formula"', "unknown-label"),
        ('id="colA"', 'id="-colB"', "duplicate-block-id"),
        (
            '<relation from="fig" to="cap" />',
            '<relation from="fig" to="nope" />',
            "unknown-relation-block",
        ),
        ('width="80" height="30"', 'width="-80" height="30"', "malformed-bbox"),
        ('width="80" height="30"', 'width="wide" height="30"', "malformed-bbox"),
        ('<task id="syn-hand">', '<task id="syn hand">', "invalid-task-id"),
        ('<task id="syn-hand">', "<task>", "missing-task-id"),
    ],
)
def test_malformed_references(root, old, new, code):
    edit_reference(root, old, new)
    assert code in failure_codes(root)


def test_cli_import_then_validate(root, capsys):
    assert main(["dataset", "import-archer", str(root)]) == 0
    assert "wrote" in capsys.readouterr().out
    assert main(["dataset", "validate", str(root / "ocrbench" / "manifest.json")]) == 0
    assert main(["dataset", "import-archer", str(root)]) == 1
    assert "use --force" in capsys.readouterr().err
    assert main(["dataset", "import-archer", str(root), "--force", "--name", "x"]) == 0
    assert load_manifest(root / "ocrbench" / "manifest.json")[0].name == "x"


def test_job_entries_carry_no_reference_data(root):
    manifest_path, _ = run_import(root)
    manifest, _ = load_manifest(manifest_path)

    class Engine:
        metadata = ("language", "year")

    for sample in manifest.samples:
        entry = _job_entry(sample, Engine())
        assert set(entry) <= {"id", "unit", "language", "year"}
        assert "archer" not in json.dumps(entry)
        assert "page/" not in json.dumps(entry)
