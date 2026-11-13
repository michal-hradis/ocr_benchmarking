"""ARCHER hypothesis export (#136): document names, block modes, statuses, XML round trip.

Runs are created on a synthetic ARCHER import (``fixtures/archer/ref/synthetic.xml``, tasks
``syn-print``, ``syn-columns``, ``syn-hand`` and ``syn-table``). Their records and standardized
predictions are written directly, as the runner stores them, so each test controls the exact
prediction; one test goes through the fake engine end to end.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
from lxml import etree
from PIL import Image

from ocrbench.archer import import_archer
from ocrbench.archer_export import (
    ArcherExportError,
    export_archer,
    read_hypothesis,
    xml_text,
)
from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.runner import ProcessLauncher, create_run, execute

FIXTURES = Path(__file__).parent / "fixtures"
FAKE = [sys.executable, str(FIXTURES / "fake-engine" / "fake_engine.py")]
SIZES = {"print.png": (200, 100), "columns.png": (200, 100), "hand.png": (100, 100)}
SIZES["table.png"] = (100, 100)
IDS = ("syn-columns", "syn-hand", "syn-print", "syn-table")  # manifest order
NAMES = {"syn-columns": "columns.png", "syn-hand": "hand.png", "syn-print": "print.png"}
NAMES["syn-table"] = "table.png"

# A schema for the accepted example form, to show that --xsd validation is applied. It is
# written for these tests from the local examples; it is not the official ARCHER schema.
EXAMPLE_XSD = """<?xml version="1.0" encoding="UTF-8"?>
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
  <xs:element name="archer-ocr">
    <xs:complexType><xs:sequence>
      <xs:element name="doc" minOccurs="0" maxOccurs="unbounded">
        <xs:complexType><xs:sequence>
          <xs:element name="block" type="xs:string" minOccurs="0" maxOccurs="unbounded"/>
        </xs:sequence><xs:attribute name="{attribute}" type="xs:string" use="required"/>
        </xs:complexType>
      </xs:element>
    </xs:sequence></xs:complexType>
  </xs:element>
</xs:schema>
"""


def engine(behaviors=None):
    return parse_engine_config(
        {
            "name": "fake",
            "engine_version": "1",
            "image": "ocrbench-test/fake-engine:test-v1",
            "params": {"behaviors": behaviors or {}},
            "timeouts": {"sample": 10, "startup": 10},
        }
    )


def archer_import(tmp_path, edit=None):
    root = tmp_path / "archer"
    (root / "ref").mkdir(parents=True)
    (root / "src").mkdir()
    reference = (FIXTURES / "archer" / "ref" / "synthetic.xml").read_text(encoding="utf-8")
    if edit:
        assert edit[0] in reference
        reference = reference.replace(*edit)
    (root / "ref" / "synthetic.xml").write_text(reference, encoding="utf-8")
    for name, size in SIZES.items():
        Image.new("L", size, 255).save(root / "src" / name)
    manifest, _ = import_archer(root / "src", root / "ref", root / "ocrbench")
    return manifest


def make_run(tmp_path, manifest=None):
    manifest = manifest or archer_import(tmp_path)
    return create_run(engine(), manifest, tmp_path / "runs", runtime="process")


def store(run, sample_id, status="success", prediction=None, *, reason=None, message=None):
    """Store an attempt the way the runner does: record.json and prediction.json."""
    attempt = run.path / "samples" / sample_id / "attempt-1"
    attempt.mkdir(parents=True)
    if prediction is not None:
        data = {"format": "ocrbench.prediction", "version": 1, "sample_id": sample_id}
        (attempt / "prediction.json").write_text(
            json.dumps(data | prediction, ensure_ascii=False), encoding="utf-8"
        )
    record = {
        "format": "ocrbench.execution-record",
        "version": 1,
        "sample_id": sample_id,
        "attempt": 1,
        "status": status,
        "reason": reason,
        "message": message,
        "prediction": "prediction.json" if prediction is not None else None,
    }
    (attempt / "record.json").write_text(json.dumps(record), encoding="utf-8")


def text_only(text):
    return {"text": text}


def regions(*texts_or_lines):
    """Regions in array order: a string is region text, a list is the region's lines."""
    items = []
    for index, value in enumerate(texts_or_lines):
        region = {"id": f"r{index}"}
        if isinstance(value, list):
            region["lines"] = [{"text": line} for line in value]
        elif value is None:
            region["type"] = "figure"
        else:
            region["text"] = value
        items.append(region)
    return {"text": "", "regions": items}


def complete(run, prediction=None, skip=()):
    for sample_id in IDS:
        if sample_id not in skip:
            store(run, sample_id, prediction=prediction or text_only(f"text of {sample_id}"))


def export(run, tmp_path, **kwargs):
    output = tmp_path / "submission.xml"
    docs, issues = export_archer(run.path, output, **kwargs)
    return read_hypothesis(output), docs, issues


def errors(run, tmp_path, **kwargs):
    with pytest.raises(ArcherExportError) as raised:
        export_archer(run.path, tmp_path / "submission.xml", **kwargs)
    assert not (tmp_path / "submission.xml").exists(), "nothing is written on an error"
    return [i.code for i in raised.value.issues if i.severity == "error"]


def test_page_mode_writes_one_document_per_sample_named_by_source_name(tmp_path):
    run = make_run(tmp_path)
    complete(run)
    docs, _, issues = export(run, tmp_path)
    assert docs == [(NAMES[sid], [f"text of {sid}"]) for sid in IDS]
    assert not issues
    raw = (tmp_path / "submission.xml").read_bytes()
    assert raw.startswith(b"<?xml version='1.0' encoding='UTF-8'?>\n<archer-ocr>\n  <doc ")


def test_page_mode_keeps_text_line_breaks_unicode_and_escapes_xml(tmp_path):
    run = make_run(tmp_path)
    tricky = 'Fish & "chips" <b>bold</b> ]]> ſtraße ž 漢字 🙂\nsecond line\n\nnext paragraph'
    complete(run, text_only(tricky))
    docs, _, _ = export(run, tmp_path)
    assert docs[0][1] == [tricky]
    raw = (tmp_path / "submission.xml").read_text(encoding="utf-8")
    assert 'Fish &amp; "chips" &lt;b&gt;bold&lt;/b&gt; ]]&gt; ſtraße' in raw


def test_regions_mode_emits_text_regions_in_reading_order_lines_joined(tmp_path):
    run = make_run(tmp_path)
    prediction = regions("Heading", ["first line", "second line"], None, "  ", "Footer")
    prediction["reading_order"] = ["r4", "r0", "r1", "r2", "r3"]
    complete(run, prediction)
    docs, _, _ = export(run, tmp_path, blocks="regions")
    # A figure without text and a whitespace-only region add no block.
    assert docs[0][1] == ["Footer", "Heading", "first line\nsecond line"]


def test_regions_mode_refuses_text_only_predictions_unless_page_fallback(tmp_path):
    run = make_run(tmp_path)
    complete(run, skip=("syn-print",))
    store(run, "syn-print", prediction=regions("A region"))
    assert errors(run, tmp_path, blocks="regions") == ["text-only-prediction"] * 3
    docs, exported, issues = export(run, tmp_path, blocks="regions", fallback="page")
    assert dict(docs)["print.png"] == ["A region"]
    assert dict(docs)["hand.png"] == ["text of syn-hand"]
    assert [d.fallback for d in exported] == [True, True, False, True]
    assert [i.code for i in issues] == ["page-fallback"] * 3


@pytest.mark.parametrize("blocks", ["page", "regions"])
def test_empty_valid_text_is_a_document_without_blocks(tmp_path, blocks):
    run = make_run(tmp_path)
    complete(run, text_only(" \n"))
    docs, _, _ = export(run, tmp_path, blocks=blocks)
    assert docs == [(NAMES[sid], []) for sid in IDS]


def test_source_name_comes_from_meta_not_the_image_file(tmp_path):
    edit = ("<source_name>print.png</source_name>", "<source_name>print-original.tif</source_name>")
    run = make_run(tmp_path, archer_import(tmp_path, edit))
    complete(run)
    docs, _, _ = export(run, tmp_path)
    assert dict(docs)["print-original.tif"] == ["text of syn-print"]


def test_missing_source_name_is_an_error_not_a_guess(tmp_path):
    edit = ("<source_name>print.png</source_name>", "")
    run = make_run(tmp_path, archer_import(tmp_path, edit))
    complete(run)
    assert errors(run, tmp_path) == ["missing-source-name"]


def test_duplicate_source_names_are_refused(tmp_path):
    # The importer refuses this already (ambiguous-image); the export checks it independently.
    manifest = archer_import(tmp_path)
    sidecar = manifest.parent / "archer" / "syn-hand.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["image"]["source_name"] = "print.png"
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    run = make_run(tmp_path, manifest)
    complete(run)
    assert errors(run, tmp_path) == ["duplicate-source-name"]


def test_a_run_over_part_of_the_import_is_refused(tmp_path):
    manifest = archer_import(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["samples"] = data["samples"][1:]
    subset = manifest.with_name("subset.json")
    subset.write_text(json.dumps(data), encoding="utf-8")
    run = create_run(engine(), subset, tmp_path / "runs", runtime="process")
    complete(run, skip=("syn-columns",))
    assert errors(run, tmp_path) == ["document-not-in-run"]


def test_run_on_a_dataset_that_is_not_an_archer_import_is_refused(tmp_path):
    manifest = archer_import(tmp_path)
    (manifest.parent / "import.json").unlink()
    run = make_run(tmp_path, manifest)
    complete(run)
    assert errors(run, tmp_path) == ["not-archer-import"]


def test_changed_reference_is_refused(tmp_path):
    manifest = archer_import(tmp_path)
    run = make_run(tmp_path, manifest)
    complete(run)
    page = manifest.parent / "page" / "syn-hand.xml"
    page.write_bytes(page.read_bytes() + b"\n")
    assert errors(run, tmp_path) == ["gt-changed"]


def test_pending_samples_are_always_an_error(tmp_path):
    run = make_run(tmp_path)
    complete(run, skip=("syn-hand",))
    for failed in ("error", "empty", "omit"):
        assert errors(run, tmp_path, failed=failed) == ["pending"]


def test_failed_samples_are_refused_or_explicitly_exported(tmp_path):
    run = make_run(tmp_path)
    complete(run, skip=("syn-hand",))
    store(run, "syn-hand", "failure", reason="timeout", message="no sample within 600 s")
    assert errors(run, tmp_path) == ["failed-sample"]

    docs, exported, issues = export(run, tmp_path, failed="empty")
    assert dict(docs)["hand.png"] == []
    hand = next(d for d in exported if d.sample_id == "syn-hand")
    assert (hand.status, hand.reason, hand.exported) == ("failure", "timeout", "empty")
    assert [(i.severity, i.code, i.sample_id) for i in issues] == [
        ("warning", "failed-sample", "syn-hand")
    ]

    report = tmp_path / "report.json"
    docs, _, _ = export(run, tmp_path, failed="omit", force=True, report=report)
    assert [name for name, _ in docs] == ["columns.png", "print.png", "table.png"]
    entries = {d["sample_id"]: d for d in json.loads(report.read_text())["documents"]}
    assert entries["syn-hand"] == {
        "sample_id": "syn-hand",
        "source_name": "hand.png",
        "status": "failure",
        "reason": "timeout",
        "exported": "omitted",
        "blocks": 0,
        "page_fallback": False,
    }


def test_truncated_and_partial_samples_are_exported_with_a_warning(tmp_path):
    run = make_run(tmp_path)
    complete(run, skip=("syn-hand", "syn-print"))
    store(run, "syn-hand", "truncated", text_only("cut"), message="token limit")
    store(run, "syn-print", "partial", text_only("part"))
    docs, exported, issues = export(run, tmp_path)
    assert dict(docs)["hand.png"] == ["cut"]
    assert dict(docs)["print.png"] == ["part"]
    assert [d.status for d in exported] == ["success", "truncated", "partial", "success"]
    assert [(i.code, i.sample_id) for i in issues] == [
        ("truncated-sample", "syn-hand"),
        ("partial-sample", "syn-print"),
    ]


def test_invalid_stored_prediction_is_refused(tmp_path):
    run = make_run(tmp_path)
    complete(run, skip=("syn-hand",))
    store(run, "syn-hand", prediction={"text": "x", "sample_id": "other"})
    assert errors(run, tmp_path) == ["invalid-prediction"]


def test_characters_xml_cannot_hold_are_replaced_and_reported(tmp_path):
    assert xml_text("a\r\nb\rc\x0cd\x00e￾") == ("a\nb\nc d e ", 3)
    run = make_run(tmp_path)
    complete(run, text_only("page one\x0c"))
    docs, _, issues = export(run, tmp_path)
    assert docs[0][1] == ["page one "]
    assert {i.code for i in issues} == {"xml-invalid-character"}


def test_existing_output_is_replaced_only_with_force(tmp_path):
    run = make_run(tmp_path)
    complete(run)
    output = tmp_path / "submission.xml"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(ArcherExportError):
        export_archer(run.path, output)
    assert output.read_text(encoding="utf-8") == "keep"
    export_archer(run.path, output, force=True)
    assert len(read_hypothesis(output)) == 4


def test_xsd_validation_is_applied(tmp_path):
    run = make_run(tmp_path)
    complete(run)
    good = tmp_path / "good.xsd"
    good.write_text(EXAMPLE_XSD.format(attribute="source_name"), encoding="utf-8")
    export(run, tmp_path, xsd=good)
    wrong = tmp_path / "id.xsd"  # the Annex B form, <doc id="…">
    wrong.write_text(EXAMPLE_XSD.format(attribute="id"), encoding="utf-8")
    with pytest.raises(ArcherExportError) as raised:
        export_archer(run.path, tmp_path / "other.xml", xsd=wrong)
    assert [i.code for i in raised.value.issues] == ["xsd-invalid"]
    assert not (tmp_path / "other.xml").exists()


def test_reader_rejects_structures_outside_the_example_form(tmp_path):
    path = tmp_path / "h.xml"
    for bad in (
        "<archer-ocr><doc id='a'/></archer-ocr>",
        "<archer-ocr><doc source_name='a'><block x='1'>t</block></doc></archer-ocr>",
        "<archer-ocr><doc source_name='a'><block>t<b/></block></doc></archer-ocr>",
        "<archer-ocr><doc source_name='a'>loose<block>t</block></doc></archer-ocr>",
        "<archer-ocr><doc source_name='a'/><doc source_name='a'/></archer-ocr>",
        "<other/>",
    ):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=str(path)):
            read_hypothesis(path)


def test_cli_exports_a_run_from_the_fake_engine(tmp_path, capsys):
    manifest = archer_import(tmp_path)
    run = create_run(
        engine({"syn-table": "failed"}), manifest, tmp_path / "runs", runtime="process"
    )
    execute(run, ProcessLauncher(FAKE), poll_interval=0.02)
    output = tmp_path / "out" / "submission.xml"
    assert main(["export", "archer", str(run.path), "-o", str(output)]) == 1
    assert "failed-sample" in capsys.readouterr().err
    assert not output.exists()

    args = ["export", "archer", str(run.path), "-o", str(output), "--failed", "omit"]
    assert main([*args, "--blocks", "regions"]) == 0
    printed = capsys.readouterr()
    assert "3 documents, 6 blocks (regions)" in printed.out
    assert "3 success, 1 failure; 1 omitted" in printed.out
    # The fake engine's Markdown "# id\n\nText of *id*." as converted, standardized text.
    assert read_hypothesis(output)[0] == ("columns.png", ["syn-columns", "Text of syn-columns."])
    etree.parse(str(output))  # also well-formed for a validating parser
    shutil.rmtree(tmp_path / "out")
