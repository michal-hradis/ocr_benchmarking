"""PAGE XML reader: namespaces, reading order, text precedence and malformed input."""

from pathlib import Path

import pytest

from ocrbench.issues import Severity
from ocrbench.pagexml import read_page_xml

FIXTURES = Path(__file__).parent / "fixtures" / "pagexml"


def codes(issues, severity=None):
    return [i.code for i in issues if severity is None or i.severity is severity]


def write_page(tmp_path, body, *, namespace="2019-07-15", size='imageWidth="100" imageHeight="50"'):
    path = tmp_path / "page.xml"
    path.write_text(
        f'<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/{namespace}">'
        f'<Page imageFilename="x.png" {size}>{body}</Page></PcGts>',
        encoding="utf-8",
    )
    return path


def test_reading_order_differs_from_document_order():
    page, issues = read_page_xml(FIXTURES / "columns_2019.xml", "cols")
    assert page is not None
    assert page.schema_version == "2019-07-15"
    assert page.reading_order == "page"
    assert [r.id for r in page.regions] == ["r_head", "r_left", "r_right", "r_caption"]
    assert [line.text for line in page.reference_lines] == [
        "Die Zeitung",  # TextEquiv with the lowest @index wins
        "erste Zeile",  # lines ordered by @index, not document order
        "zweite Zeile",
        "Der Krieg.",  # repeated lines are both kept
        "Der Krieg.",
        "  Ende  ſchön  ",  # text is preserved exactly, not normalized
        "Bild 1.",  # region-level text of a region without lines
    ]
    caption = page.reference_lines[-1]
    assert (caption.region_id, caption.line_id) == ("r_caption", None)
    assert codes(issues, Severity.ERROR) == []
    assert codes(issues, Severity.WARNING) == ["line-without-text"]
    assert issues[codes(issues).index("line-without-text")].location == "columns_2019.xml#l_right_3"
    assert set(codes(issues, Severity.INFO)) == {"unordered-group", "region-text-ignored"}


def test_legacy_namespace_point_coordinates_and_nested_regions():
    page, issues = read_page_xml(FIXTURES / "legacy_2010.xml")
    assert page is not None
    assert page.schema_version == "2010-03-19"
    assert page.reading_order == "document"
    assert [(r.id, r.parent_id) for r in page.regions] == [
        ("r1", None),
        ("r1a", "r1"),
        ("cell1", None),
    ]
    assert [line.text for line in page.reference_lines] == [
        "outer line",
        "nested line",
        "cell text",
    ]
    line = page.regions[0].lines[0]
    assert line.coords == ((0, 0), (200, 0), (200, 20), (0, 20))
    assert line.baseline is None
    assert codes(issues) == ["no-reading-order"]


def test_paragraph_crop_out_of_bounds_points_are_info_only():
    page, issues = read_page_xml(FIXTURES / "paragraph_2013.xml")
    assert page is not None
    assert [line.text for line in page.reference_lines] == ["Příliš žluťoučký", "kůň."]
    assert page.regions[0].lines[0].coords[0] == (-2, 0)  # kept as annotated
    assert codes(issues) == ["points-out-of-bounds"]
    assert issues[0].severity is Severity.INFO
    assert "4 point(s)" in issues[0].message


def test_region_text_is_never_counted_twice(tmp_path):
    path = write_page(
        tmp_path,
        '<TextRegion id="r"><TextEquiv><Unicode>a b</Unicode></TextEquiv>'
        '<TextLine id="l1"><TextEquiv><Unicode>a</Unicode></TextEquiv></TextLine>'
        '<TextLine id="l2"><TextEquiv><Unicode>b</Unicode></TextEquiv></TextLine>'
        "</TextRegion>",
    )
    page, _ = read_page_xml(path)
    assert [line.text for line in page.reference_lines] == ["a", "b"]


@pytest.mark.parametrize("reading_order", [False, True])
def test_parent_region_text_is_not_counted_twice_with_nested_regions(tmp_path, reading_order):
    order = (
        '<ReadingOrder><OrderedGroup id="g"><RegionRefIndexed index="0" regionRef="parent"/>'
        "</OrderedGroup></ReadingOrder>"
        if reading_order
        else ""
    )
    path = write_page(
        tmp_path,
        order + '<TextRegion id="parent"><TextEquiv><Unicode>Child text</Unicode></TextEquiv>'
        '<TextRegion id="child"><TextLine id="line">'
        "<TextEquiv><Unicode>Child text</Unicode></TextEquiv></TextLine></TextRegion>"
        "</TextRegion>",
    )
    page, issues = read_page_xml(path)
    assert [(line.text, line.line_id) for line in page.reference_lines] == [("Child text", "line")]
    ignored = [i for i in issues if i.code == "region-text-ignored"]
    assert [i.location for i in ignored] == ["page.xml#parent"]


def test_region_text_fallback_applies_to_whole_subtree(tmp_path):
    path = write_page(
        tmp_path,
        # grandparent aggregates text provided two levels down: ignored
        '<TextRegion id="gp"><TextEquiv><Unicode>deep</Unicode></TextEquiv>'
        '<TextRegion id="p"><TextRegion id="c"><TextEquiv><Unicode>deep</Unicode></TextEquiv>'
        "</TextRegion></TextRegion></TextRegion>"
        # nested region without any text: the parent's text is the only reference
        '<TextRegion id="q"><TextEquiv><Unicode>only parent</Unicode></TextEquiv>'
        '<TextRegion id="empty"><TextLine id="blank"/></TextRegion></TextRegion>',
    )
    page, _ = read_page_xml(path)
    assert [(line.text, line.region_id) for line in page.reference_lines] == [
        ("deep", "c"),
        ("only parent", "q"),
    ]


def test_nested_regions_without_id_with_reading_order(tmp_path):
    path = write_page(
        tmp_path,
        "<ReadingOrder/>"
        "<TextRegion><TextLine><TextEquiv><Unicode>outer</Unicode></TextEquiv></TextLine>"
        "<TextRegion><TextLine><TextEquiv><Unicode>inner</Unicode></TextEquiv></TextLine>"
        "</TextRegion></TextRegion>",
    )
    page, issues = read_page_xml(path)
    assert [line.text for line in page.reference_lines] == ["outer", "inner"]
    assert "missing-id" in codes(issues, Severity.WARNING)


def test_region_text_used_when_lines_have_no_text(tmp_path):
    path = write_page(
        tmp_path,
        '<TextRegion id="r"><TextEquiv><Unicode>whole region</Unicode></TextEquiv>'
        '<TextLine id="l1"><TextEquiv><Unicode> </Unicode></TextEquiv></TextLine>'
        "</TextRegion>",
    )
    page, issues = read_page_xml(path)
    assert [(line.text, line.line_id) for line in page.reference_lines] == [("whole region", None)]
    assert "line-without-text" in codes(issues, Severity.WARNING)


def test_unreferenced_region_is_appended_with_warning(tmp_path):
    path = write_page(
        tmp_path,
        '<ReadingOrder><OrderedGroup id="g"><RegionRefIndexed index="0" regionRef="b"/>'
        '<RegionRefIndexed index="1" regionRef="ghost"/></OrderedGroup></ReadingOrder>'
        '<TextRegion id="a"><TextEquiv><Unicode>A</Unicode></TextEquiv></TextRegion>'
        '<TextRegion id="b"><TextEquiv><Unicode>B</Unicode></TextEquiv></TextRegion>',
    )
    page, issues = read_page_xml(path)
    assert [line.text for line in page.reference_lines] == ["B", "A"]
    assert set(codes(issues, Severity.WARNING)) == {
        "unknown-reading-order-ref",
        "region-not-in-reading-order",
    }


def test_regions_without_id_are_all_kept(tmp_path):
    path = write_page(
        tmp_path,
        '<ReadingOrder><OrderedGroup id="g"><RegionRefIndexed index="0" regionRef="b"/>'
        "</OrderedGroup></ReadingOrder>"
        "<TextRegion><TextEquiv><Unicode>A1</Unicode></TextEquiv></TextRegion>"
        "<TextRegion><TextEquiv><Unicode>A2</Unicode></TextEquiv></TextRegion>"
        '<TextRegion id="b"><TextEquiv><Unicode>B</Unicode></TextEquiv></TextRegion>',
    )
    page, issues = read_page_xml(path)
    assert [line.text for line in page.reference_lines] == ["B", "A1", "A2"]
    assert codes(issues, Severity.WARNING).count("missing-id") == 2


def test_duplicate_line_index_falls_back_to_document_order(tmp_path):
    path = write_page(
        tmp_path,
        '<TextRegion id="r">'
        '<TextLine id="l1" index="0"><TextEquiv><Unicode>first</Unicode></TextEquiv></TextLine>'
        '<TextLine id="l2" index="0"><TextEquiv><Unicode>second</Unicode></TextEquiv></TextLine>'
        "</TextRegion>",
    )
    page, issues = read_page_xml(path)
    assert [line.text for line in page.reference_lines] == ["first", "second"]
    assert sorted(codes(issues)) == ["inconsistent-line-index", "no-reading-order"]


def test_empty_page_warns(tmp_path):
    page, issues = read_page_xml(write_page(tmp_path, ""))
    assert page is not None
    assert page.reference_lines == ()
    assert "no-reference-text" in codes(issues, Severity.WARNING)


def test_duplicate_ids_warn(tmp_path):
    path = write_page(
        tmp_path,
        '<TextRegion id="x"><TextLine id="x"><TextEquiv><Unicode>t</Unicode></TextEquiv>'
        "</TextLine></TextRegion>",
    )
    _, issues = read_page_xml(path)
    assert "duplicate-id" in codes(issues, Severity.WARNING)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("<PcGts><Page", "xml-parse-error"),
        ("<html/>", "not-page-xml"),
        ('<PcGts xmlns="http://example.org/other"><Page/></PcGts>', "not-page-xml"),
        (
            '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"/>',
            "missing-page",
        ),
    ],
)
def test_unusable_files_are_errors(tmp_path, content, code):
    path = tmp_path / "bad.xml"
    path.write_text(content, encoding="utf-8")
    page, issues = read_page_xml(path, "s1")
    assert page is None
    assert codes(issues, Severity.ERROR) == [code]
    assert issues[0].sample_id == "s1"


def test_missing_page_size_is_error(tmp_path):
    page, issues = read_page_xml(write_page(tmp_path, "", size='imageWidth="100"'))
    assert page is None
    assert codes(issues) == ["missing-page-size"]


def test_malformed_points_are_errors(tmp_path):
    path = write_page(
        tmp_path,
        '<TextRegion id="r"><Coords points="1,2 3"/>'
        '<TextLine id="l"><TextEquiv><Unicode>t</Unicode></TextEquiv></TextLine></TextRegion>',
    )
    page, issues = read_page_xml(path)
    assert codes(issues, Severity.ERROR) == ["malformed-points"]
    assert page.regions[0].coords is None
