"""Host-side conversion of native text, Markdown and HTML, and the prediction CLI."""

import json
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.convert import CONVERTER_VERSION, convert_native, fragment_text, line_text
from ocrbench.issues import Severity
from ocrbench.prediction import validate_prediction

FIXTURES = Path(__file__).parent / "fixtures" / "predictions"


def convert(name_or_text, source_format):
    path = FIXTURES / name_or_text
    content = path.read_text(encoding="utf-8") if path.suffix else name_or_text
    prediction, issues = validate_prediction(
        convert_native(content, source_format, "s1"), sample_id="s1"
    )
    assert prediction is not None, [i.format() for i in issues]
    assert [i.code for i in issues if i.severity is not Severity.INFO] == []
    return prediction


def regions(prediction):
    return [(r.type, r.content) for r in prediction.regions]


def test_markdown_becomes_plain_text_and_typed_regions():
    prediction = convert("vlm_page.md", "markdown")
    assert regions(prediction) == [
        ("heading", "Die Zeitung"),
        ("text", "Erste Spalte mit x^2 und\nzweiter Zeile & ſchön\ndritte"),
        ("list-item", "1. Punkt eins"),
        ("list-item", "2. Punkt zwei"),
        ("list-item", "Unterpunkt"),  # a bullet marker is not page text
        ("formula", "E = mc^2"),
        ("table", "Jahr\tPreis\n1890\t5 Kr."),
        ("caption", "Tab. 1"),  # HTML embedded in Markdown uses the HTML rules
        ("table", "a\tb\nc\td"),
    ]
    assert prediction.text == "\n\n".join(text for _, text in regions(prediction))
    for markup in ("#", "*", "$", "|", "<", "&amp;", "Abbildung"):
        assert markup not in prediction.text
    assert prediction.conversion == {"from": "markdown", "version": CONVERTER_VERSION}
    assert prediction.image_size is None
    assert all(r.polygon is None and not r.lines for r in prediction.regions)


def test_html_becomes_plain_text_and_typed_regions():
    prediction = convert("vlm_page.html", "html")
    assert regions(prediction) == [
        ("heading", "Die Zeitung"),
        ("text", "Erste Spalte mit Umbruch\nzweite & ſchön"),
        ("text", "Neuer Absatz"),
        ("list-item", "3. drei"),
        ("list-item", "innen"),
        ("list-item", "4. vier"),
        ("text", "Loser Text"),
        ("caption", "Abb. 2"),
        ("formula", "E = mc^2"),
        ("table", "1890\t5 Kr.\n1891 neu\tinner"),
    ]
    for markup in ("<", "&amp;", "alert", "ignored", "Bild"):
        assert markup not in prediction.text


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("```\n  code  \n```", [("other", "  code  ")]),
        ("> quoted\n\n---\n\nafter", [("text", "quoted"), ("text", "after")]),
        ("5) five\n\n   second paragraph", [("list-item", "5) five\nsecond paragraph")]),
        ("Setext\n======", [("heading", "Setext")]),
        ("a \\*b\\* [link](http://x)", [("text", "a *b* link")]),
        ("", []),
        # inline HTML uses the HTML rules: hidden content is not page text, visible text is
        ("Text <script>alert(1)</script> next", [("text", "Text next")]),
        ("A <style>h1{display:none}</style> Z", [("text", "A Z")]),
        ("a <span>visible</span> *b*<br>c &amp; `x<y`", [("text", "a visible b\nc & x<y")]),
    ],
)
def test_markdown_rules(markdown, expected):
    assert regions(convert(markdown, "markdown")) == expected


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        # converter version 2: a tag with an underscore is markup, its text is kept
        ("<page_number>14</page_number>", [("text", "14")]),
        ("Seite <page_number>9/22</page_number> Ende", [("text", "Seite 9/22 Ende")]),
        (
            "<watermark>COPY</watermark> und <signature>J. Novák</signature>",
            [("text", "COPY und J. Novák")],
        ),
        ("| a | <page_number>3</page_number> |\n|---|---|\n| b | c |", [("table", "a\t3\nb\tc")]),
        ("`<page_number>` x", [("text", "<page_number> x")]),  # code stays verbatim
        ("a <b_c d>", [("text", "a <b_c d>")]),  # not a tag: attributes are not matched
        # an escaped or entity-encoded tag is literal text, not markup
        (r"\<page_number>14\</page_number>", [("text", "<page_number>14</page_number>")]),
        (
            "&lt;page_number&gt;14&lt;/page_number&gt;",
            [("text", "<page_number>14</page_number>")],
        ),
        (r"a \<x_y> <b>c</b>", [("text", "a <x_y> c")]),  # also next to real inline HTML
        ("&#60;page_number>7", [("text", "<page_number>7")]),
        # text between <img> and an explicit </img> is a description or caption: nothing
        ("<img>A drawing of a horse</img>", []),
        ("Vor <img>Bild *eines* Pferdes</img> nach", [("text", "Vor nach")]),
        ("<img src=x>after", [("text", "after")]),  # a void <img> still ends at once
        # a second </img> closes nothing and drops nothing
        (
            "Before <img>description</img> Important text </img> After",
            [("text", "Before Important text After")],
        ),
        (
            "text\n\n<img>\nmultiline\ndescription\n</img>\n\nnext",
            [
                ("text", "text"),
                ("text", "next"),
            ],
        ),
    ],
)
def test_markdown_semantic_tags_and_image_text(markdown, expected):
    assert regions(convert(markdown, "markdown")) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<p>A <img>desc</img> B</p>", "A B"),
        ("<p>A <img src='x.png'>B</p>", "A B"),
        ("<img>x <b>y</b></img>Z", "Z"),
        ("<p><page_number>7</page_number></p>", "7"),
        ("<p>a</img>b</p>", "ab"),  # a stray </img> drops nothing
        # an <img> already closed is not closed again by a later </img>
        (
            "<p>Before <img>description</img> Important text </img> After</p>",
            "Before Important text After",
        ),
        ("<p>A <img src='x.png'/> B</img> C</p>", "A B C"),
        ("<p><img>a</img> b <img>c</img> d</img> e</p>", "b d e"),
    ],
)
def test_html_image_text_and_underscore_tags(content, expected):
    assert fragment_text(content, "html") == expected


def test_converter_version_is_3():
    assert CONVERTER_VERSION == 3


def test_plain_text_is_kept_exactly():
    prediction = convert("  line one\n\n# not a heading\n", "text")
    assert prediction.text == "  line one\n\n# not a heading\n"
    assert prediction.regions == ()


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="unknown native format"):
        convert_native("x", "docx", "s1")


def test_cli_convert_and_validate(tmp_path, capsys):
    out = tmp_path / "pred.json"
    assert (
        main(
            [
                "prediction",
                "convert",
                str(FIXTURES / "vlm_page.md"),
                "--sample-id",
                "s1",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text(encoding="utf-8"))["regions"][0]["text"] == "Die Zeitung"
    assert main(["prediction", "validate", str(out), "--sample-id", "s1"]) == 0
    assert main(["prediction", "validate", str(out), "--sample-id", "other"]) == 1
    assert "sample-id-mismatch" in capsys.readouterr().err


def test_cli_validate_checks_image_size_and_writes_standardized_output(tmp_path, capsys):
    image = tmp_path / "page.png"
    Image.new("L", (400, 300), 255).save(image)
    out = tmp_path / "std.json"
    source = FIXTURES / "newspaper_columns.json"
    args = ["prediction", "validate", str(source), "--image", str(image), "-o", str(out)]
    assert main(args) == 0
    assert "points-out-of-bounds" in capsys.readouterr().err
    clipped = json.loads(out.read_text(encoding="utf-8"))["regions"][2]["lines"][1]["polygon"]
    assert max(x for x, _ in clipped) == 400
    Image.new("L", (200, 150), 255).save(image)
    assert main(["prediction", "validate", str(source), "--image", str(image)]) == 1
    assert "image-size-mismatch" in capsys.readouterr().err


def test_cli_reports_unreadable_inputs_without_traceback(tmp_path, capsys):
    missing = str(tmp_path / "missing.md")
    assert main(["prediction", "convert", missing, "--sample-id", "s1"]) == 1
    assert "cannot read" in capsys.readouterr().err
    source = str(FIXTURES / "newspaper_columns.json")
    for image in (tmp_path / "missing.jpg", FIXTURES / "vlm_page.md"):
        assert main(["prediction", "validate", source, "--image", str(image)]) == 1
        assert "cannot read image" in capsys.readouterr().err
    out = str(tmp_path / "no-dir" / "pred.json")
    assert (
        main(
            ["prediction", "convert", str(FIXTURES / "vlm_page.md"), "--sample-id", "s1", "-o", out]
        )
        == 1
    )
    assert "cannot write" in capsys.readouterr().err


def test_html_and_markdown_inline_html_agree():
    html = convert("<p>A <script>x()</script> <span>visible</span>  Z</p>", "html")
    markdown = convert("A <script>x()</script> <span>visible</span>  Z", "markdown")
    assert regions(html) == regions(markdown) == [("text", "A visible Z")]


@pytest.mark.parametrize(
    ("content", "source_format", "expected"),
    [
        ("<p>Die <i>alte</i> Zeit<br>und &amp; mehr</p>", "html", "Die alte Zeit\nund & mehr"),
        ("<h2>Kapitel 1</h2><p>Text</p>", "html", "Kapitel 1\nText"),
        ("<ol><li>eins</li><li>zwei</li></ol>", "html", "1. eins\n2. zwei"),
        (
            "<table><tr><th>a</th><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>",
            "html",
            "a\tb\nc\td",
        ),
        ('<math display="block">x^2</math>', "html", "x^2"),
        ('<img src="a.png" alt="Logo">', "html", ""),
        ("# Titel\n\n*kursiv* und [Link](https://x)", "markdown", "Titel\nkursiv und Link"),
    ],
)
def test_fragment_text_uses_the_document_rules(content, source_format, expected):
    assert fragment_text(content, source_format) == expected


def test_fragment_text_rejects_unknown_formats():
    with pytest.raises(ValueError, match="markup format"):
        fragment_text("x", "text")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("1912.", [("list-item", "1912.")]),  # an empty ordered item prints its number
        ("3)", [("list-item", "3)")]),
        ("7. \n\n8. text", [("list-item", "7."), ("list-item", "8. text")]),
        ("1. - nested", [("list-item", "1."), ("list-item", "nested")]),
        ("-", []),  # an empty bullet item has no text
    ],
)
def test_empty_ordered_markdown_items_keep_their_number(content, expected):
    blocks = convert_native(content, "markdown", "s1").get("regions", [])
    assert [(r["type"], r["text"]) for r in blocks] == expected


def test_empty_ordered_items_keep_their_number_in_markdown_and_html():
    assert (
        fragment_text("1912.", "markdown")
        == fragment_text('<ol start="1912"><li></li></ol>', "html").strip()
    )


@pytest.mark.parametrize(
    ("content", "source_format", "expected"),
    [
        ("H<sub>2</sub>O &amp; <b>salt</b>", "html", "H2O & salt"),
        ("one<br>two<br/>three", "html", "one two three"),
        ("<p>a</p><p>b</p>", "html", "a b"),
        ("  spaced   out  ", "html", "spaced out"),
        ("<b></b>", "html", ""),
        ("*a*  \nb", "markdown", "a b"),
    ],
)
def test_line_text_keeps_one_line(content, source_format, expected):
    assert line_text(content, source_format) == expected
