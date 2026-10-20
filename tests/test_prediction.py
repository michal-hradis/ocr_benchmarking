"""Prediction schema and validator: positioned, text-only and malformed predictions."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ocrbench.issues import Severity
from ocrbench.prediction import (
    load_prediction,
    prediction_to_json,
    schema,
    validate_prediction,
)

FIXTURES = Path(__file__).parent / "fixtures" / "predictions"


def codes(issues, severity=None):
    return sorted(i.code for i in issues if severity is None or i.severity is severity)


def newspaper():
    return json.loads((FIXTURES / "newspaper_columns.json").read_text(encoding="utf-8"))


def minimal(**fields):
    return {"format": "ocrbench.prediction", "version": 1, "sample_id": "s1", "text": "", **fields}


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_schema_is_a_valid_json_schema():
    Draft202012Validator.check_schema(schema())


def test_positioned_newspaper_columns():
    prediction, issues = load_prediction(
        FIXTURES / "newspaper_columns.json", sample_id="columns_2019", image_size=(400, 300)
    )
    assert prediction is not None, [i.format() for i in issues]
    assert prediction.explicit_reading_order
    assert [r.id for r in prediction.regions] == [
        "r_head",
        "r_left",
        "r_right",
        "r_formula",
        "r_photo",
    ]
    left = prediction.regions[1]
    assert [(line.id, line.text, line.type) for line in left.lines] == [
        ("l_left_1", "erste Zeile", "handwritten"),
        ("l_left_2", "zweite Zeile", None),
    ]
    right = prediction.regions[2]
    assert right.lines[0].baseline == ((210, 76), (390, 76))
    assert right.lines[0].confidence == 0.93
    assert right.lines[1].polygon == ((210, 85), (400, 85), (400, 105), (210, 105))  # clipped
    assert prediction.regions[3].content == "E = mc^2"  # a region with its own text
    assert prediction.regions[4].content is None  # a figure without text
    assert prediction.text.startswith("Die Zeitung\n\nerste Zeile")
    assert codes(issues) == ["points-out-of-bounds"]
    assert issues[0].severity is Severity.WARNING
    assert issues[0].location == "newspaper_columns.json#/regions/0/lines/1/polygon"
    assert "2 point(s) in 1 element(s)" in issues[0].message


def test_text_only_vlm_has_no_geometry():
    prediction, issues = load_prediction(FIXTURES / "text_only_vlm.json")
    assert prediction is not None
    assert prediction.text == "Příliš žluťoučký\nkůň.\n"  # page text kept exactly
    assert prediction.regions == ()
    assert prediction.image_size is None
    assert not prediction.explicit_reading_order
    assert issues == []


def test_regions_without_positions_and_without_reading_order():
    data = minimal(
        text="A\n\nB",
        regions=[{"id": "a", "text": "A"}, {"id": "b", "type": "caption", "text": "B"}],
    )
    prediction, issues = validate_prediction(data)
    assert issues == []
    assert not prediction.explicit_reading_order
    assert [r.id for r in prediction.regions] == ["a", "b"]
    assert "reading_order" not in prediction_to_json(prediction)


def test_standardized_output_round_trips():
    prediction, _ = validate_prediction(newspaper(), image_size=(400, 300))
    standardized = prediction_to_json(prediction)
    again, issues = validate_prediction(copy.deepcopy(standardized), image_size=(400, 300))
    assert issues == []  # clipping is not reported twice
    assert again == prediction
    assert prediction_to_json(again) == standardized


def test_whitespace_only_differences_keep_page_text():
    data = minimal(
        text="  a b  c ", regions=[{"id": "r", "lines": [{"text": "a"}, {"text": "b c"}]}]
    )
    prediction, issues = validate_prediction(data)
    assert issues == []
    assert prediction.text == "  a b  c "


def test_line_text_has_priority_over_disagreeing_page_text():
    data = minimal(
        text="ab",
        regions=[
            {"id": "r1", "lines": [{"text": "a"}]},
            {"id": "r2", "text": "b"},
        ],
    )
    prediction, issues = validate_prediction(data)
    assert prediction.text == "a\n\nb"
    assert codes(issues) == ["text-mismatch"]
    assert issues[0].severity is Severity.WARNING


def test_region_text_is_ignored_when_region_has_lines():
    data = minimal(text="line", regions=[{"id": "r", "text": "other", "lines": [{"text": "line"}]}])
    prediction, issues = validate_prediction(data)
    assert prediction.regions[0].content == "line"
    assert prediction.regions[0].text is None
    assert codes(issues) == ["region-text-ignored"]


def test_reading_order_appends_missing_regions_with_warning():
    data = minimal(
        text="b\n\na\n\nc",
        regions=[{"id": "a", "text": "a"}, {"id": "b", "text": "b"}, {"id": "c", "text": "c"}],
        reading_order=["b"],
    )
    prediction, issues = validate_prediction(data)
    assert [r.id for r in prediction.regions] == ["b", "a", "c"]
    assert codes(issues) == ["region-not-in-reading-order"]


def test_degenerate_polygon_and_unknown_type_are_not_errors():
    data = minimal(
        image={"width": 10, "height": 10},
        text="t",
        regions=[
            {"id": "r", "type": "marginalia", "polygon": [[1, 1], [5, 1], [9, 1]], "text": "t"}
        ],
    )
    prediction, issues = validate_prediction(data)
    assert prediction is not None
    assert codes(issues, Severity.WARNING) == ["degenerate-polygon"]
    assert codes(issues, Severity.INFO) == ["unknown-region-type"]


def test_empty_prediction_is_valid():
    prediction, issues = validate_prediction(minimal())
    assert prediction is not None
    assert codes(issues) == ["empty-prediction"]


@pytest.mark.parametrize(
    ("change", "code"),
    [
        (lambda d: d.pop("text"), "schema-violation"),
        (lambda d: d.update(version=2), "schema-violation"),
        (lambda d: d.update(unexpected=1), "schema-violation"),
        (lambda d: d["regions"][0].update(polygon=[[0, 0], [1, 1]]), "schema-violation"),
        (
            lambda d: d["regions"][0]["lines"][0].update(baseline=[[0, "1"], [1, 1]]),
            "schema-violation",
        ),
        (lambda d: d["regions"][0]["lines"][0].pop("text"), "schema-violation"),
        (lambda d: d["regions"][1].update(id="r_right"), "duplicate-id"),
        (lambda d: d["regions"][2]["lines"][0].update(id="l_head"), "duplicate-id"),
        (lambda d: d["reading_order"].append("ghost"), "unknown-reading-order-ref"),
        (lambda d: d["reading_order"].append("r_head"), "duplicate-reading-order-ref"),
        (lambda d: d.pop("image"), "missing-image-size"),
        (lambda d: d.update(image={"width": 800, "height": 600}), "image-size-mismatch"),
        (lambda d: d.update(sample_id="other"), "sample-id-mismatch"),
        (lambda d: d["regions"][0].update(text="\ud800"), "invalid-unicode"),
        (
            lambda d: d["regions"][1].update(polygon=box(0, 0, float("inf"), 5)),
            "invalid-coordinate",
        ),
    ],
)
def test_invalid_predictions_are_rejected(change, code):
    data = newspaper()
    change(data)
    prediction, issues = validate_prediction(
        data, sample_id="columns_2019", image_size=(400, 300), location="p.json"
    )
    assert prediction is None  # never usable as a successful OCR result
    assert code in codes(issues, Severity.ERROR)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"{", "prediction-json-error"),
        (b'{"x": NaN}', "prediction-json-error"),
        (b"\xff\xfe", "prediction-unreadable"),
        (b"[]", "schema-violation"),
    ],
)
def test_unusable_files_are_errors(tmp_path, content, code):
    path = tmp_path / "p.json"
    path.write_bytes(content)
    prediction, issues = load_prediction(path, sample_id="s1")
    assert prediction is None
    assert codes(issues) == [code]
    assert issues[0].sample_id == "s1"
