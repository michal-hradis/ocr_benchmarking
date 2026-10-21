"""Reports: provenance, aggregates, failure listings, staleness, comparability and escaping.

Runs are produced by the fake engine (tests/fixtures/fake-engine) as a host process and scored
with ocrbench.evaluate; the expected numbers are the hand-computed ones of tests/test_evaluate.py.
"""

import csv
import json
import re
import shutil
import sys
from html import escape
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.evaluate import EvalConfig, evaluate_run, write_scores
from ocrbench.report import ReportError, diff_html, load_evaluation, write_report
from ocrbench.runner import ProcessLauncher, create_run, execute
from test_evaluate import SAMPLES, page_xml

FAKE = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake-engine" / "fake_engine.py")]


def make_dataset(tmp_path, samples):
    """``samples``: id -> (behavior, reference lines, unit, doc_type); returns the manifest."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    entries = []
    for sid, (_, lines, unit, doc_type) in samples.items():
        Image.new("L", (60, 40), 255).save(data / f"{sid}.png")
        (data / f"{sid}.xml").write_text(page_xml(lines), encoding="utf-8")
        entry = {"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml", "unit": unit}
        entries.append(entry | ({"doc_type": doc_type} if doc_type else {}))
    manifest = data / "manifest.json"
    head = {"format": "ocrbench.dataset-manifest", "version": 1, "name": "eval"}
    manifest.write_text(json.dumps(head | {"samples": entries}), encoding="utf-8")
    return manifest


def make_run(tmp_path, manifest, run_id, behaviors, *, model=None, config=None):
    """Run the fake engine with per-sample ``behaviors`` and store one evaluation."""
    engine = parse_engine_config(
        {
            "name": "fake",
            "engine_version": "1",
            "image": "ocrbench-test/fake-engine:test-v1",
            "params": {"behaviors": behaviors},
            "timeouts": {"sample": 10, "startup": 10},
        }
        | ({"model": model} if model else {})
    )
    run = create_run(engine, manifest, tmp_path / "runs", run_id=run_id, runtime="process")
    execute(run, ProcessLauncher(FAKE), poll_interval=0.02)
    write_scores(run, evaluate_run(run, config or EvalConfig()))
    return run


@pytest.fixture
def two_runs(tmp_path):
    manifest = make_dataset(tmp_path, SAMPLES)
    first = make_run(tmp_path, manifest, "a", {sid: s[0] for sid, s in SAMPLES.items()})
    second = make_run(tmp_path, manifest, "b", dict.fromkeys(SAMPLES, "markdown"), model="m@2")
    return manifest, first, second


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_report_compares_runs_with_provenance_and_stored_scores(tmp_path, two_runs):
    _, first, second = two_runs
    out = tmp_path / "report"
    report = write_report([load_evaluation(first.path), load_evaluation(second.path)], out)
    assert sorted(p.name for p in out.iterdir()) == [
        "aggregates.csv",
        "index.html",
        "report.json",
        "samples.csv",
    ]
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report
    assert report["evaluation"]["config"]["failure_policy"] == "empty-output"
    assert report["evaluation"]["metrics_version"] == 2
    run_a, run_b = report["runs"]
    assert (run_a["run_id"], run_b["run_id"]) == ("a", "b")
    assert run_b["engine"] == {
        "name": "fake",
        "engine_version": "1",
        "model": "m@2",
        "image": "ocrbench-test/fake-engine:test-v1",
    }
    assert run_a["statuses"] == {"success": 2, "truncated": 1, "partial": 0, "failure": 1}
    assert run_a["evaluation"].endswith("evaluations/1")
    assert len(run_a["scores_sha256"]) == 64
    assert run_a["engine_config_sha256"]
    # The aggregates are the stored ones, per run.
    stored = json.loads((first.path / "evaluations/1/scores.json").read_text("utf-8"))
    assert [g for g in report["aggregates"] if g["run_id"] == "a"] == [
        {"run_id": "a", **g} for g in stored["aggregates"]
    ]

    rows = read_csv(out / "aggregates.csv")
    overall = next(r for r in rows if r["run_id"] == "a" and r["group"] == "all")
    assert (overall["cer_micro_errors"], overall["cer_micro_reference"]) == ("25", "68")
    assert float(overall["cer_micro_rate"]) == pytest.approx(25 / 68)
    assert (overall["samples"], overall["scored"], overall["failure"], overall["truncated"]) == (
        "4",
        "4",
        "1",
        "1",
    )
    assert overall["failure_policy"] == "empty-output"
    # Groups without a document type keep an empty doc_type.
    paragraph = [r for r in rows if r["run_id"] == "a" and r["unit"] == "paragraph"]
    assert [(r["group"], r["doc_type"]) for r in paragraph] == [("unit", ""), ("doc_type", "")]
    assert len(rows) == 2 * len(stored["aggregates"])

    samples = read_csv(out / "samples.csv")
    assert len(samples) == 8
    failed = next(r for r in samples if (r["run_id"], r["sample_id"]) == ("a", "failed"))
    assert (failed["status"], failed["reason"], failed["scored_as"]) == (
        "failure",
        "engine-error",
        "empty",
    )
    assert (failed["cer_rate"], failed["cer_errors"], failed["bow_precision"]) == ("1.0", "15", "")
    assert failed["model"] == ""


def test_every_non_success_sample_is_listed_under_both_policies(tmp_path):
    manifest = make_dataset(tmp_path, SAMPLES)
    behaviors = {sid: s[0] for sid, s in SAMPLES.items()}
    for policy in ("empty-output", "exclude"):
        run = make_run(
            tmp_path / policy,
            manifest,
            policy,
            behaviors,
            config=EvalConfig(policy, failure_policy=policy),
        )
        out = tmp_path / f"report-{policy}"
        report = write_report([load_evaluation(run.path)], out)
        listed = {(i["sample_id"], i["status"], i["reason"]) for i in report["non_success"]}
        assert listed == {("failed", "failure", "engine-error"), ("trunc", "truncated", None)}
        messages = {i["sample_id"]: i["message"] for i in report["non_success"]}
        assert messages == {"failed": "unsupported page", "trunc": "output token limit reached"}
        html = (out / "index.html").read_text(encoding="utf-8")
        assert "unsupported page" in html
        assert "output token limit reached" in html
        assert "attempt-1/record.json" in html
        assert "1 truncated" in html
        assert "1 failure" in html
        overall = next(r for r in read_csv(out / "aggregates.csv") if r["group"] == "all")
        if policy == "exclude":
            assert "not the primary score" in html
            assert (overall["scored"], overall["coverage"]) == ("3", "0.75")
            assert (
                next(i for i in report["non_success"] if i["sample_id"] == "failed")["scored_as"]
                == "excluded"
            )
        else:
            assert "not the primary score" not in html
            assert overall["coverage"] == "1.0"


def test_a_run_without_any_success_is_reported(tmp_path):
    samples = {"failed": SAMPLES["failed"], "other": ("failed", ["more"], "page", None)}
    manifest = make_dataset(tmp_path, samples)
    run = make_run(
        tmp_path,
        manifest,
        "none",
        dict.fromkeys(samples, "failed"),
        config=EvalConfig(failure_policy="exclude"),
    )
    out = tmp_path / "report"
    report = write_report([load_evaluation(run.path)], out, inspect=True)
    assert len(report["non_success"]) == 2
    overall = report["aggregates"][0]
    assert (overall["scored"], overall["chars"]["micro"]) == (0, None)
    assert (
        next(r for r in read_csv(out / "aggregates.csv") if r["group"] == "all")["cer_micro_rate"]
        == ""
    )
    page = (out / "samples/none/failed.html").read_text(encoding="utf-8")
    assert "Excluded by the failure policy" in page
    assert "lost words here" in page


def test_inspection_pages_are_optional_and_show_differences(tmp_path, two_runs):
    _, first, _ = two_runs
    write_report([load_evaluation(first.path)], tmp_path / "plain")
    assert not (tmp_path / "plain" / "samples").exists()
    out = tmp_path / "inspected"
    write_report([load_evaluation(first.path)], out, inspect=True)
    assert sorted(p.name for p in (out / "samples" / "a").iterdir()) == [
        "failed.html",
        "md.html",
        "trunc.html",
        "txt.html",
    ]
    txt = (out / "samples/a/txt.html").read_text(encoding="utf-8")
    assert "plain t<del>e</del>xt" in txt  # GT "plain text", prediction "plain txt"
    failed = (out / "samples/a/failed.html").read_text(encoding="utf-8")
    assert "scored as an empty prediction" in failed
    assert "<del>lost words here</del>" in failed
    assert "Unmatched GT lines (deleted): 1" in failed
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'href="samples/a/txt.html"' in index


def test_flexible_groups_are_shown_from_stored_line_ranges(tmp_path):
    samples = {"md": ("markdown", ["Text of md.", "md"], "page", None)}
    manifest = make_dataset(tmp_path, samples)
    run = make_run(tmp_path, manifest, "flex", {"md": "markdown"})
    out = tmp_path / "report"
    write_report([load_evaluation(run.path)], out, inspect=True)
    page = (out / "samples/flex/md.html").read_text(encoding="utf-8")
    # group_lines [[1, 2, 0, 1], [0, 1, 2, 3]]: both groups match exactly, in prediction order.
    assert "1 order breaks" in page
    rows = page.split("<h2>Flexible groups (fCER)</h2>")[1]
    assert rows.index(">md<") < rows.index(">Text of md.<")
    assert "<del>" not in rows
    assert "<ins>" not in rows
    assert "Unmatched GT lines (deleted): 0" in rows
    assert "Unmatched prediction lines (inserted): 0" in rows


def test_untrusted_text_is_escaped(tmp_path):
    samples = {
        "txt": ("prediction", ["<b>bold</b> & co"], "page", None),
        "failed": SAMPLES["failed"],
    }
    manifest = make_dataset(tmp_path, samples)
    engine_out = {"txt": "prediction", "failed": "failed"}
    run = make_run(tmp_path, manifest, "x", engine_out)
    # Replace the stored prediction and the engine message with markup, then re-evaluate.
    attempt = run.path / "samples" / "txt" / "attempt-1"
    script = "<script>alert(1)</script> <img src=x onerror=alert(2)>"
    prediction = {"format": "ocrbench.prediction", "version": 1, "sample_id": "txt", "text": script}
    (attempt / "prediction.json").write_text(json.dumps(prediction), encoding="utf-8")
    record_path = run.path / "samples" / "failed" / "attempt-1" / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["message"] = '"><script>alert(3)</script>'
    record_path.write_text(json.dumps(record), encoding="utf-8")
    write_scores(run, evaluate_run(run))
    out = tmp_path / "report"
    write_report([load_evaluation(run.path)], out, inspect=True)
    for page in out.rglob("*.html"):
        html = page.read_text(encoding="utf-8")
        for markup in ("<script", "<img", "<b>bold"):
            assert markup not in html, page
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "&quot;&gt;&lt;script&gt;alert(3)&lt;/script&gt;" in index
    txt = (out / "samples/x/txt.html").read_text(encoding="utf-8")
    ordered = txt.split("<h2>Ordered differences</h2>")[1].split("<h2>")[0]
    # Both sides of the diff are escaped: drop deletions to get the prediction, and vice versa.
    predicted = re.sub(r"</?ins>", "", re.sub(r"<del>.*?</del>", "", ordered))
    referenced = re.sub(r"</?del>", "", re.sub(r"<ins>.*?</ins>", "", ordered))
    assert escape(script) in predicted
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; co" in referenced
    assert diff_html("a<b", "a>b") == "a<del>&lt;</del><ins>&gt;</ins>b"


def test_stale_evaluations_are_refused(tmp_path, two_runs):
    manifest, first, second = two_runs
    # A newer attempt than the evaluated one (as after 'run resume --retry-failed').
    samples = first.path / "samples" / "failed"
    shutil.copytree(samples / "attempt-1", samples / "attempt-2")
    record = json.loads((samples / "attempt-2" / "record.json").read_text("utf-8"))
    (samples / "attempt-2" / "record.json").write_text(json.dumps(record | {"attempt": 2}))
    with pytest.raises(
        ReportError, match="failed: the current attempt 2 .* is not the evaluated attempt 1"
    ):
        load_evaluation(first.path)
    write_scores(first, evaluate_run(first))  # a new evaluation is current again
    assert load_evaluation(first.path).path.name == "2"
    with pytest.raises(ReportError, match="is not the evaluated attempt"):
        load_evaluation(first.path / "evaluations" / "1")  # an explicitly chosen old one
    # A changed stored prediction.
    stored = second.path / "samples" / "md" / "attempt-1" / "prediction.json"
    stored.write_text(stored.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ReportError, match="md: the stored prediction changed"):
        load_evaluation(second.path)
    # Changed GT.
    (tmp_path / "data" / "txt.xml").write_text(page_xml(["corrected"]), encoding="utf-8")
    with pytest.raises(ReportError, match="gt-changed"):
        load_evaluation(first.path)


def test_missing_old_or_modified_evaluations_are_refused(tmp_path, two_runs):
    _, first, second = two_runs
    with pytest.raises(ReportError, match="neither a run nor an evaluation"):
        load_evaluation(tmp_path)
    shutil.rmtree(first.path / "evaluations")
    with pytest.raises(ReportError, match="has no evaluation; run 'ocrbench evaluate"):
        load_evaluation(first.path)
    scores_path = second.path / "evaluations" / "1" / "scores.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    scores_path.write_text(json.dumps(scores | {"version": 1}), encoding="utf-8")
    with pytest.raises(ReportError, match="needs version 2"):
        load_evaluation(second.path)
    scores_path.write_text(json.dumps(scores), encoding="utf-8")
    frozen = second.path / "run.json"
    frozen.write_text(frozen.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ReportError, match="run.json changed after the evaluation"):
        load_evaluation(second.path)


def test_evaluations_that_differ_are_not_compared(tmp_path, two_runs):
    manifest, first, second = two_runs
    out = tmp_path / "report"
    write_scores(second, evaluate_run(second, EvalConfig(failure_policy="exclude")))
    with pytest.raises(ReportError, match="b: config_sha256"):
        write_report([load_evaluation(first.path), load_evaluation(second.path)], out)
    with pytest.raises(ReportError, match="given more than once"):
        write_report([load_evaluation(first.path), load_evaluation(first.path)], out)
    other = make_dataset(tmp_path / "other", {"md": SAMPLES["md"]})
    third = make_run(tmp_path, other, "c", {"md": "markdown"})
    with pytest.raises(ReportError, match="c: the samples or their GT differ"):
        write_report([load_evaluation(first.path), load_evaluation(third.path)], out)
    assert not out.exists()
    out.mkdir()
    with pytest.raises(ReportError, match="already exists"):
        write_report([load_evaluation(first.path)], out)


def test_cli_report_and_failure_logging(tmp_path, two_runs, capsys):
    _, first, second = two_runs
    assert main(["evaluate", str(first.path)]) == 0
    err = capsys.readouterr().err
    assert "failed: failure (engine-error), scored as empty" in err
    assert "trunc: truncated, scored as prediction" in err
    out = tmp_path / "report"
    args = ["report", str(first.path), str(second.path / "evaluations" / "1"), "-o", str(out)]
    assert main([*args, "--inspect"]) == 0
    captured = capsys.readouterr()
    assert "a: failed: failure (engine-error)" in captured.err
    assert "failure policy empty-output" in captured.out
    assert (out / "index.html").is_file()
    assert (out / "samples" / "b" / "md.html").is_file()
    assert main(args) == 1
    assert "already exists" in capsys.readouterr().err
