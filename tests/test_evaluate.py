"""Evaluator: scoring stored runs, failure policies, aggregates, GT pinning and the CLI.

Runs are produced by the fake engine (tests/fixtures/fake-engine) as a host process, against
synthetic PAGE XML whose reference text is chosen per test.
"""

import hashlib
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.engine import parse_engine_config
from ocrbench.evaluate import (
    EvalConfig,
    EvalConfigError,
    EvaluationError,
    evaluate_run,
    parse_eval_config,
    write_scores,
)
from ocrbench.runner import ProcessLauncher, create_run, execute, load_run

FAKE = [sys.executable, str(Path(__file__).parent / "fixtures" / "fake-engine" / "fake_engine.py")]


def page_xml(lines):
    body = "".join(
        f'<TextLine id="l{i}"><Coords points="0,0 60,0 60,10 0,10"/>'
        f"<TextEquiv><Unicode>{escape(text)}</Unicode></TextEquiv></TextLine>"
        for i, text in enumerate(lines)
    )
    return (
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        '<Page imageWidth="60" imageHeight="40"><TextRegion id="r1">'
        f'<Coords points="0,0 60,0 60,40 0,40"/>{body}</TextRegion></Page></PcGts>'
    )


def make_run(tmp_path, samples, *, execute_run=True):
    """``samples``: id -> (behavior, reference lines or raw GT text, unit, doc_type)."""
    data = tmp_path / "data"
    data.mkdir(parents=True)
    entries = []
    for sid, (_, lines, unit, doc_type) in samples.items():
        Image.new("L", (60, 40), 255).save(data / f"{sid}.png")
        gt = lines if isinstance(lines, str) else page_xml(lines)
        (data / f"{sid}.xml").write_text(gt, encoding="utf-8")
        entry = {"id": sid, "image": f"{sid}.png", "gt": f"{sid}.xml", "unit": unit}
        entries.append(entry | ({"doc_type": doc_type} if doc_type else {}))
    manifest = data / "manifest.json"
    manifest_data = {"format": "ocrbench.dataset-manifest", "version": 1, "name": "eval"}
    manifest.write_text(json.dumps(manifest_data | {"samples": entries}), encoding="utf-8")
    engine = parse_engine_config(
        {
            "name": "fake",
            "engine_version": "1",
            "image": "ocrbench-test/fake-engine:test-v1",
            "params": {"behaviors": {sid: s[0] for sid, s in samples.items()}},
            "timeouts": {"sample": 10, "startup": 10},
        }
    )
    run = create_run(engine, manifest, tmp_path / "runs", runtime="process")
    if execute_run:
        execute(run, ProcessLauncher(FAKE), poll_interval=0.02)
    return run


SAMPLES = {
    # The fake engine writes "# md\n\nText of *md*." as Markdown: the markup is not scored.
    "md": ("markdown", ["md", "Text of md."], "page", "newspaper"),
    # Native text "plain txt": one deleted character, one substituted word.
    "txt": ("text", ["plain text"], "paragraph", None),
    "trunc": ("truncated", ["trunc", "Text of trunc. and more"], "page", "book"),
    "failed": ("failed", ["lost words here"], "page", "newspaper"),
}


def by_id(scores):
    return {s["sample_id"]: s for s in scores["samples"]}


def group(scores, **key):
    return next(g for g in scores["aggregates"] if g["group"] == key)


def test_samples_are_scored_from_standardized_predictions(tmp_path):
    scores = evaluate_run(make_run(tmp_path, SAMPLES))
    samples = by_id(scores)
    md = samples["md"]
    assert (md["status"], md["scored_as"], md["chars"]["errors"], md["words"]["errors"]) == (
        "success",
        "prediction",
        0,
        0,
    )
    assert md["bag_of_words"]["f1"] == 1.0
    txt = samples["txt"]
    assert (txt["chars"]["deletions"], txt["chars"]["errors"], txt["chars"]["reference"]) == (
        1,
        1,
        10,
    )
    assert (txt["words"]["substitutions"], txt["words"]["errors"]) == (1, 1)
    trunc = samples["trunc"]  # scored as produced; the status is kept
    assert (trunc["status"], trunc["scored_as"]) == ("truncated", "prediction")
    assert (trunc["chars"]["deletions"], trunc["chars"]["errors"]) == (9, 9)
    assert (trunc["words"]["deletions"], trunc["words"]["errors"]) == (2, 2)
    failed = samples["failed"]  # empty-output default: all reference text is deleted
    assert (failed["status"], failed["reason"], failed["scored_as"]) == (
        "failure",
        "engine-error",
        "empty",
    )
    assert failed["chars"] == {
        "reference": 15,
        "hypothesis": 0,
        "substitutions": 0,
        "deletions": 15,
        "insertions": 0,
        "errors": 15,
        "rate": 1.0,
    }
    assert failed["bag_of_words"]["recall"] == 0.0
    assert failed["bag_of_words"]["precision"] is None
    assert all(s["gt_sha256"] for s in scores["samples"])
    # No prediction is created for the failure.
    assert not list((tmp_path / "runs").rglob("failed/attempt-1/prediction*"))


def test_micro_and_macro_averages_differ(tmp_path):
    scores = evaluate_run(make_run(tmp_path, SAMPLES))
    overall = group(scores)
    # md 0/14, txt 1/10, trunc 9/29, failed 15/15
    assert overall["chars"]["micro"]["errors"] == 25
    assert overall["chars"]["micro"]["reference"] == 68
    assert overall["chars"]["micro"]["rate"] == pytest.approx(25 / 68)
    macro = overall["chars"]["macro"]["rate"]
    assert macro == {
        "mean": pytest.approx((0 + 1 / 10 + 9 / 29 + 1) / 4),
        "samples": 4,
        "undefined": 0,
    }
    assert (overall["samples"], overall["scored"], overall["coverage"]) == (4, 4, 1.0)
    assert overall["statuses"] == {"success": 2, "truncated": 1, "failure": 1}
    assert overall["failure_reasons"] == {"engine-error": 1}


def test_strata_by_unit_and_document_type(tmp_path):
    scores = evaluate_run(make_run(tmp_path, SAMPLES))
    keys = [g["group"] for g in scores["aggregates"]]
    assert keys == [
        {},
        {"unit": "page"},
        {"unit": "page", "doc_type": "book"},
        {"unit": "page", "doc_type": "newspaper"},
        {"unit": "paragraph"},
        {"unit": "paragraph", "doc_type": None},
    ]
    news = group(scores, unit="page", doc_type="newspaper")
    assert news["samples"] == 2
    assert news["chars"]["micro"]["rate"] == pytest.approx(15 / 29)
    assert group(scores, unit="paragraph")["chars"]["micro"]["rate"] == pytest.approx(0.1)


def test_exclude_policy_reports_coverage(tmp_path):
    run = make_run(tmp_path, SAMPLES)
    scores = evaluate_run(run, EvalConfig(name="exclude", failure_policy="exclude"))
    failed = by_id(scores)["failed"]
    assert failed["scored_as"] == "excluded"
    assert failed["chars"] is None
    overall = group(scores)
    assert (overall["samples"], overall["scored"], overall["coverage"]) == (4, 3, 0.75)
    assert overall["chars"]["micro"]["rate"] == pytest.approx(10 / 53)
    assert overall["statuses"]["failure"] == 1
    assert scores["config"]["failure_policy"] == "exclude"
    default = evaluate_run(run)
    assert default["config_sha256"] != scores["config_sha256"]


def test_empty_reference_rates_are_undefined_but_counted(tmp_path):
    samples = {"ghost": ("text", [], "page", None), "md": SAMPLES["md"]}
    scores = evaluate_run(make_run(tmp_path, samples))
    ghost = by_id(scores)["ghost"]
    assert (ghost["chars"]["insertions"], ghost["chars"]["rate"]) == (11, None)  # "plain ghost"
    overall = group(scores)
    assert overall["chars"]["micro"]["errors"] == 11
    assert overall["chars"]["micro"]["rate"] == pytest.approx(11 / 14)
    assert overall["chars"]["macro"]["rate"] == {"mean": 0.0, "samples": 1, "undefined": 1}


def test_all_failed_with_exclude_has_no_scores(tmp_path):
    run = make_run(tmp_path, {"failed": SAMPLES["failed"]})
    overall = group(evaluate_run(run, EvalConfig(failure_policy="exclude")))
    assert (overall["scored"], overall["coverage"], overall["chars"]["micro"]) == (0, 0.0, None)
    assert overall["chars"]["macro"]["rate"]["mean"] is None


def test_scores_record_settings_and_identity(tmp_path):
    run = make_run(tmp_path, SAMPLES)
    scores = evaluate_run(run, EvalConfig(unicode="NFKC"))
    assert scores["format"] == "ocrbench.scores"
    assert scores["run_id"] == run.run_id
    assert (scores["version"], scores["metrics_version"]) == (2, 2)
    assert scores["rules"] == {
        "characters": "code-points",
        "whitespace": "collapse-to-space",
        "words": "whitespace-split",
        "reference_lines": "joined-by-space",
        "flexible_units": "reference-lines/prediction-lines",
        "flexible_groups": "consecutive-runs-any-order",
        "unicode": "NFKC",
    }
    assert scores["rapidfuzz_version"]
    first, second = write_scores(run, scores), write_scores(run, scores)
    assert (first.parent.name, second.parent.name) == ("1", "2")
    assert json.loads(first.read_text(encoding="utf-8")) == scores


def test_changed_gt_and_pending_samples_are_refused(tmp_path):
    run = make_run(tmp_path, SAMPLES)
    (tmp_path / "data" / "md.xml").write_text(page_xml(["md", "corrected"]), encoding="utf-8")
    with pytest.raises(EvaluationError, match="gt-changed"):
        evaluate_run(run)
    pending = make_run(tmp_path / "other", SAMPLES, execute_run=False)
    with pytest.raises(EvaluationError, match="4 samples are pending"):
        evaluate_run(pending)


def test_eval_config_is_validated():
    assert parse_eval_config({}) == EvalConfig()
    with pytest.raises(EvalConfigError) as error:
        parse_eval_config({"unicode": "NFD", "failure_policy": "skip", "cer": True})
    assert error.value.errors == [
        "unknown field 'cer'",
        "'unicode' must be one of ('NFC', 'NFKC')",
        "'failure_policy' must be one of ('empty-output', 'exclude')",
    ]


def test_cli_evaluate(tmp_path, capsys):
    run = make_run(tmp_path, SAMPLES)
    config = tmp_path / "exclude.toml"
    config.write_text('name = "exclude"\nfailure_policy = "exclude"\n', encoding="utf-8")
    assert main(["evaluate", str(run.path)]) == 0
    assert main(["evaluate", str(run.path), "--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "all" in out
    assert "failure policy exclude" in out
    stored = json.loads((run.path / "evaluations" / "2" / "scores.json").read_text("utf-8"))
    assert stored["config"]["name"] == "exclude"
    config.write_text('failure_policy = "skip"\n', encoding="utf-8")
    assert main(["evaluate", str(run.path), "--config", str(config)]) == 1
    assert "failure_policy" in capsys.readouterr().err
    assert not (run.path / "evaluations" / "3").exists()


@pytest.mark.parametrize("policy", ["empty-output", "exclude"])
def test_invalid_gt_is_refused_even_for_an_excluded_failure(tmp_path, policy):
    # The GT is malformed already when the run is created, so its hash is pinned and unchanged.
    samples = {"md": SAMPLES["md"], "failed": ("failed", "<PcGts>not PAGE", "page", None)}
    run = make_run(tmp_path, samples)
    assert run.samples[1].gt_sha256 is not None
    with pytest.raises(EvaluationError, match="failed: invalid GT"):
        evaluate_run(run, EvalConfig(failure_policy=policy))


def test_stored_prediction_is_revalidated_against_the_frozen_image_size(tmp_path):
    run = make_run(tmp_path, {"md": SAMPLES["md"]})
    stored = run.path / "samples" / "md" / "attempt-1" / "prediction.json"
    data = json.loads(stored.read_text(encoding="utf-8"))
    stored.write_text(json.dumps(data | {"image": {"width": 61, "height": 40}}), "utf-8")
    with pytest.raises(EvaluationError, match="image-size-mismatch"):
        evaluate_run(run)
    stored.write_text(json.dumps(data | {"image": {"width": 60, "height": 40}}), "utf-8")
    entry = by_id(evaluate_run(run))["md"]
    assert entry["prediction_sha256"] == hashlib.sha256(stored.read_bytes()).hexdigest()


def test_prediction_hash_is_recorded_only_for_scored_predictions(tmp_path):
    samples = by_id(evaluate_run(make_run(tmp_path, SAMPLES)))
    assert all(len(samples[sid]["prediction_sha256"]) == 64 for sid in ("md", "txt", "trunc"))
    assert samples["failed"]["prediction_sha256"] is None


def test_runs_created_without_doc_type_are_still_evaluated(tmp_path):
    run = make_run(tmp_path, SAMPLES)
    frozen = json.loads((run.path / "run.json").read_text(encoding="utf-8"))
    for sample in frozen["samples"]:
        del sample["doc_type"]  # run.json as written before #8
    (run.path / "run.json").write_text(json.dumps(frozen), encoding="utf-8")
    scores = evaluate_run(load_run(run.path))
    assert [g["group"] for g in scores["aggregates"]] == [
        {},
        {"unit": "page"},
        {"unit": "page", "doc_type": None},
        {"unit": "paragraph"},
        {"unit": "paragraph", "doc_type": None},
    ]


def test_flexible_scores_tolerate_reordered_output(tmp_path):
    samples = {
        # The prediction "md", "Text of md." has the GT lines in the opposite order.
        "md": ("markdown", ["Text of md.", "md"], "page", None),
        "failed": ("failed", ["lost words here"], "page", None),
    }
    scores = evaluate_run(make_run(tmp_path, samples))
    md, failed = by_id(scores)["md"], by_id(scores)["failed"]
    assert md["chars"]["errors"] > 0
    assert md["flexible_chars"] == {
        "reference": 14,
        "hypothesis": 14,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "errors": 0,
        "rate": 0.0,
        "lower_bound": 0,
        "exact": True,
        "method": "exhaustive",
        "groups": 2,
        "order_breaks": 1,
        "unmatched_reference_units": 0,
        "unmatched_hypothesis_units": 0,
    }
    assert md["flexible_words"]["errors"] == 0
    assert md["flexible_units"] == {"reference": 2, "hypothesis": 2, "anchored": 2}
    # A failure is scored as an empty prediction: every reference character is deleted.
    assert failed["flexible_chars"]["deletions"] == failed["chars"]["deletions"] == 15
    total = group(scores)["flexible_chars"]
    assert total["micro"]["errors"] == 15
    assert (total["micro"]["lower_bound"], total["micro"]["lower_bound_rate"]) == (15, 15 / 29)
    assert total["not_exact"] == 0
    assert total["macro"]["rate"]["mean"] == pytest.approx(0.5)
    excluded = evaluate_run(
        load_run(tmp_path / "runs" / scores["run_id"]), EvalConfig(failure_policy="exclude")
    )
    assert by_id(excluded)["failed"]["flexible_chars"] is None
