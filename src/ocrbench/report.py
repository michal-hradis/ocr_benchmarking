"""Reports: compare evaluated runs from their stored scores, without rescoring or rerunning OCR.

The reporter reads the scores.json of one evaluation per run, checks that it still describes the
run (run.json, the newest attempt and the scored prediction of every sample, the frozen GT) and
that all runs were scored alike (metrics version, evaluation configuration, samples and GT), and
writes report.json, aggregates.csv, samples.csv and index.html into a new directory. Optional
inspection pages show GT versus prediction differences; they read the GT and the stored
prediction on the host, never anything an engine container sees. Every non-``success`` sample is
listed with its status and reason. All OCR text and engine messages are escaped. The format is
documented in docs/REPORTS.md.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rapidfuzz.distance import Levenshtein

import ocrbench
from ocrbench.alignment import prediction_lines
from ocrbench.evaluate import (
    SCORES_FORMAT,
    SCORES_VERSION,
    EvaluationError,
    reference_lines,
    stored_prediction,
)
from ocrbench.manifest import sha256_file
from ocrbench.metrics import normalize
from ocrbench.runner import STATUSES, Run, RunnerError, latest_record, load_run, verify_ground_truth

REPORT_FORMAT = "ocrbench.report"
REPORT_VERSION = 1


class ReportError(RuntimeError):
    """The evaluations cannot be reported: missing, stale, or not comparable."""


@dataclass(frozen=True)
class Evaluation:
    run: Run
    path: Path  # runs/<id>/evaluations/<n>
    scores: dict[str, Any]
    scores_sha256: str
    run_json: dict[str, Any]
    records: dict[str, dict[str, Any]]  # the newest execution record of every sample


def _read_json(path: Path) -> Any:
    def reject(name: str) -> Any:
        raise ValueError(f"{name} is not standard JSON")

    with open(path, encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject)


def _evaluation_dir(path: Path) -> Path:
    """An evaluation directory as given, or the newest evaluation of a run directory."""
    path = Path(path).resolve()
    if (path / "scores.json").is_file():
        return path
    if not (path / "run.json").is_file():
        raise ReportError(f"{path} is neither a run nor an evaluation directory")
    root = path / "evaluations"
    numbered = [p for p in root.iterdir() if p.name.isdigit()] if root.is_dir() else []
    numbered = [p for p in numbered if (p / "scores.json").is_file()]
    if not numbered:
        raise ReportError(f"{path} has no evaluation; run 'ocrbench evaluate {path}' first")
    return max(numbered, key=lambda p: int(p.name))


def load_evaluation(path: Path) -> Evaluation:
    """Load an evaluation and refuse it when it no longer describes the run it scored."""
    directory = _evaluation_dir(path)
    run_dir = directory.parent.parent
    try:
        run = load_run(run_dir)
        run_json = _read_json(run_dir / "run.json")
        scores = _read_json(directory / "scores.json")
    except (RunnerError, OSError, ValueError) as error:
        raise ReportError(f"{directory}: cannot read the evaluation: {error}") from error
    if (scores.get("format"), scores.get("version")) != (SCORES_FORMAT, SCORES_VERSION):
        raise ReportError(
            f"{directory}: scores are {scores.get('format')} version {scores.get('version')},"
            f" the report needs version {SCORES_VERSION}; run 'ocrbench evaluate {run_dir}'"
        )
    problems = []
    if scores["run_id"] != run.run_id:
        problems.append(f"scores belong to run {scores['run_id']!r}, not {run.run_id!r}")
    if scores["run_sha256"] != sha256_file(run_dir / "run.json"):
        problems.append("run.json changed after the evaluation")
    problems += [issue.format() for issue in verify_ground_truth(run)]
    entries = {entry["sample_id"]: entry for entry in scores["samples"]}
    if list(entries) != [s.id for s in run.samples]:
        problems.append("the evaluated samples differ from the samples of the run")
    records = {}
    for sample in run.samples:
        record, entry = latest_record(run, sample.id), entries.get(sample.id)
        if record is None:
            problems.append(f"{sample.id}: no execution record")
            continue
        if entry is None:
            continue
        records[sample.id] = record
        if (record["attempt"], record["status"]) != (entry["attempt"], entry["status"]):
            problems.append(
                f"{sample.id}: the current attempt {record['attempt']} ({record['status']}) is not"
                f" the evaluated attempt {entry['attempt']} ({entry['status']})"
            )
        elif entry["prediction_sha256"] is not None:
            stored = run_dir / "samples" / sample.id / f"attempt-{record['attempt']}"
            try:
                digest = sha256_file(stored / record["prediction"])
            except (OSError, TypeError):
                digest = None
            if digest != entry["prediction_sha256"]:
                problems.append(f"{sample.id}: the stored prediction changed after the evaluation")
    if problems:
        raise ReportError(
            f"{directory} is stale:\n"
            + "\n".join(problems)
            + f"\nrun 'ocrbench evaluate {run_dir}'"
        )
    return Evaluation(
        run, directory, scores, sha256_file(directory / "scores.json"), run_json, records
    )


def check_comparable(evaluations: list[Evaluation]) -> None:
    """Refuse to put evaluations side by side that were not scored alike on the same samples."""
    if not evaluations:
        raise ReportError("no evaluation to report")
    ids = [e.run.run_id for e in evaluations]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ReportError(f"run {', '.join(duplicates)} is given more than once")
    first = evaluations[0]
    problems = []
    for other in evaluations[1:]:
        for key in ("metrics_version", "config_sha256"):
            if other.scores[key] != first.scores[key]:
                problems.append(
                    f"{other.run.run_id}: {key} {other.scores[key]!r} differs from"
                    f" {first.scores[key]!r} of {first.run.run_id}"
                )
        samples = {(s.id, s.gt_sha256) for s in other.run.samples}
        if samples != {(s.id, s.gt_sha256) for s in first.run.samples}:
            problems.append(
                f"{other.run.run_id}: the samples or their GT differ from {first.run.run_id}"
            )
    if problems:
        raise ReportError(
            "the evaluations are not comparable; evaluate every run with the same configuration"
            " and the same samples:\n" + "\n".join(problems)
        )


# Report data -----------------------------------------------------------------------------------


def _record_path(evaluation: Evaluation, sample_id: str) -> Path:
    attempt = evaluation.records[sample_id]["attempt"]
    return evaluation.run.path / "samples" / sample_id / f"attempt-{attempt}" / "record.json"


def _run_entry(evaluation: Evaluation) -> dict[str, Any]:
    run, scores, frozen = evaluation.run, evaluation.scores, evaluation.run_json
    statuses = dict.fromkeys(STATUSES, 0)
    for entry in scores["samples"]:
        statuses[entry["status"]] += 1
    return {
        "run_id": run.run_id,
        "run": str(run.path),
        "run_created": frozen["created"],
        "run_sha256": scores["run_sha256"],
        "run_ocrbench_version": frozen["ocrbench_version"],
        "engine": {
            "name": run.engine.name,
            "engine_version": run.engine.engine_version,
            "model": run.engine.model,
            "image": run.engine.image,
        },
        "engine_config_sha256": frozen["engine_config_sha256"],
        "image_id": run.image_id,
        "runtime": run.runtime,
        "dataset": frozen["dataset"],
        "host": frozen["host"],
        "evaluation": str(evaluation.path),
        "evaluated": scores["created"],
        "scores_sha256": evaluation.scores_sha256,
        "evaluation_ocrbench_version": scores["ocrbench_version"],
        "rapidfuzz_version": scores["rapidfuzz_version"],
        "samples": len(scores["samples"]),
        "statuses": statuses,
    }


def build_report(evaluations: list[Evaluation]) -> dict[str, Any]:
    """The report of comparable evaluations; the scores are copied, never recomputed."""
    check_comparable(evaluations)
    first = evaluations[0].scores
    non_success = []
    for evaluation in evaluations:
        for entry in evaluation.scores["samples"]:
            if entry["status"] == "success":
                continue
            record = evaluation.records[entry["sample_id"]]
            non_success.append(
                {
                    "run_id": evaluation.run.run_id,
                    "sample_id": entry["sample_id"],
                    "unit": entry["unit"],
                    "doc_type": entry["doc_type"],
                    "status": entry["status"],
                    "reason": entry["reason"],
                    "message": record.get("message"),
                    "attempt": entry["attempt"],
                    "scored_as": entry["scored_as"],
                    "record": str(_record_path(evaluation, entry["sample_id"])),
                }
            )
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ocrbench_version": ocrbench.__version__,
        "evaluation": {
            "config": first["config"],
            "config_sha256": first["config_sha256"],
            "metrics_version": first["metrics_version"],
            "rules": first["rules"],
        },
        "runs": [_run_entry(e) for e in evaluations],
        "aggregates": [
            {"run_id": e.run.run_id, **group}
            for e in evaluations
            for group in e.scores["aggregates"]
        ],
        "non_success": non_success,
    }


# CSV ---------------------------------------------------------------------------------------------

_EDIT = ("chars", "words", "flexible_chars", "flexible_words")
_OVERLAP = ("bag_of_words", "rouge2")
_SHORT = {
    "chars": "cer",
    "words": "wer",
    "flexible_chars": "fcer",
    "flexible_words": "fwer",
    "bag_of_words": "bow",
    "rouge2": "rouge2",
}


def _sample_row(run: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": run["run_id"],
        "engine": run["engine"]["name"],
        "engine_version": run["engine"]["engine_version"],
        "model": run["engine"]["model"],
        "sample_id": entry["sample_id"],
        "unit": entry["unit"],
        "doc_type": entry["doc_type"],
        "status": entry["status"],
        "reason": entry["reason"],
        "attempt": entry["attempt"],
        "scored_as": entry["scored_as"],
    }
    for key in _EDIT:
        counts = entry[key] or {}
        for field in ("rate", "errors", "reference"):
            row[f"{_SHORT[key]}_{field}"] = counts.get(field)
        if key.startswith("flexible"):
            row[f"{_SHORT[key]}_exact"] = counts.get("exact")
    for key in _OVERLAP:
        counts = entry[key] or {}
        for field in ("precision", "recall", "f1"):
            row[f"{_SHORT[key]}_{field}"] = counts.get(field)
    return row


def _aggregate_row(run: dict[str, Any], group: dict[str, Any], policy: str) -> dict[str, Any]:
    key = group["group"]
    row = {
        "run_id": run["run_id"],
        "engine": run["engine"]["name"],
        "engine_version": run["engine"]["engine_version"],
        "model": run["engine"]["model"],
        "failure_policy": policy,
        "group": "doc_type" if "doc_type" in key else "unit" if "unit" in key else "all",
        "unit": key.get("unit"),
        "doc_type": key.get("doc_type"),
        "samples": group["samples"],
        "scored": group["scored"],
        "coverage": group["coverage"],
        **{status: group["statuses"].get(status, 0) for status in STATUSES},
    }
    for name in _EDIT:
        micro, macro = group[name]["micro"] or {}, group[name]["macro"]["rate"]
        short = _SHORT[name]
        row |= {f"{short}_micro_{f}": micro.get(f) for f in ("rate", "errors", "reference")}
        row |= {
            f"{short}_macro_rate": macro["mean"],
            f"{short}_macro_samples": macro["samples"],
            f"{short}_macro_undefined": macro["undefined"],
        }
        if name.startswith("flexible"):
            row[f"{short}_lower_bound_rate"] = micro.get("lower_bound_rate")
            row[f"{short}_not_exact"] = group[name]["not_exact"]
    for name in _OVERLAP:
        micro, short = group[name]["micro"] or {}, _SHORT[name]
        fields = ("precision", "recall", "f1", "matched", "reference", "hypothesis")
        row |= {f"{short}_micro_{f}": micro.get(f) for f in fields}
        row[f"{short}_macro_f1"] = group[name]["macro"]["f1"]["mean"]
        row[f"{short}_macro_f1_samples"] = group[name]["macro"]["f1"]["samples"]
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "x", encoding="utf-8", newline="") as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: "" if v is None else v for k, v in row.items()} for row in rows)


# HTML --------------------------------------------------------------------------------------------

_CSS = """
body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #1b1b1b; background: #fff; }
table { border-collapse: collapse; margin: 0.5rem 0 1.5rem; }
th, td { border: 1px solid #ccc; padding: 0.2rem 0.5rem; vertical-align: top; }
th { background: #f0f0f0; text-align: left; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.note { background: #fff6d6; border: 1px solid #e0c060; padding: 0.5rem 0.75rem; }
.warn { background: #fde2e1; border: 1px solid #d08080; padding: 0.5rem 0.75rem; }
.text { white-space: pre-wrap; font-family: ui-monospace, monospace; overflow-wrap: anywhere; }
del { background: #f8c8c8; text-decoration: line-through; }
ins { background: #c8ecc8; text-decoration: none; }
code { font-size: 0.9em; }
"""


def _e(value: Any) -> str:
    return escape("-" if value is None else str(value), quote=True)


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}"


def _link(target: Path, base: Path) -> str:
    """A relative URL from a page in ``base`` to a local file."""
    return quote(Path(os.path.relpath(target, base)).as_posix())


def _page(title: str, body: list[str]) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>\n"
        + "\n".join(body)
        + "\n</body></html>\n"
    )


def _table(header: list[str], rows: list[list[str]]) -> str:
    """``rows`` are already escaped HTML cells; a cell starting with '#' is right-aligned."""
    head = "".join(f"<th>{_e(h)}</th>" for h in header)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{c[1:]}</td>' if c.startswith("#") else f"<td>{c}</td>" for c in row
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _group_name(key: dict[str, Any]) -> str:
    if not key:
        return "all samples"
    parts = [f"unit {key['unit']}"]
    if "doc_type" in key:
        doc_type = key["doc_type"]
        parts.append(
            "document type unspecified" if doc_type is None else f"document type {doc_type}"
        )
    return ", ".join(parts)


def _engine_label(run: dict[str, Any]) -> str:
    engine = run["engine"]
    model = f", model {engine['model']}" if engine["model"] else ""
    return f"{engine['name']} {engine['engine_version']}{model}"


def _micro(group: dict[str, Any], key: str, field: str) -> str:
    micro = group[key]["micro"] or {}
    value = _percent(micro.get(field))
    return value + ("*" if group[key].get("not_exact") else "")


def _macro(group: dict[str, Any], key: str, field: str) -> str:
    macro = group[key]["macro"][field]
    undefined = f", {macro['undefined']} undefined" if macro["undefined"] else ""
    return f"{_percent(macro['mean'])} (n={macro['samples']}{undefined})"


def _results(report: dict[str, Any]) -> list[str]:
    runs = {run["run_id"]: run for run in report["runs"]}
    keys: list[dict[str, Any]] = []
    for group in report["aggregates"]:
        if group["group"] not in keys:
            keys.append(group["group"])
    body = []
    for key in keys:
        groups = [g for g in report["aggregates"] if g["group"] == key]
        body.append(f"<h3>{_e(_group_name(key))}</h3>")
        micro_rows, macro_rows = [], []
        for g in groups:
            label = f"{_e(g['run_id'])}<br><small>{_e(_engine_label(runs[g['run_id']]))}</small>"
            chars, words = g["chars"]["micro"] or {}, g["words"]["micro"] or {}
            micro_rows.append(
                [
                    label,
                    f"#{g['samples']}",
                    f"#{g['scored']}",
                    f"#{_percent(g['coverage'])}",
                    f"#{_e(chars.get('reference'))}",
                    f"#{_e(words.get('reference'))}",
                    *(f"#{_micro(g, k, 'rate')}" for k in _EDIT),
                    *(f"#{_micro(g, 'bag_of_words', f)}" for f in ("precision", "recall", "f1")),
                    f"#{_micro(g, 'rouge2', 'f1')}",
                ]
            )
            macro_rows.append(
                [
                    label,
                    *(f"#{_macro(g, k, 'rate')}" for k in _EDIT),
                    f"#{_macro(g, 'bag_of_words', 'f1')}",
                    f"#{_macro(g, 'rouge2', 'f1')}",
                ]
            )
        body.append("<p>Micro averages (counts summed over the samples), in %:</p>")
        body.append(
            _table(
                ["run", "samples", "scored", "coverage %", "GT chars", "GT words"]
                + ["CER", "WER", "fCER", "fWER", "BoW P", "BoW R", "BoW F1", "ROUGE-2 F1"],
                micro_rows,
            )
        )
        body.append("<p>Macro averages (mean of the defined per-sample values), in %:</p>")
        body.append(
            _table(["run", "CER", "WER", "fCER", "fWER", "BoW F1", "ROUGE-2 F1"], macro_rows)
        )
    return body


def _index(
    report: dict[str, Any], samples: list[tuple[dict, list[dict]]], out: Path, inspected: bool
) -> str:
    config = report["evaluation"]["config"]
    exclude = config["failure_policy"] == "exclude"
    count = report["runs"][0]["samples"]
    body = [
        "<h1>OCR benchmark report</h1>",
        f"<p>Created {_e(report['created'])} by ocrbench {_e(report['ocrbench_version'])}."
        f" Evaluation <code>{_e(config['name'])}</code>: Unicode {_e(config['unicode'])},"
        f" failure policy <code>{_e(config['failure_policy'])}</code>, metrics version"
        f" {_e(report['evaluation']['metrics_version'])}, configuration"
        f" <code>{_e(report['evaluation']['config_sha256'][:12])}</code>.</p>",
        f'<p class="note">{count} samples per run. Every score is shown with its sample'
        " counts; scores on a small sample set are diagnostics, not evidence of generalizable"
        " model rankings.</p>",
    ]
    if exclude:
        body.append(
            '<p class="warn">Failure policy <code>exclude</code>: failed samples are left out'
            " of these scores. This is not the primary score; compare runs only by their"
            " coverage and the failures listed below.</p>"
        )
    else:
        body.append(
            "<p>Failure policy <code>empty-output</code> (primary): a failed sample is scored as"
            " an empty prediction.</p>"
        )
    body.append("<h2>Runs</h2>")
    rows = []
    for run in report["runs"]:
        image_id = (run["image_id"] or "-")[:19]
        rows.append(
            [
                _e(run["run_id"]),
                _e(_engine_label(run)),
                f"{_e(run['engine']['image'])}<br><code>{_e(image_id)}</code>",
                f"<code>{_e(run['engine_config_sha256'][:12])}</code>",
                f"{_e(run['dataset']['name'])}<br>"
                f"<code>{_e(run['dataset']['manifest_sha256'][:12])}</code>",
                _e(run["run_created"]),
                f'<a href="{_link(Path(run["evaluation"]) / "scores.json", out)}">'
                f"{_e(run['evaluated'])}</a>",
                _e(f"{run['run_ocrbench_version']} / {run['evaluation_ocrbench_version']}"),
                ", ".join(f"{n} {s}" for s, n in run["statuses"].items() if n) or "-",
            ]
        )
    body.append(
        _table(
            ["run", "engine", "image", "engine config", "dataset", "run created", "evaluated"]
            + ["ocrbench run / eval", "statuses"],
            rows,
        )
    )
    body.append("<h2>Results</h2>")
    body.append(
        "<p>fCER/fWER: flexible (order-independent) CER/WER. A value marked * includes samples"
        " whose flexible score is not proven optimal and may overstate errors.</p>"
    )
    body += _results(report)
    body.append("<h2>Samples that did not succeed</h2>")
    rows = []
    for item in report["non_success"]:
        record = _link(Path(item["record"]), out)
        rows.append(
            [
                _e(item["run_id"]),
                _e(item["sample_id"]),
                _e(item["status"]),
                _e(item["reason"]),
                f'<span class="text">{_e(item["message"])}</span>',
                _e(item["scored_as"]),
                f'<a href="{record}">record</a>',
            ]
        )
    if rows:
        body.append(
            _table(["run", "sample", "status", "reason", "message", "scored as", "record"], rows)
        )
    else:
        body.append("<p>Every sample of every run succeeded.</p>")
    body.append("<h2>Per sample</h2>")
    rows = []
    for sample_id in [entry["sample_id"] for entry in samples[0][1]]:
        for run, entries in samples:
            entry = next(e for e in entries if e["sample_id"] == sample_id)
            name = _e(sample_id)
            if inspected:
                page = _link(out / _inspect_path(run["run_id"], sample_id), out)
                name = f'<a href="{page}">{name}</a>'
            values = [
                f"#{_percent((entry[k] or {}).get('rate'))}"
                + ("*" if entry[k] and not entry[k].get("exact", True) else "")
                for k in _EDIT
            ]
            values += [f"#{_percent((entry[k] or {}).get('f1'))}" for k in _OVERLAP]
            rows.append(
                [name, _e(run["run_id"]), _e(entry["status"]), _e(entry["scored_as"]), *values]
            )
    body.append(
        _table(
            ["sample", "run", "status", "scored as", "CER", "WER", "fCER", "fWER", "BoW F1"]
            + ["ROUGE-2 F1"],
            rows,
        )
    )
    return _page("OCR benchmark report", body)


# Inspection pages --------------------------------------------------------------------------------


def _inspect_path(run_id: str, sample_id: str) -> Path:
    return Path("samples") / run_id / f"{sample_id}.html"


def diff_html(reference: str, hypothesis: str) -> str:
    """Escaped character differences: deleted reference text in <del>, inserted text in <ins>."""
    parts = []
    for op in Levenshtein.opcodes(reference, hypothesis):
        old = escape(reference[op.src_start : op.src_end])
        new = escape(hypothesis[op.dest_start : op.dest_end])
        if op.tag == "equal":
            parts.append(old)
        else:
            parts.append(
                (f"<del>{old}</del>" if old else "") + (f"<ins>{new}</ins>" if new else "")
            )
    return "".join(parts)


def _joined(lines: list[str], start: int, stop: int, form: str) -> str:
    return " ".join(u for u in (normalize(line, form) for line in lines[start:stop]) if u)


def _unmatched(lines: list[str], covered: set[int], form: str) -> list[str]:
    return [
        f"{n}: {_e(normalize(line, form))}"
        for n, line in enumerate(lines)
        if n not in covered and normalize(line, form)
    ]


def _inspect(evaluation: Evaluation, entry: dict[str, Any], out: Path) -> str:
    run = evaluation.run
    sample = next(s for s in run.samples if s.id == entry["sample_id"])
    record = evaluation.records[sample.id]
    form = evaluation.scores["config"]["unicode"]
    base = (out / _inspect_path(run.run_id, sample.id)).parent
    body = [
        f"<h1>{_e(sample.id)}</h1>",
        f'<p><a href="{_link(out / "index.html", base)}">report</a> · run {_e(run.run_id)}'
        f" · engine {_e(run.engine.name)} {_e(run.engine.engine_version)}"
        f" · unit {_e(entry['unit'])} · document type {_e(entry['doc_type'])}</p>",
        f"<p>Status <b>{_e(entry['status'])}</b>, reason {_e(entry['reason'])}, attempt"
        f" {_e(entry['attempt'])}, scored as <b>{_e(entry['scored_as'])}</b>."
        f' <a href="{_link(_record_path(evaluation, sample.id), base)}">execution record</a></p>',
    ]
    if record.get("message"):
        body.append(f'<p>Engine message: <span class="text">{_e(record["message"])}</span></p>')
    reference = reference_lines(sample.id, sample.gt)
    if entry["scored_as"] == "excluded":
        body.append('<p class="warn">Excluded by the failure policy; no score.</p>')
        body.append("<h2>Reference</h2>")
        body.append(f'<p class="text">{_e(normalize(" ".join(reference), form))}</p>')
        return _page(f"{sample.id} · {run.run_id}", body)
    hypothesis: list[str] = []
    text = ""
    if entry["scored_as"] == "prediction":
        prediction, _ = stored_prediction(run, sample, record)
        hypothesis, text = prediction_lines(prediction), prediction.text
    else:
        body.append('<p class="warn">Failed invocation scored as an empty prediction.</p>')
    rows = [
        [
            _e(name),
            f"#{_percent(entry[key]['rate'])}",
            f"#{entry[key]['errors']}",
            f"#{entry[key]['reference']}",
            f"#{entry[key]['substitutions']}",
            f"#{entry[key]['deletions']}",
            f"#{entry[key]['insertions']}",
        ]
        for name, key in (("CER", "chars"), ("WER", "words"))
        + (("fCER", "flexible_chars"), ("fWER", "flexible_words"))
    ]
    body.append(_table(["metric", "rate %", "errors", "reference", "S", "D", "I"], rows))
    body.append("<h2>Ordered differences</h2>")
    body.append(
        "<p>Normalized GT in reading order against the normalized prediction:"
        " <del>missing GT text</del>, <ins>extra predicted text</ins>.</p>"
    )
    ordered = diff_html(normalize(" ".join(reference), form), normalize(text, form))
    body.append(f'<p class="text">{ordered}</p>')
    flexible = entry["flexible_chars"]
    body.append("<h2>Flexible groups (fCER)</h2>")
    body.append(
        f"<p>Method {_e(flexible['method'])}, {flexible['groups']} groups,"
        f" {flexible['order_breaks']} order breaks, proven optimal: {_e(flexible['exact'])}."
        " Groups in prediction order; GT and prediction line numbers start at 0.</p>"
    )
    rows, ref_covered, hyp_covered = [], set(), set()
    for ref_start, ref_stop, hyp_start, hyp_stop in flexible["group_lines"]:
        ref_covered.update(range(ref_start, ref_stop))
        hyp_covered.update(range(hyp_start, hyp_stop))
        diff = diff_html(
            _joined(reference, ref_start, ref_stop, form),
            _joined(hypothesis, hyp_start, hyp_stop, form),
        )
        rows.append(
            [
                f"#{ref_start}–{ref_stop - 1}",
                f"#{hyp_start}–{hyp_stop - 1}",
                f'<span class="text">{diff}</span>',
            ]
        )
    if rows:
        body.append(_table(["GT lines", "prediction lines", "differences"], rows))
    for title, lines, covered in (
        ("Unmatched GT lines (deleted)", reference, ref_covered),
        ("Unmatched prediction lines (inserted)", hypothesis, hyp_covered),
    ):
        unmatched = _unmatched(lines, covered, form)
        body.append(f"<h3>{_e(title)}: {len(unmatched)}</h3>")
        if unmatched:
            body.append('<p class="text">' + "\n".join(unmatched) + "</p>")
    return _page(f"{sample.id} · {run.run_id}", body)


# Writing -----------------------------------------------------------------------------------------


def write_report(evaluations: list[Evaluation], out: Path, *, inspect: bool = False) -> dict:
    """Write the report into a new directory ``out``; an existing directory is refused, so a
    report never mixes files of different evaluations. Nothing is written when it fails."""
    report = build_report(evaluations)
    out = Path(out).resolve()
    runs = {run["run_id"]: run for run in report["runs"]}
    pairs = [(runs[e.run.run_id], e.scores["samples"]) for e in evaluations]
    pages = {}
    if inspect:
        for evaluation in evaluations:
            for entry in evaluation.scores["samples"]:
                path = _inspect_path(evaluation.run.run_id, entry["sample_id"])
                try:
                    pages[path] = _inspect(evaluation, entry, out)
                except EvaluationError as error:
                    raise ReportError(str(error)) from error
    try:
        out.mkdir(parents=True)
    except FileExistsError as error:
        raise ReportError(f"{out} already exists; choose a new report directory") from error
    text = json.dumps(report, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    (out / "report.json").write_text(text, encoding="utf-8")
    policy = report["evaluation"]["config"]["failure_policy"]
    _write_csv(
        out / "aggregates.csv",
        [_aggregate_row(runs[g["run_id"]], g, policy) for g in report["aggregates"]],
    )
    _write_csv(out / "samples.csv", [_sample_row(r, s) for r, ss in pairs for s in ss])
    for path, page in pages.items():
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_text(page, encoding="utf-8")
    (out / "index.html").write_text(_index(report, pairs, out, inspect), encoding="utf-8")
    return report
