"""Command-line interface: ``ocrbench dataset``, ``prediction``, ``run``, ``evaluate`` and
``report``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from ocrbench.convert import FORMATS, SUFFIXES, convert_native
from ocrbench.engine import EngineConfigError, load_engine_config
from ocrbench.evaluate import (
    DEFAULT_CONFIG,
    EvalConfigError,
    EvaluationError,
    evaluate_run,
    load_eval_config,
    write_scores,
)
from ocrbench.issues import Issue, Severity
from ocrbench.manifest import load_manifest, validate_dataset, validate_sample
from ocrbench.prediction import (
    Prediction,
    load_prediction,
    prediction_to_json,
    validate_prediction,
)
from ocrbench.report import ReportError, load_evaluation, write_report
from ocrbench.runner import (
    DockerLauncher,
    Run,
    RunnerError,
    create_run,
    execute,
    load_run,
    resolve_image,
    summarize,
    verify_ground_truth,
    verify_image,
)


def _validate(args: argparse.Namespace) -> int:
    report = validate_dataset(args.manifest)
    shown = [i for i in report.issues if args.verbose or i.severity is not Severity.INFO]
    for issue in shown:
        print(issue.format())
    if report.summaries:
        print(f"\n{'sample':<48} {'unit':<9} {'size':>11} {'regions':>7} {'lines':>6} {'chars':>7}")
        for s in report.summaries:
            size = f"{s.width}x{s.height}"
            print(
                f"{s.sample_id:<48} {s.unit:<9} {size:>11} {s.regions:>7} "
                f"{s.reference_lines:>6} {s.reference_chars:>7}"
            )
    total = len(report.manifest.samples) if report.manifest else 0
    hidden = "" if args.verbose else " (use -v to list info)"
    print(
        f"\n{len(report.summaries)}/{total} samples valid; "
        f"{report.count(Severity.ERROR)} errors, {report.count(Severity.WARNING)} warnings, "
        f"{report.count(Severity.INFO)} info{hidden}"
    )
    return 0 if report.ok else 1


def _text(args: argparse.Namespace) -> int:
    manifest, issues = load_manifest(args.manifest)
    for issue in issues:
        print(issue.format(), file=sys.stderr)
    if manifest is None:
        return 1
    sample = next((s for s in manifest.samples if s.id == args.sample_id), None)
    if sample is None:
        print(f"error: no sample {args.sample_id!r} in {args.manifest}", file=sys.stderr)
        return 1
    page, issues = validate_sample(sample)
    for issue in issues:
        if issue.severity is not Severity.INFO:
            print(issue.format(), file=sys.stderr)
    if page is None:
        return 1
    for line in page.reference_lines:
        print(f"{line.region_id}\t{line.line_id or '-'}\t{line.text}" if args.ids else line.text)
    return 0


def _finish_prediction(
    prediction: Prediction | None, issues: list[Issue], args: argparse.Namespace
) -> int:
    for issue in issues:
        if args.verbose or issue.severity is not Severity.INFO:
            print(issue.format(), file=sys.stderr)
    if prediction is None:
        print("invalid prediction", file=sys.stderr)
        return 1
    if args.output:
        text = json.dumps(prediction_to_json(prediction), ensure_ascii=False, indent=1)
        try:
            args.output.write_text(text + "\n", encoding="utf-8")
        except OSError as error:
            print(f"error: cannot write {args.output}: {error}", file=sys.stderr)
            return 1
    return 0


def _prediction_validate(args: argparse.Namespace) -> int:
    size = None
    if args.image:
        try:
            with Image.open(args.image) as image:  # reads the header only
                size = image.size
        except OSError as error:  # includes missing files and unidentified images
            print(f"error: cannot read image {args.image}: {error}", file=sys.stderr)
            return 1
    prediction, issues = load_prediction(args.prediction, sample_id=args.sample_id, image_size=size)
    return _finish_prediction(prediction, issues, args)


def _prediction_convert(args: argparse.Namespace) -> int:
    source_format = args.source_format or SUFFIXES.get(args.native.suffix.lower())
    if source_format is None:
        print(f"error: cannot infer the format of {args.native}; use --from", file=sys.stderr)
        return 1
    try:
        content = args.native.read_bytes().decode("utf-8")
    except UnicodeDecodeError as error:
        print(f"error: {args.native} is not UTF-8: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: cannot read {args.native}: {error}", file=sys.stderr)
        return 1
    data = convert_native(content, source_format, args.sample_id)
    prediction, issues = validate_prediction(
        data, sample_id=args.sample_id, location=args.native.name
    )
    status = _finish_prediction(prediction, issues, args)
    if status == 0 and not args.output:
        print(json.dumps(prediction_to_json(prediction), ensure_ascii=False, indent=1))
    return status


def _print_summary(run: Run) -> None:
    summary = summarize(run)
    for sample_id, reason, message in summary["failures"]:
        print(f"failure: {sample_id}: {reason}: {message}")
    counts = ", ".join(f"{count} {status}" for status, count in summary["counts"].items())
    print(f"{run.path}: {counts}")


def _progress(record: dict) -> None:
    reason = f" ({record['reason']})" if record["reason"] else ""
    print(f"{record['sample_id']}: {record['status']}{reason}", file=sys.stderr, flush=True)


def _execute(run: Run, image_id: str, retry_failed: bool) -> int:
    launcher = DockerLauncher(run.engine, image_id, run.cache)
    try:
        execute(run, launcher, retry_failed=retry_failed, progress=_progress)
    except KeyboardInterrupt:
        print(f"interrupted; continue with: ocrbench run resume {run.path}", file=sys.stderr)
        return 130
    _print_summary(run)
    return 0


def _run_start(args: argparse.Namespace) -> int:
    try:
        engine = load_engine_config(args.engine)
        image_id = resolve_image(engine.image)
        run = create_run(
            engine,
            args.dataset,
            args.runs_dir,
            run_id=args.run_id,
            image_id=image_id,
            cache=args.cache.resolve() if args.cache else None,
        )
        print(f"run {run.run_id}: {len(run.samples)} samples, image {image_id}", file=sys.stderr)
        return _execute(run, image_id, retry_failed=False)
    except (EngineConfigError, RunnerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run_resume(args: argparse.Namespace) -> int:
    try:
        run = load_run(args.run)
        if run.runtime != "docker":
            raise RunnerError(f"{args.run} was not run with Docker and cannot be resumed here")
        image_id = resolve_image(run.engine.image)
        verify_image(run, image_id)
        return _execute(run, image_id, retry_failed=args.retry_failed)
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _run_status(args: argparse.Namespace) -> int:
    try:
        run = load_run(args.run)
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _print_summary(run)
    for issue in verify_ground_truth(run):  # predictions cannot be scored against changed GT
        print(issue.format(), file=sys.stderr)
    return 0


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}"


def _log_non_success(samples: list[dict], run_id: str | None = None) -> None:
    """One stderr line per sample that did not succeed, with how it was scored."""
    prefix = f"{run_id}: " if run_id else ""
    for entry in samples:
        if entry["status"] != "success":
            reason = f" ({entry['reason']})" if entry["reason"] else ""
            print(
                f"{prefix}{entry['sample_id']}: {entry['status']}{reason},"
                f" scored as {entry['scored_as']}",
                file=sys.stderr,
            )


def _evaluate(args: argparse.Namespace) -> int:
    try:
        config = load_eval_config(args.config) if args.config else DEFAULT_CONFIG
        run = load_run(args.run)
        scores = evaluate_run(run, config)
        path = write_scores(run, scores)
    except (EvalConfigError, EvaluationError, RunnerError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _log_non_success(scores["samples"])
    columns = (("CER", "chars", "rate"), ("WER", "words", "rate"))
    columns += (("fCER", "flexible_chars", "rate"), ("fWER", "flexible_words", "rate"))
    columns += (("BoW F1", "bag_of_words", "f1"), ("R-2 F1", "rouge2", "f1"))
    print(f"{'group':<32} {'samples':>7} {'coverage':>8}", *(f"{c[0]:>7}" for c in columns))
    for group in scores["aggregates"]:
        name = "/".join("-" if v is None else str(v) for v in group["group"].values()) or "all"
        # "*": the flexible score of some samples in the group is not proven optimal.
        values = [
            _percent((group[key]["micro"] or {}).get(m))
            + ("*" if group[key].get("not_exact") else "")
            for _, key, m in columns
        ]
        print(
            f"{name:<32} {group['samples']:>7} {_percent(group['coverage']):>8}",
            *(f"{v:>7}" for v in values),
        )
    print(f"micro averages in %, failure policy {config.failure_policy}; scores: {path}")
    print("fCER/fWER: flexible (order-independent) CER/WER")
    overall = scores["aggregates"][0]
    not_exact = [(m, overall[key]["not_exact"]) for m, key, _ in columns[2:4]]
    if any(n for _, n in not_exact):
        counts = ", ".join(f"{m} {n} of {overall['scored']}" for m, n in not_exact)
        print(
            f"* not proven optimal, may overstate errors: {counts} scored samples;"
            " see exact, lower_bound and flexible_units in scores.json"
        )
    return 0


def _report(args: argparse.Namespace) -> int:
    try:
        evaluations = [load_evaluation(path) for path in args.evaluations]
        report = write_report(evaluations, args.output, inspect=args.inspect)
    except (ReportError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for evaluation in evaluations:
        _log_non_success(evaluation.scores["samples"], evaluation.run.run_id)
    for run in report["runs"]:
        counts = ", ".join(f"{n} {s}" for s, n in run["statuses"].items() if n)
        print(f"{run['run_id']}: {run['evaluation']} ({counts})")
    policy = report["evaluation"]["config"]["failure_policy"]
    print(f"failure policy {policy}; report: {args.output.resolve() / 'index.html'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ocrbench", description="OCR/HTR benchmark tools")
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("dataset", help="dataset manifests and PAGE XML ground truth")
    actions = dataset.add_subparsers(dest="action", required=True)

    validate = actions.add_parser("validate", help="check a manifest and all its samples")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("-v", "--verbose", action="store_true", help="also list info findings")
    validate.set_defaults(handler=_validate)

    text = actions.add_parser("text", help="print a sample's reference lines in reading order")
    text.add_argument("manifest", type=Path)
    text.add_argument("sample_id")
    text.add_argument("--ids", action="store_true", help="prefix region and line ids")
    text.set_defaults(handler=_text)

    prediction = commands.add_parser("prediction", help="standardized OCR prediction JSON")
    actions = prediction.add_subparsers(dest="action", required=True)

    check = actions.add_parser("validate", help="check a prediction JSON file")
    check.add_argument("prediction", type=Path)
    check.add_argument("--image", type=Path, help="sample image, to check the declared size")
    check.add_argument("--sample-id", help="expected sample ID")
    check.add_argument("-o", "--output", type=Path, help="write the standardized prediction")
    check.add_argument("-v", "--verbose", action="store_true", help="also list info findings")
    check.set_defaults(handler=_prediction_validate)

    convert = actions.add_parser(
        "convert", help="convert native text, Markdown or HTML output into a prediction"
    )
    convert.add_argument("native", type=Path)
    convert.add_argument("--sample-id", required=True)
    convert.add_argument(
        "--from", dest="source_format", choices=FORMATS, help="default: from the file suffix"
    )
    convert.add_argument("-o", "--output", type=Path, help="default: print to stdout")
    convert.add_argument("-v", "--verbose", action="store_true", help="also list info findings")
    convert.set_defaults(handler=_prediction_convert)

    run = commands.add_parser("run", help="run an OCR engine over a dataset")
    actions = run.add_subparsers(dest="action", required=True)

    start = actions.add_parser("start", help="start a new run in a Docker container")
    start.add_argument("--engine", type=Path, required=True, help="engine configuration (TOML)")
    start.add_argument("--dataset", type=Path, required=True, help="dataset manifest")
    start.add_argument("--runs-dir", type=Path, default=Path("runs"), help="default: runs/")
    start.add_argument("--run-id", help="default: <engine>-<dataset>-<UTC time>")
    start.add_argument("--cache", type=Path, help="host directory mounted at /cache")
    start.set_defaults(handler=_run_start)

    resume = actions.add_parser("resume", help="continue a run; finished samples are kept")
    resume.add_argument("run", type=Path, help="run directory")
    resume.add_argument(
        "--retry-failed", action="store_true", help="also retry failed samples (new attempts)"
    )
    resume.set_defaults(handler=_run_resume)

    status = actions.add_parser("status", help="count samples by status and list failures")
    status.add_argument("run", type=Path, help="run directory")
    status.set_defaults(handler=_run_status)

    evaluate = commands.add_parser("evaluate", help="score a finished run against its GT")
    evaluate.add_argument("run", type=Path, help="run directory")
    evaluate.add_argument("--config", type=Path, help="evaluation configuration (TOML)")
    evaluate.set_defaults(handler=_evaluate)

    report = commands.add_parser("report", help="compare evaluated runs in CSV, JSON and HTML")
    report.add_argument(
        "evaluations",
        type=Path,
        nargs="+",
        help="run directories (their newest evaluation) or runs/<id>/evaluations/<n>",
    )
    report.add_argument(
        "-o", "--output", type=Path, required=True, help="new directory for the report"
    )
    report.add_argument(
        "--inspect",
        action="store_true",
        help="also write a GT-versus-prediction page per run and sample",
    )
    report.set_defaults(handler=_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
