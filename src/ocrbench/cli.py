"""Command-line interface: ``ocrbench dataset validate|text``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ocrbench.issues import Severity
from ocrbench.manifest import load_manifest, validate_dataset, validate_sample


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
