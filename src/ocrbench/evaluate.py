"""Evaluate a run: score stored predictions against the GT frozen with the run.

The evaluator reads run.json, the newest execution record of every sample, its standardized
prediction and the PAGE XML reference; it never runs an engine. A failed sample is scored as an
empty prediction by default (failure policy ``empty-output``) or left out of the aggregates
(``exclude``, with coverage). Nothing is written back over predictions: every evaluation gets a
new directory with its frozen configuration. The rules are documented in docs/EVALUATION.md.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any

import ocrbench
from ocrbench.issues import Severity
from ocrbench.manifest import sha256_file
from ocrbench.metrics import (
    METRICS_VERSION,
    UNICODE_FORMS,
    EditCounts,
    Overlap,
    TextScores,
    score_text,
)
from ocrbench.pagexml import read_page_xml
from ocrbench.prediction import load_prediction
from ocrbench.runner import Run, RunSample, latest_record, verify_ground_truth

EVAL_CONFIG_FORMAT = "ocrbench.eval-config"
EVAL_CONFIG_VERSION = 1
SCORES_FORMAT = "ocrbench.scores"
SCORES_VERSION = 1
FAILURE_POLICIES = ("empty-output", "exclude")
# Fixed rules of METRICS_VERSION 1, recorded with every score.
TEXT_RULES = {
    "characters": "code-points",
    "whitespace": "collapse-to-space",
    "words": "whitespace-split",
    "reference_lines": "joined-by-space",
}
_KEYS = {"format", "version", "name", "unicode", "failure_policy"}


class EvaluationError(RuntimeError):
    """The run cannot be scored (pending samples, changed or invalid GT, broken run store)."""


class EvalConfigError(ValueError):
    """The evaluation configuration is invalid; ``errors`` lists every problem found."""

    def __init__(self, location: str, errors: list[str]) -> None:
        super().__init__(f"{location}: " + "; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class EvalConfig:
    name: str = "strict"
    unicode: str = "NFC"  # Unicode normalization form, one of UNICODE_FORMS
    failure_policy: str = "empty-output"  # one of FAILURE_POLICIES

    def to_json(self) -> dict[str, Any]:
        return {
            "format": EVAL_CONFIG_FORMAT,
            "version": EVAL_CONFIG_VERSION,
            "name": self.name,
            "unicode": self.unicode,
            "failure_policy": self.failure_policy,
        }

    def sha256(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_eval_config(data: Any, location: str = "evaluation") -> EvalConfig:
    if not isinstance(data, dict):
        raise EvalConfigError(location, ["the configuration must be a table"])
    errors = [f"unknown field {key!r}" for key in sorted(set(data) - _KEYS)]
    if data.get("format", EVAL_CONFIG_FORMAT) != EVAL_CONFIG_FORMAT:
        errors.append(f"'format' must be {EVAL_CONFIG_FORMAT!r}")
    if data.get("version", EVAL_CONFIG_VERSION) != EVAL_CONFIG_VERSION:
        errors.append(f"version {data.get('version')!r} is not supported")
    defaults = EvalConfig()
    name = data.get("name", defaults.name)
    if not (isinstance(name, str) and name):
        errors.append("'name' must be a non-empty string")
    unicode = data.get("unicode", defaults.unicode)
    if unicode not in UNICODE_FORMS:
        errors.append(f"'unicode' must be one of {UNICODE_FORMS}")
    policy = data.get("failure_policy", defaults.failure_policy)
    if policy not in FAILURE_POLICIES:
        errors.append(f"'failure_policy' must be one of {FAILURE_POLICIES}")
    if errors:
        raise EvalConfigError(location, errors)
    return EvalConfig(name, unicode, policy)


DEFAULT_CONFIG = EvalConfig()


def load_eval_config(path: Path) -> EvalConfig:
    path = Path(path)
    try:
        with open(path, "rb") as stream:
            data = tomllib.load(stream)
    except OSError as error:
        raise EvalConfigError(path.name, [f"cannot read: {error}"]) from error
    except tomllib.TOMLDecodeError as error:
        raise EvalConfigError(path.name, [f"invalid TOML: {error}"]) from error
    return parse_eval_config(data, path.name)


# Per-sample scoring ---------------------------------------------------------------------------


def _edit_json(counts: EditCounts) -> dict[str, Any]:
    return {
        "reference": counts.reference,
        "hypothesis": counts.hypothesis,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "errors": counts.errors,
        "rate": counts.rate,
    }


def _overlap_json(counts: Overlap) -> dict[str, Any]:
    return {
        "reference": counts.reference,
        "hypothesis": counts.hypothesis,
        "matched": counts.matched,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
    }


def _scores_json(scores: TextScores | None) -> dict[str, Any]:
    if scores is None:
        return dict.fromkeys(("chars", "words", "bag_of_words", "rouge2"))
    return {
        "chars": _edit_json(scores.chars),
        "words": _edit_json(scores.words),
        "bag_of_words": _overlap_json(scores.bag_of_words),
        "rouge2": _overlap_json(scores.rouge2),
    }


def _reference_text(sample_id: str, gt: Path) -> str:
    page, issues = read_page_xml(gt, sample_id)
    errors = [issue.format() for issue in issues if issue.severity is Severity.ERROR]
    if page is None or errors:
        raise EvaluationError(f"{sample_id}: invalid GT {gt}:\n" + "\n".join(errors))
    return " ".join(line.text for line in page.reference_lines)


def _predicted_text(run: Run, sample_id: str, record: dict[str, Any]) -> str:
    """The standardized page text of a valid output; never native Markdown or HTML."""
    if record["prediction"] is None:
        raise EvaluationError(f"{sample_id}: status {record['status']!r} without a prediction")
    attempt = run.path / "samples" / sample_id / f"attempt-{record['attempt']}"
    prediction, issues = load_prediction(attempt / record["prediction"], sample_id=sample_id)
    if prediction is None:
        errors = "\n".join(issue.format() for issue in issues)
        raise EvaluationError(f"{sample_id}: stored prediction is invalid:\n{errors}")
    return prediction.text


def _sample(
    run: Run, sample: RunSample, record: dict[str, Any], config: EvalConfig
) -> tuple[dict[str, Any], TextScores | None]:
    failed = record["status"] == "failure"
    if failed and config.failure_policy == "exclude":
        source, scores = "excluded", None
    else:
        reference = _reference_text(sample.id, sample.gt)
        # A failure is scored as an empty input here; no prediction is created or stored.
        hypothesis = "" if failed else _predicted_text(run, sample.id, record)
        source = "empty" if failed else "prediction"
        scores = score_text(reference, hypothesis, config.unicode)
    entry = {
        "sample_id": sample.id,
        "unit": sample.job["unit"],
        "doc_type": sample.doc_type,
        "status": record["status"],
        "reason": record["reason"],
        "attempt": record["attempt"],
        "gt_sha256": sample.gt_sha256,
        "scored_as": source,
        **_scores_json(scores),
    }
    return entry, scores


# Aggregates -----------------------------------------------------------------------------------


def _macro(values: list[float | None]) -> dict[str, Any]:
    """Mean of the defined per-sample values; ``undefined`` counts the samples left out."""
    defined = [v for v in values if v is not None]
    return {
        "mean": fmean(defined) if defined else None,
        "samples": len(defined),
        "undefined": len(values) - len(defined),
    }


def _aggregate(group: dict[str, Any], members: list[tuple[dict, TextScores | None]]):
    scored = [scores for _, scores in members if scores is not None]
    statuses: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for entry, _ in members:
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1
        if entry["reason"]:
            reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1
    result: dict[str, Any] = {
        "group": group,
        "samples": len(members),
        "scored": len(scored),
        "coverage": len(scored) / len(members),
        "statuses": statuses,
        "failure_reasons": reasons,
    }
    total = sum(scored[1:], scored[0]) if scored else None
    for key in ("chars", "words"):
        counts = [getattr(s, key) for s in scored]
        result[key] = {
            "micro": _edit_json(getattr(total, key)) if total else None,
            "macro": {"rate": _macro([c.rate for c in counts])},
        }
    for key in ("bag_of_words", "rouge2"):
        counts = [getattr(s, key) for s in scored]
        result[key] = {
            "micro": _overlap_json(getattr(total, key)) if total else None,
            "macro": {
                m: _macro([getattr(c, m) for c in counts]) for m in ("precision", "recall", "f1")
            },
        }
    return result


def _groups(samples: list[tuple[dict, TextScores | None]]) -> list[dict[str, Any]]:
    """Overall, per unit, and per document type within each unit (None: not specified)."""
    keys: list[dict[str, Any]] = [{}]
    units = sorted({entry["unit"] for entry, _ in samples})
    for unit in units:
        keys.append({"unit": unit})
        types = {entry["doc_type"] for entry, _ in samples if entry["unit"] == unit}
        for doc_type in sorted(types, key=lambda t: (t is not None, t or "")):
            keys.append({"unit": unit, "doc_type": doc_type})
    return [
        _aggregate(key, [m for m in samples if all(m[0][k] == v for k, v in key.items())])
        for key in keys
    ]


def evaluate_run(run: Run, config: EvalConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Score every sample of a finished run; raises EvaluationError when it cannot be scored."""
    gt_issues = verify_ground_truth(run)
    if gt_issues:
        raise EvaluationError(
            "predictions can only be scored against the GT frozen with the run:\n"
            + "\n".join(issue.format() for issue in gt_issues)
        )
    records = {s.id: latest_record(run, s.id) for s in run.samples}
    pending = [sid for sid, record in records.items() if record is None]
    if pending:
        raise EvaluationError(
            f"{len(pending)} samples are pending ({', '.join(pending[:5])}"
            f"{', ...' if len(pending) > 5 else ''}); finish the run with 'run resume' first"
        )
    samples = [_sample(run, s, records[s.id], config) for s in run.samples]
    return {
        "format": SCORES_FORMAT,
        "version": SCORES_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_id": run.run_id,
        "run_sha256": sha256_file(run.path / "run.json"),
        "config": config.to_json(),
        "config_sha256": config.sha256(),
        "metrics_version": METRICS_VERSION,
        "rules": TEXT_RULES | {"unicode": config.unicode},
        "ocrbench_version": ocrbench.__version__,
        "rapidfuzz_version": version("rapidfuzz"),
        "aggregates": _groups(samples),
        "samples": [entry for entry, _ in samples],
    }


def write_scores(run: Run, scores: dict[str, Any]) -> Path:
    """Store scores in a new runs/<id>/evaluations/<n>/ directory; nothing is overwritten."""
    root = run.path / "evaluations"
    root.mkdir(exist_ok=True)
    number = 1
    while True:
        directory = root / str(number)
        try:
            directory.mkdir()
            break
        except FileExistsError:
            number += 1
    path = directory / "scores.json"
    text = json.dumps(scores, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(text)
    return path
