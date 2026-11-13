"""Export a stored run as an ARCHER OCR hypothesis XML submission (#136).

The export reads only what a finished run stored: the execution record and the standardized
prediction of every sample (``prediction.json``, never native Markdown or HTML). No engine is
run. Document names come from the ARCHER import (#135): each sample's sidecar
``archer/<id>.json`` gives ``meta/source_name``, the identifier the accepted ARCHER hypothesis
examples use (``<doc source_name="…">``). Only these identifiers are read from the import; the
reference text and blocks are never used to choose, split or order hypothesis text.

Format: one UTF-8 document ``<archer-ocr>`` with one ``<doc source_name="…">`` per exported
sample and zero or more plain-text ``<block>`` children. See docs/DATASETS.md.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ocrbench.archer import IMPORT_FORMAT, IMPORTER_VERSION, SIDECAR_FORMAT
from ocrbench.evaluate import EvaluationError, stored_prediction
from ocrbench.issues import Issue, Severity
from ocrbench.prediction import Prediction
from ocrbench.runner import Run, RunnerError, latest_record, load_run, verify_ground_truth

EXPORT_REPORT_FORMAT = "ocrbench.archer-export"
EXPORT_REPORT_VERSION = 1
BLOCK_MODES = ("page", "regions")
FAILED_POLICIES = ("error", "empty", "omit")
ROOT_TAG = "archer-ocr"
DOC_TAG = "doc"
BLOCK_TAG = "block"
ID_ATTRIBUTE = "source_name"

# Characters XML 1.0 cannot represent, even escaped (C0 controls except tab, LF and CR; U+FFFE,
# U+FFFF). Lone surrogates cannot occur: predictions are validated as UTF-8.
_NOT_XML = re.compile("[^\t\n\r -퟿-�\U00010000-\U0010ffff]")


class ArcherExportError(Exception):
    """The run cannot be exported; ``issues`` lists every problem found. Nothing is written."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        errors = [i for i in issues if i.severity is Severity.ERROR]
        super().__init__(f"{len(errors)} error(s)")


@dataclass
class ExportedDoc:
    sample_id: str
    source_name: str
    status: str  # the run status: success, truncated, partial, failure
    reason: str | None
    exported: str = "blocks"  # blocks, empty (a failure exported without blocks) or omitted
    blocks: list[str] = field(default_factory=list)
    fallback: bool = False  # regions mode used the page text of a text-only prediction


def xml_text(text: str) -> tuple[str, int]:
    """Text that survives an XML round trip: line breaks become ``\\n`` (an XML parser
    normalizes CR and CRLF anyway) and characters XML 1.0 cannot hold become a space.
    Returns the text and the number of replaced characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _NOT_XML.subn(" ", text)


def prediction_blocks(prediction: Prediction, mode: str, fallback: str | None) -> list[str] | None:
    """The block texts of a prediction in its own order; None for a text-only prediction in
    ``regions`` mode without a fallback.

    ``page``: the standardized page text as one block. ``regions``: every text-bearing region
    in the prediction's reading order, a region with lines as one block with its lines in order.
    Whitespace-only text adds no block.
    """
    if mode == "page":
        texts = [prediction.text]
    elif mode == "regions":
        texts = [r.content for r in prediction.regions if r.content is not None]
        if not texts and prediction.text.strip():  # text-only: no region carries text
            if fallback != "page":
                return None
            texts = [prediction.text]
    else:
        raise ValueError(f"unknown block mode {mode!r}")
    return [text for text in texts if text.strip()]


def hypothesis_xml(docs: list[ExportedDoc]) -> bytes:
    root = ET.Element(ROOT_TAG)
    for doc in docs:
        if doc.exported == "omitted":
            continue
        element = ET.SubElement(root, DOC_TAG, {ID_ATTRIBUTE: doc.source_name})
        for text in doc.blocks:
            ET.SubElement(element, BLOCK_TAG).text = text
    ET.indent(root)  # whitespace only between elements; block text is never touched
    # The declaration of the accepted ARCHER examples.
    body = ET.tostring(root, encoding="unicode")
    return f"<?xml version='1.0' encoding='UTF-8'?>\n{body}".encode()


def read_hypothesis(path: Path) -> list[tuple[str, list[str]]]:
    """Parse an ARCHER hypothesis file in the accepted example form; raises ValueError.

    Checks the structure of the local ``hyp/`` examples: root ``archer-ocr`` without attributes,
    only ``doc`` children with exactly a non-empty ``source_name``, only ``block`` children
    without attributes or children, no other text, and unique document names.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise ValueError(f"{path}: not well-formed XML: {error}") from error
    if root.tag != ROOT_TAG or root.attrib:
        raise ValueError(f"{path}: root must be <{ROOT_TAG}> without attributes")
    if (root.text or "").strip():
        raise ValueError(f"{path}: text directly in <{ROOT_TAG}>")
    docs: list[tuple[str, list[str]]] = []
    for doc in root:
        if doc.tag != DOC_TAG or set(doc.attrib) != {ID_ATTRIBUTE} or not doc.get(ID_ATTRIBUTE):
            raise ValueError(f'{path}: expected <{DOC_TAG} {ID_ATTRIBUTE}="…">, got <{doc.tag}>')
        if (doc.text or "").strip() or (doc.tail or "").strip():
            raise ValueError(f"{path}: text outside a block in {doc.get(ID_ATTRIBUTE)!r}")
        blocks = []
        for block in doc:
            if block.tag != BLOCK_TAG or block.attrib or len(block):
                raise ValueError(f"{path}: <{DOC_TAG}> may contain only plain <{BLOCK_TAG}>")
            if (block.tail or "").strip():
                raise ValueError(f"{path}: text outside a block in {doc.get(ID_ATTRIBUTE)!r}")
            blocks.append(block.text or "")
        docs.append((doc.get(ID_ATTRIBUTE), blocks))
    names = [name for name, _ in docs]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate {ID_ATTRIBUTE}")
    return docs


def _import_dir(run: Run, dataset: Path | None, issues: list[Issue]) -> Path | None:
    if dataset is not None:
        dataset = Path(dataset)
        directory = dataset.parent if dataset.is_file() else dataset
    else:
        # The importer writes page/<id>.xml next to import.json.
        parents = {sample.gt.resolve().parent.parent for sample in run.samples}
        if len(parents) != 1:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "not-archer-import",
                    "the run's references are not in one ARCHER import directory; use --dataset",
                )
            )
            return None
        directory = parents.pop()
    return directory.resolve()


def _load_import(directory: Path, issues: list[Issue]) -> dict[str, Any] | None:
    path = directory / "import.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        issues.append(
            Issue(
                Severity.ERROR,
                "not-archer-import",
                f"{path}: cannot read the ARCHER import record ({error}); the run was not "
                "created from `ocrbench dataset import-archer` output",
            )
        )
        return None
    if record.get("format") != IMPORT_FORMAT or record.get("version") != IMPORTER_VERSION:
        issues.append(
            Issue(
                Severity.ERROR,
                "not-archer-import",
                f"{path} is not an {IMPORT_FORMAT} v{IMPORTER_VERSION} record",
            )
        )
        return None
    return record


def _source_name(
    directory: Path, files: set[str], sample_id: str, gt: Path, issues: list[Issue]
) -> str | None:
    """``meta/source_name`` of the sample's task, from the importer's sidecar."""
    relative = f"archer/{sample_id}.json"

    def error(code: str, message: str) -> None:
        issues.append(Issue(Severity.ERROR, code, message, sample_id, relative))

    if relative not in files:
        error("missing-sidecar", "the ARCHER import has no sidecar for this sample")
        return None
    try:
        sidecar = json.loads((directory / relative).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error_:
        error("missing-sidecar", f"cannot read the sidecar: {error_}")
        return None
    if sidecar.get("format") != SIDECAR_FORMAT or sidecar.get("task", {}).get("id") != sample_id:
        error("sidecar-mismatch", "the sidecar does not describe this sample")
        return None
    if (directory / sidecar.get("page_xml", "")).resolve() != gt.resolve():
        error("sidecar-mismatch", f"the sidecar's reference is not the run's GT {gt}")
        return None
    name = (sidecar.get("image") or {}).get("source_name")
    if not isinstance(name, str) or not name.strip():
        error(
            "missing-source-name",
            "the task has no meta/source_name, the document identifier of ARCHER hypothesis "
            "files; the identifier is not guessed from the image name",
        )
        return None
    return name


def collect(
    run: Run,
    *,
    blocks: str = "page",
    fallback: str | None = None,
    failed: str = "error",
    dataset: Path | None = None,
) -> tuple[list[ExportedDoc], list[Issue]]:
    """Map every run sample to an exported document; see export_archer."""
    if blocks not in BLOCK_MODES:
        raise ValueError(f"blocks must be one of {BLOCK_MODES}")
    if failed not in FAILED_POLICIES:
        raise ValueError(f"failed must be one of {FAILED_POLICIES}")
    if fallback not in (None, "page"):
        raise ValueError("fallback must be None or 'page'")
    # The identifiers come from the import the run's GT was pinned to.
    issues = verify_ground_truth(run)
    directory = _import_dir(run, dataset, issues)
    record = _load_import(directory, issues) if directory else None
    if record is None:
        return [], issues
    files = set(record.get("files", []))
    imported = {
        f.removeprefix("archer/").removesuffix(".json")
        for f in files
        if re.fullmatch(r"archer/[^/]+\.json", f)
    }
    run_ids = {sample.id for sample in run.samples}
    for sample_id in sorted(imported - run_ids):
        issues.append(
            Issue(
                Severity.ERROR,
                "document-not-in-run",
                f"the ARCHER import has this document but the run does not ({directory})",
                sample_id,
            )
        )

    docs: list[ExportedDoc] = []
    names: dict[str, str] = {}
    for sample in run.samples:
        name = _source_name(directory, files, sample.id, sample.gt, issues)
        if name is not None and name in names:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "duplicate-source-name",
                    f"source_name {name!r} is also used by sample {names[name]!r}",
                    sample.id,
                )
            )
        if name is not None:
            names.setdefault(name, sample.id)
        try:
            run_record = latest_record(run, sample.id)
        except (OSError, ValueError) as error:
            issues.append(Issue(Severity.ERROR, "record-unreadable", str(error), sample.id))
            continue
        if run_record is None:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "pending",
                    "the sample has no result yet; finish the run with `ocrbench run resume`",
                    sample.id,
                )
            )
            continue
        status, reason = run_record["status"], run_record["reason"]
        doc = ExportedDoc(sample.id, name or "", status, reason)
        docs.append(doc)
        if status == "failure":
            message = f"failure ({reason}): {run_record.get('message') or 'no message'}"
            if failed == "error":
                issues.append(
                    Issue(
                        Severity.ERROR,
                        "failed-sample",
                        f"{message}; retry it with `ocrbench run resume --retry-failed`, or "
                        "choose --failed empty or --failed omit",
                        sample.id,
                    )
                )
            else:
                doc.exported = "empty" if failed == "empty" else "omitted"
                what = "a document without blocks" if failed == "empty" else "no document"
                issues.append(
                    Issue(
                        Severity.WARNING,
                        "failed-sample",
                        f"{message}; exported as {what}",
                        sample.id,
                    )
                )
            continue
        if status != "success":
            issues.append(
                Issue(
                    Severity.WARNING,
                    f"{status}-sample",
                    f"the engine reported {status} output"
                    + (f": {run_record['message']}" if run_record.get("message") else "")
                    + "; its valid prediction is exported",
                    sample.id,
                )
            )
        try:
            prediction, _ = stored_prediction(run, sample, run_record)
        except EvaluationError as error:
            issues.append(Issue(Severity.ERROR, "invalid-prediction", str(error), sample.id))
            continue
        texts = prediction_blocks(prediction, blocks, fallback)
        if texts is None:
            issues.append(
                Issue(
                    Severity.ERROR,
                    "text-only-prediction",
                    "the prediction has page text but no text-bearing regions; use --blocks "
                    "page, or --fallback page to export its page text as one block",
                    sample.id,
                )
            )
            continue
        doc.fallback = (
            blocks == "regions"
            and not any(r.content is not None for r in prediction.regions)
            and bool(texts)
        )
        if doc.fallback:
            issues.append(
                Issue(
                    Severity.INFO,
                    "page-fallback",
                    "text-only prediction exported as one page block",
                    sample.id,
                )
            )
        replaced = 0
        for text in texts:
            clean, count = xml_text(text)
            replaced += count
            doc.blocks.append(clean)
        if replaced:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "xml-invalid-character",
                    f"{replaced} character(s) that XML 1.0 cannot represent (control characters "
                    "such as a form feed) were replaced by a space",
                    sample.id,
                )
            )
    return docs, issues


def export_report(run: Run, docs: list[ExportedDoc], output: Path, options: dict) -> dict:
    return {
        "format": EXPORT_REPORT_FORMAT,
        "version": EXPORT_REPORT_VERSION,
        "run_id": run.run_id,
        "run": str(run.path),
        "output": str(output),
        "options": options,
        "documents": [
            {
                "sample_id": d.sample_id,
                "source_name": d.source_name,
                "status": d.status,
                "reason": d.reason,
                "exported": d.exported,
                "blocks": len(d.blocks),
                "page_fallback": d.fallback,
            }
            for d in docs
        ],
    }


def export_archer(
    run_path: Path,
    output: Path,
    *,
    blocks: str = "page",
    fallback: str | None = None,
    failed: str = "error",
    dataset: Path | None = None,
    xsd: Path | None = None,
    force: bool = False,
    report: Path | None = None,
) -> tuple[list[ExportedDoc], list[Issue]]:
    """Write the run's predictions as one ARCHER hypothesis XML file.

    ``blocks``: ``page`` (the page text as one block) or ``regions`` (each text-bearing region).
    ``fallback="page"`` lets ``regions`` export a text-only prediction as one page block.
    ``failed``: a failed sample is an error (default), a ``<doc>`` without blocks (``empty``),
    or left out (``omit``); both are reported. Pending samples are always an error. Truncated and
    partial samples are exported with a warning. The file is checked by a round trip (and against
    ``xsd`` when given) before it replaces ``output``. Raises ``ArcherExportError`` and writes
    nothing on any error.
    """
    output = Path(output)
    try:
        run = load_run(run_path)
    except RunnerError as error:
        raise ArcherExportError([Issue(Severity.ERROR, "run-unreadable", str(error))]) from error
    docs, issues = collect(run, blocks=blocks, fallback=fallback, failed=failed, dataset=dataset)
    for path in (output, report):
        if path is not None and path.exists() and not force:
            issues.append(
                Issue(Severity.ERROR, "output-exists", f"{path} exists; use --force to replace it")
            )
    if any(issue.severity is Severity.ERROR for issue in issues):
        raise ArcherExportError(issues)

    data = hypothesis_xml(docs)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        _verify(Path(temporary), docs, xsd)
        if report is not None:
            options = {"blocks": blocks, "fallback": fallback, "failed": failed}
            text = json.dumps(
                export_report(run, docs, output, options), ensure_ascii=False, indent=1
            )
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(text + "\n", encoding="utf-8")
        os.chmod(temporary, 0o666 & ~_umask())
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return docs, issues


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _verify(path: Path, docs: list[ExportedDoc], xsd: Path | None) -> None:
    """The written file parses in the accepted form and holds exactly the intended text."""
    try:
        parsed = read_hypothesis(path)
    except ValueError as error:
        raise ArcherExportError([Issue(Severity.ERROR, "round-trip", str(error))]) from error
    expected = [(d.source_name, d.blocks) for d in docs if d.exported != "omitted"]
    if parsed != expected:
        raise ArcherExportError(
            [Issue(Severity.ERROR, "round-trip", "the written XML does not parse back unchanged")]
        )
    if xsd is not None:
        from lxml import etree

        try:
            schema = etree.XMLSchema(etree.parse(str(xsd)))
        except (OSError, etree.XMLSchemaParseError, etree.XMLSyntaxError) as error:
            raise ArcherExportError(
                [Issue(Severity.ERROR, "xsd-unreadable", f"{xsd}: {error}")]
            ) from error
        document = etree.parse(str(path))
        if not schema.validate(document):
            messages = [f"line {e.line}: {e.message}" for e in schema.error_log][:10]
            raise ArcherExportError(
                [Issue(Severity.ERROR, "xsd-invalid", f"against {xsd}: " + "; ".join(messages))]
            )
