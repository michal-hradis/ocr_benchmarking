# 0015: ARCHER hypothesis export from stored predictions

- Status: proposed
- Date: 2026-11-13
- Related issue: [#136](https://github.com/michal-hradis/ocr_benchmarking/issues/136)

## Context

ARCHER submissions are hypothesis XML files: `<archer-ocr>` with one `<doc>` per source image and plain-text `<block>` children. Runs already store standardized predictions ([ADR 0006](0006-standardized-prediction-format.md)) and their execution status ([ADR 0007](0007-runner-and-engine-protocol.md)); ARCHER references are imported per task with a sidecar ([ADR 0014](0014-archer-import-and-text-view.md)). ARCHER's documentation is inconsistent about the document identifier: D9.1 section 3.3 shows `<doc source_name="…">`, the Annex B XSD requires `<doc id="…">`.

## Decision

- `ocrbench export archer RUN -o FILE` writes one UTF-8 file from the stored `prediction.json` of every sample, revalidated as for scoring. No engine runs ([DATASETS.md](../DATASETS.md#exporting-archer-hypothesis-xml)).
- **Identifier.** `<doc source_name="…">`, the form of all 50 accepted local examples, with the task's `meta/source_name` from the import sidecar, matched by task ID = sample ID. A missing name is an error, never derived from the image file. Only this identifier is read from the import; reference text never influences the hypothesis.
- **Blocks.** `regions` (default): the engine's own blocks, i.e. each text-bearing region in the prediction's reading order, lines joined by `\n`; a text-only prediction has one block, its page text (`--fallback page`, default; `--fallback none` makes it an error). `page`: the standardized page text as one block. Whitespace-only text adds no block. (Amended 2026-11-27: the default was `page`, which discarded the engine's segmentation into blocks.)
- **Statuses.** Pending samples are always an error. Failures are an error by default; `--failed empty` (a `<doc>` without blocks) and `--failed omit` are explicit choices and are reported. Truncated and partial outputs are exported with a warning. `--report` records every sample's run status and how it was exported.
- **Coverage and integrity.** The run must cover the whole import, GT must be unchanged since the run was created, names must be unique. Any error writes nothing.
- **Verification.** The output is parsed back and compared with the intended text, and validated against `--xsd` when given, before it replaces the target.

## Alternatives considered

- *`page` blocks by default.* The original default; it merges all regions into one block and loses the engine's block segmentation, which the submission should carry.
- *`<doc id="…">` per Annex B.* Schema-conformant on paper, but no accepted example uses it and what `id` holds (task ID or file name) is not documented; guessing would risk a rejected submission.
- *Silently writing empty documents for failures.* Would look like valid empty transcriptions; rejected.
- *Allowing a subset of the import.* Easier for partial runs, but a submission missing documents would go unnoticed.
- *Stripping control characters or rejecting them.* XML 1.0 cannot contain them; they are replaced by a space with a warning, which keeps word boundaries.

## Consequences and verification

- By default a submission keeps the blocks the OCR system produced; a text-only engine yields one block per page, reported as a `page-fallback` info finding and `page_fallback` in the export report. `--fallback none` refuses such runs.
- If ARCHER confirms the Annex B `id` form, the identifier rule changes in one place and this ADR is superseded.
- `tests/test_archer_export.py` covers both block modes, text-only predictions, multi-line regions, empty text, Unicode and XML-special characters, statuses, coverage and identifier errors, overwrite protection, `--xsd`, the reader and a fake-engine run through the CLI, on the synthetic import fixture. Real ARCHER examples stay in ignored `dev_data/`.
