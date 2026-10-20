# 0005: PAGE XML reference extraction and dataset manifests

- Status: proposed
- Date: 2026-10-20
- Related issue: [#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5) (maintainer decisions: [comment](https://github.com/michal-hradis/ocr_benchmarking/issues/5#issuecomment-6298136212))

## Context

Every score compares a prediction with reference text extracted from PAGE XML. PAGE allows text on regions, lines, words and glyphs, several alternative `TextEquiv` elements, an optional reading order, and coordinates outside the image. If extraction counted text twice, silently dropped it or used an unstable order, CER/WER would change without any change in the engine. Annotation tools also commonly let polygons overshoot the image edge.

## Decision

- Datasets are described by a versioned **JSON manifest** with explicit, stable sample IDs and paths relative to the manifest. Metadata (document type, language, year, split) is optional, and no split is required.
- A **paragraph-crop** sample is an independent image with its own PAGE XML in crop coordinates. Deriving crops from a parent page is out of scope.
- Reference text comes from **line** `TextEquiv` when any line in a region has text. Region text is used only as a fallback, so the same text is never counted twice. Word and glyph text are not used.
- A line without text is **skipped with a warning**.
- Region order follows `ReadingOrder` when present, and XML document order otherwise. Line order follows `TextLine/@index` when it is complete and unique, and document order otherwise.
- Text is stored **exactly as annotated**. Normalization and joining lines into page text belong to the evaluator ([#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8)).
- A mismatch between the PAGE `imageWidth`/`imageHeight` and the actual image size is an **error**. Coordinates are kept as annotated, and out-of-bounds points are only reported.

## Alternatives considered

- *YAML or JSONL manifests.* YAML adds a dependency and its parsing quirks. JSONL needs a separate header for the version and dataset name.
- *Treating a line without text as an error.* That would be stricter, but it would block datasets with occasional unlabelled lines.
- *A parent page plus a line selection for paragraph crops.* Deferred: it needs coordinate shifting and a check of the selection.
- *Failing on out-of-bounds geometry.* Rejected: annotation tools allow it, and the current metrics use only text. The size check catches GT that belongs to a different image or resolution.

## Consequences and verification

- The rules are documented in [DATASETS.md](../DATASETS.md) and implemented in `ocrbench.pagexml` and `ocrbench.manifest`. `ocrbench dataset validate` reports findings with their severity.
- Synthetic fixtures in `tests/fixtures/pagexml/` cover these cases: reading order that differs from document order, legacy namespaces with `<Point>` coordinates, nested regions and table cells, repeated lines, missing text, region-text fallback and out-of-bounds points.
- Changing any of these rules changes the reference text, so it requires a new ADR and must be considered when comparing evaluation versions.
