# 0014: ARCHER references imported as PAGE XML with a versioned text view

- Status: proposed
- Date: 2026-11-06
- Related issue: [#135](https://github.com/michal-hradis/ocr_benchmarking/issues/135)

## Context

ARCHER OCR references describe a page as labelled blocks with percentage-based, possibly rotated boxes, block-level transcriptions with `(PR)`/`(HW)` span tags and `(YYY)` illegibility markers, Markdown-like tables and a directed relation graph between blocks. The benchmark's evaluator reads PAGE XML ([ADR 0005](0005-page-xml-reference-extraction.md)). The issue asks for evaluation with existing OCRbench metrics without engine changes, while keeping the original reference for a future official scorer.

## Decision

- `ocrbench dataset import-archer` converts each task into a **PAGE XML evaluation reference** plus a **lossless JSON sidecar** and a copy of the original file ([DATASETS.md](../DATASETS.md#importing-archer-ocr-datasets)). The manifest format, PAGE reader, runner and evaluator are unchanged.
- Each block is one region with region-level text; regions keep **annotation order** and no reading order is derived from the relation graph. Blocks without a box get no coordinates.
- The scored text follows the versioned **text view `archer-text-v1`**: only `Text` and `Table` are scored; `Handwritten_annotations` (non-textual additions), `Image` and `Noise` are not. Span tags are removed, each `(YYY)` marker becomes a word break so that text around illegible material is never fused, and table pipe rows become space-separated cells.
- The PAGE output is **schema-valid PAGE 2019**. A block without a box carries ARCHER's own empty box (`0,0 0,0 0,0 0,0`), marked `geometry:none`, because PAGE requires `Coords`; it is not a position.
- `--force` replaces only an import whose files match its own recorded file list, through a staged swap; image names must stay inside the image directory.
- The version is in the default dataset name, the PAGE metadata and `import.json`; the GT hashes frozen in each run pin it.
- Scores are **OCRbench metrics on an ARCHER dataset**, not official ARCHER scores.

## Alternatives considered

- *A native ARCHER reference reader in the evaluator.* No intermediate files, but a second GT format in the evaluator and reports, for the same scores.
- *Lines split from block text, or an order derived from the relations.* Splitting would need invented line coordinates; following the Text→Text relations is closer to ARCHER's reconstruction but invents an order for unlinked blocks. Flexible CER/WER already tolerate block order.
- *Keeping `(YYY)` in the reference.* Every engine would pay for the literal marker.
- *Deleting `(YYY)`.* Fuses the surrounding text into tokens the annotator never transcribed (`1(YYY)87` → `187`) and can credit an engine for them.
- *Masking illegible spans.* The fairest option, but the current metrics have no masking; left to a future scorer.
- *Omitting `Coords` for blocks without a box.* Makes the output invalid PAGE; a PAGE-like internal format was rejected in favour of schema validity with a marked empty box.

## Consequences and verification

- Strict CER/WER depend on the annotation order of blocks; flexible metrics can reorder blocks but never split them. A long block is one unit.
- `tests/test_archer.py` validates the generated PAGE XML against the vendored PAGE 2019-07-15 schema and checks pairing and validation errors, path containment, `--force` ownership, the lossless sidecar, geometry and rotation, the text view (tags, illegibility, tables, empty text, unscored labels), byte-identical re-import and that job entries carry no reference data, on a synthetic fixture.
