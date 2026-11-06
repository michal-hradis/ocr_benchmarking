# Dataset manifests and PAGE XML ground truth

A dataset is described by a **manifest**: a JSON file that pairs each sample image with its PAGE XML ground truth (GT). The reader extracts reference text, regions, lines and reading order from the GT. The decisions behind these rules are recorded in [#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5#issuecomment-6298136212) and [ADR 0005](adr/0005-page-xml-reference-extraction.md).

GT is **host-side only**. The local reader, evaluator and reports may use it. The runner ([RUNNER.md](RUNNER.md)) never passes a sample's `gt` path or anything derived from it to an engine container ([ADR 0002](adr/0002-ground-truth-isolation.md)).

## Commands

```sh
uv run ocrbench dataset validate datasets/dev-01.json      # add -v to list info findings
uv run ocrbench dataset text datasets/dev-01.json <sample-id> --ids
uv run ocrbench dataset import-archer <dir>                  # see Importing ARCHER OCR datasets
```

- **`validate`** checks the manifest and every sample. It prints findings and a per-sample summary of size, text regions, reference lines and characters. It exits with status 1 if there is any error.
- **`text`** prints a sample's reference lines in reading order, so you can check the extraction against the image by eye.

## Manifest format (version 1)

```json
{
  "format": "ocrbench.dataset-manifest",
  "version": 1,
  "name": "dev-01",
  "description": "optional free text",
  "samples": [
    {
      "id": "example-0001",
      "image": "images/example-0001.jpg",
      "gt": "page/example-0001.xml",
      "unit": "page",
      "doc_type": "newspaper",
      "language": ["ces"],
      "year": 1923,
      "split": "dev",
      "image_sha256": "…",
      "gt_sha256": "…"
    }
  ]
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Stable sample ID: `[A-Za-z0-9][A-Za-z0-9._-]*`, at most 128 characters, unique in the manifest. It is set explicitly and never derived from file order. |
| `image`, `gt` | yes | Paths to the image and to the PAGE XML, relative to the manifest file (absolute paths also work). Two samples must not share an image. |
| `unit` | no | `page` (default) or `paragraph`. A paragraph crop is its own image with its own PAGE XML in crop coordinates. Deriving crops from a parent page is not supported. |
| `doc_type`, `language`, `year`, `split` | no | Descriptive metadata. `language` is a list of strings. None of these is validated against a vocabulary, and no split is required. |
| `image_sha256`, `gt_sha256` | no | If present, the file must match the hash. Use them to pin the files a result was computed on. |

Unknown fields are errors, so typos do not pass silently. A new field or a change in meaning requires a new `version`.

[`datasets/dev-01.json`](../datasets/dev-01.json) references the local development set. It expects the files in `dev_data/data_01/`, which is ignored by Git and is not distributed. Currently that set holds 7 full pages.

## Reference extraction from PAGE XML

- **Versions.** Any namespace `http://schema.primaresearch.org/PAGE/gts/pagecontent/<date>` is accepted. Coordinates may be given as `points="x,y …"` or, in the 2009/2010 schemas, as `<Point>` children.
- **Regions.** Every `TextRegion` is read, including regions nested in another `TextRegion` and table cells inside a `TableRegion`. Other region types are not text sources.
- **Region order.** If the page has a `ReadingOrder`, regions follow it:
  - `OrderedGroup` members are sorted by `@index`, and nested groups are flattened in place.
  - An `UnorderedGroup` keeps document order and is reported as info.
  - A nested region that is not referenced itself follows its parent.
  - Other unreferenced text regions are appended in document order, with a warning.

  Without a `ReadingOrder`, regions keep XML document order, with an info finding.
- **Line order.** Lines within a region are sorted by `TextLine/@index` when every line has a distinct integer index. Otherwise they keep document order, with a warning when the indexes are only partial or duplicated.
- **Text precedence (no double counting).**
  - Each region contributes the text of its own lines.
  - A region's own `TextEquiv` usually aggregates everything inside it. It is therefore used only as a fallback for the region's **whole subtree**: when neither the region's lines nor any nested region (at any depth) provide text. It then becomes one reference unit, without a line ID.
  - Otherwise it is ignored, with an info finding. For example, a parent region whose text repeats a nested child's line yields that text once, from the child's line.
  - Text from `Word` and `Glyph` elements is not used.
- **Choice of `TextEquiv`.** When an element has several, the one with the lowest `@index` is used; otherwise the first one.
- **Missing text.** A `TextLine` without `TextEquiv/Unicode`, or with only whitespace, is skipped and reported as a warning. A page without any reference text is also a warning.
- **No normalization.** Text is kept exactly as annotated, including Unicode form, inner and outer whitespace and historic characters such as `ſ`. Normalization and the way lines are joined into page text belong to the evaluator ([EVALUATION.md](EVALUATION.md#text-preparation)).
- **Geometry.** Coordinates are kept as annotated. Points outside the image are counted and reported as info, never as an error. Code that uses geometry clips it to the image.

## Validation findings

| Severity | Meaning | Examples (`code`) |
| --- | --- | --- |
| error | The sample cannot be used, and `validate` fails | `missing-file`, `checksum-mismatch`, `image-unreadable`, `image-size-mismatch` (PAGE `imageWidth`/`imageHeight` differ from the actual image), `xml-parse-error`, `not-page-xml`, `missing-page-size`, `malformed-points`, manifest structure errors, duplicate sample IDs or images |
| warning | Usable, but the reference may not match the annotator's intent | `line-without-text`, `no-reference-text`, `region-not-in-reading-order`, `unknown-reading-order-ref`, `inconsistent-line-index`, `duplicate-id`, `exif-orientation` (engines may rotate the image while PAGE coordinates refer to the stored pixels) |
| info | Noteworthy only | `no-reading-order`, `unordered-group`, `region-text-ignored`, `points-out-of-bounds`, `image-filename-differs` |

Each finding names the sample, the file and, where possible, the PAGE element ID.

## Importing ARCHER OCR datasets

ARCHER OCR references (`<dataset><task>…` XML with labelled, percentage-based `block` boxes and a `relation from/to` graph) are converted into an ordinary manifest with PAGE XML references, so `dataset validate`, `run start`, `evaluate` and `report` work unchanged ([#135](https://github.com/michal-hradis/ocr_benchmarking/issues/135), [ADR 0014](adr/0014-archer-import-and-text-view.md)).

```sh
uv run ocrbench dataset import-archer dev_data/archer_examples/en      # reads src/ and ref/
uv run ocrbench dataset validate dev_data/archer_examples/en/ocrbench/manifest.json
```

`--src`, `--ref` and `-o` override the default `ROOT/src`, `ROOT/ref` and `ROOT/ocrbench`; `--name` overrides the dataset name (default `archer-<language>-text-v1`, so the text-view version appears in run IDs and reports). An existing import is replaced only with `--force`, and only if it contains exactly the files its `import.json` lists. A directory holding anything else (another tool's output or files added later) is never replaced, and the new import is written to a staging directory and moved into place only when it is complete. Keep the output under `dev_data/`: it contains real reference text.

**This scores an ARCHER dataset with OCRbench metrics** (strict and flexible CER/WER, bag-of-words, ROUGE-2). It is **not** the official ARCHER block-reconstruction CER/WER, which matches, splits and merges semantic blocks along the relation graph; fCER/fWER on an imported dataset must not be reported as ARCHER scores.

### Pairing and validation

Each `task` becomes one page sample with ID `task/@id`. Its image is the file in `src/` named by `data/image` or `meta/source_name`. The import **writes nothing** if any error is found:

| Error (`code`) | Cause |
| --- | --- |
| `missing-image` | Neither name exists; a file with the same stem and another suffix is named as a hint |
| `image-outside-src` | An image name is absolute or uses `..` (or a symlink) to leave the image directory |
| `ambiguous-image` | `data/image` and `meta/source_name` name two different existing files (if only one exists, the difference is a `source-name-differs` warning) |
| `image-reused`, `duplicate-task-id` | Two tasks share an image or an ID, also across reference files |
| `image-unreadable`, `image-size-mismatch`, `invalid-dimensions` | The image cannot be read, its size differs from `original_width`/`original_height`, or these are not positive integers |
| `xml-parse-error`, `not-archer-xml`, `no-task`, `missing-task-id`, `invalid-task-id` | Malformed file, wrong root, or a task ID that is not a valid sample ID |
| `unknown-label`, `missing-block-id`, `duplicate-block-id`, `malformed-bbox`, `malformed-relation`, `unknown-relation-block` | Block labels other than `Text`, `Table`, `Handwritten_annotations`, `Image`, `Noise`; non-numeric or negative boxes; relations naming no block |

Warnings and info findings (`source-name-differs`, unknown elements or attributes, `self-relation`, `duplicate-relation`, `no-reference-text`, `text-not-scored`, `no-geometry`, `empty-text`, `illegible-removed`, `bbox-out-of-bounds`) are printed and stored in `import.json`.

### Output

| File | Content |
| --- | --- |
| `manifest.json` | One sample per task, sorted by ID: image path (not copied), `gt`, `unit: page`, `language` from `dataset/@language` as annotated (for example `EN`), `year` from `meta/published_at`, `image_sha256` and `gt_sha256` |
| `page/<id>.xml` | PAGE 2019 evaluation reference, described below |
| `archer/<id>.json` | Lossless sidecar: source file, its SHA-256 and task position; dataset attributes; `meta`; every block's ID, label, raw `bbox` attributes, raw text, scored text and pixel polygon; all relations in document order |
| `archer/ref/<file>` | Unchanged copies of the original reference files |
| `import.json` | Importer and text-view versions, reference hashes, all findings and the list of files written |

The output is deterministic: no timestamps, and a re-import of the same files is byte-identical. The PAGE files' `gt_sha256` (frozen in every run) and their `Metadata/Comments` identify the source file, its hash and the text view.

**PAGE XML reference.** Every block becomes a region with ID `b_<block id>` (block IDs need not be valid XML IDs) and the original ID and label in `@custom`, in **annotation order**; no `ReadingOrder` is invented, so the reader reports `no-reading-order` and keeps that order. `Text` blocks are `TextRegion type="paragraph"`, `Table` blocks `TextRegion type="other"`, `Handwritten_annotations` a `GraphicRegion type="handwritten-annotation"`, `Image` an `ImageRegion` and `Noise` a `NoiseRegion`. The scored text is the region's own `TextEquiv`, with its line breaks, so each block is one reference unit: flexible matching can reorder whole blocks but does not split them, and no line coordinates are invented. Boxes are converted to pixel corners, rotated clockwise by `rotation` degrees about the top-left corner (the region's `@orientation`, where PAGE allows one, is the correcting angle). ARCHER writes an all-zero box for a block without a position. Such a region keeps its text. Because PAGE requires `Coords`, the region carries ARCHER's own empty box as `points="0,0 0,0 0,0 0,0"`, marked `geometry:none` in `@custom`. This is **not a position**: the sidecar's `polygon` is `null`, and code that uses reference geometry must skip such regions. The relations are mirrored as PAGE `Relation type="link"` elements (IDs `rel_1`, `rel_2`, … in document order) for reference only; the evaluator does not use them. The generated files validate against the PAGE 2019-07-15 schema (`tests/test_archer.py`).

### Text view `archer-text-v1`

The scored text of each block, applied identically to every sample:

- **Labels.** Only `Text` and `Table` blocks are transcription. `Handwritten_annotations` (non-textual handwritten additions such as signatures, arrows, ticks and underlines; transcribable handwriting is in `Text` blocks with `(HW)` spans), `Image` and `Noise` blocks are never scored. Any text on them stays in the sidecar and is reported as a `text-not-scored` warning.
- **Span tags.** Exactly `(PR)`, `(/PR)`, `(HW)` and `(/HW)` are removed. Printed and handwritten text are both scored; other parenthesized text, such as `(E)` in a UN document symbol, is kept.
- **Illegible text.** Each `(YYY)` marker is replaced by a **word break** (a space), so the text around it is never fused: `1(YYY)87` becomes `1 87`, not the untranscribed token `187`. A standalone marker leaves only spaces, which the evaluator collapses. No legible text is substituted, and an engine's reading of the illegible material counts against it as edits. Masking these spans out of scoring would need scorer support and is left to a future scorer. A block that becomes empty contributes no reference text (`empty-text`).
- **Tables** (`Table` blocks only). A pipe row `| a | b |` becomes its non-empty cells joined by single spaces (`a b`), `\|` becomes `|`, and a Markdown rule row such as `|---|:--:|` becomes an empty line. Other lines are kept.
- Nothing else changes: whitespace, Unicode and line breaks are kept and normalized only by the evaluator.

A change to these rules is a new text-view version and a re-import; stored predictions can then be re-evaluated without rerunning OCR. The sidecar keeps the raw text, tags and relations, so an official-style ARCHER scorer can be added later without re-importing.
