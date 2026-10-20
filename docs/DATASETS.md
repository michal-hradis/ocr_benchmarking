# Dataset manifests and PAGE XML ground truth

A dataset is described by a **manifest**: a JSON file that pairs each sample image with its PAGE XML ground truth (GT). The reader extracts reference text, regions, lines and reading order from the GT. The decisions behind these rules are recorded in [#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5#issuecomment-6298136212) and [ADR 0005](adr/0005-page-xml-reference-extraction.md).

GT is **host-side only**. The local reader, evaluator and reports may use it. The runner ([#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7)) must never pass a sample's `gt` path or anything derived from it to an engine container ([ADR 0002](adr/0002-ground-truth-isolation.md)).

## Commands

```sh
uv run ocrbench dataset validate datasets/dev-01.json      # add -v to list info findings
uv run ocrbench dataset text datasets/dev-01.json <sample-id> --ids
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
  - If any line of a region has text, the region's reference is its line texts, and region-level text is ignored.
  - If no line has text, the region-level `TextEquiv` is used as one reference unit, without a line ID.
  - Text from `Word` and `Glyph` elements is not used.
- **Choice of `TextEquiv`.** When an element has several, the one with the lowest `@index` is used; otherwise the first one.
- **Missing text.** A `TextLine` without `TextEquiv/Unicode`, or with only whitespace, is skipped and reported as a warning. A page without any reference text is also a warning.
- **No normalization.** Text is kept exactly as annotated, including Unicode form, inner and outer whitespace and historic characters such as `ſ`. Normalization and the way lines are joined into page text belong to the evaluator ([#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8)).
- **Geometry.** Coordinates are kept as annotated. Points outside the image are counted and reported as info, never as an error. Code that uses geometry clips it to the image.

## Validation findings

| Severity | Meaning | Examples (`code`) |
| --- | --- | --- |
| error | The sample cannot be used, and `validate` fails | `missing-file`, `checksum-mismatch`, `image-unreadable`, `image-size-mismatch` (PAGE `imageWidth`/`imageHeight` differ from the actual image), `xml-parse-error`, `not-page-xml`, `missing-page-size`, `malformed-points`, manifest structure errors, duplicate sample IDs or images |
| warning | Usable, but the reference may not match the annotator's intent | `line-without-text`, `no-reference-text`, `region-not-in-reading-order`, `unknown-reading-order-ref`, `inconsistent-line-index`, `duplicate-id`, `exif-orientation` (engines may rotate the image while PAGE coordinates refer to the stored pixels) |
| info | Noteworthy only | `no-reading-order`, `unordered-group`, `region-text-ignored`, `points-out-of-bounds`, `image-filename-differs` |

Each finding names the sample, the file and, where possible, the PAGE element ID.
