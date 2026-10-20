# Standardized OCR predictions

A **prediction** is the standardized JSON form of one engine's valid OCR result for one sample. It holds the plain page text and, when the engine provides them, typed regions with optional lines and optional positions. The decisions behind the format are recorded in [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6#issuecomment-6299290606) and [ADR 0006](adr/0006-standardized-prediction-format.md). The machine-readable structure is the JSON Schema [`prediction-v1.schema.json`](../src/ocrbench/schemas/prediction-v1.schema.json); the rules that span several fields are checked by `ocrbench.prediction` and described below.

A prediction describes only a **valid** OCR result. Execution status, failure reasons and provenance are recorded by the runner ([RUNNER.md](RUNNER.md)), including for samples without any valid prediction ([ADR 0003](adr/0003-engine-run-output-contract-boundaries.md)). A prediction with any validation error is invalid: it must never be treated as a successful result.

## Commands

```sh
uv run ocrbench prediction validate pred.json --image page.jpg --sample-id <id> [-o standardized.json]
uv run ocrbench prediction convert page.md --sample-id <id> [-o pred.json]   # .txt, .md, .html
```

- **`validate`** prints findings and exits with status 1 if there is an error. `--image` checks the declared image size against the real image; `-o` writes the standardized prediction (clipped positions, rebuilt page text).
- **`convert`** converts a native plain-text, Markdown or HTML output into a validated prediction. The format follows the suffix unless `--from` is given.

## Format (version 1)

```json
{
  "format": "ocrbench.prediction",
  "version": 1,
  "sample_id": "example-0001",
  "image": {"width": 2480, "height": 3508},
  "text": "Die Zeitung\n\nerste Zeile\nzweite Zeile",
  "regions": [
    {
      "id": "r1",
      "type": "heading",
      "native_type": "title",
      "polygon": [[10, 10], [900, 10], [900, 80], [10, 80]],
      "text": "Die Zeitung"
    },
    {
      "id": "r2",
      "type": "text",
      "lines": [
        {"id": "l1", "text": "erste Zeile", "polygon": [[10, 100], [900, 100], [900, 140], [10, 140]],
         "baseline": [[10, 132], [900, 132]], "confidence": 0.97},
        {"id": "l2", "text": "zweite Zeile"}
      ]
    }
  ],
  "reading_order": ["r1", "r2"]
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `format`, `version` | yes | `"ocrbench.prediction"` and `1`. A change of meaning requires a new version. |
| `sample_id` | yes | The manifest sample ID ([DATASETS.md](DATASETS.md)). The validator checks it when the expected ID is known. |
| `text` | yes | Plain UTF-8 page transcription, without Markdown or HTML. It may be empty. |
| `image` | when positions are given | `width` and `height` of the original input image in pixels. |
| `regions` | no | A **flat** list of regions; regions are never nested. Absent or empty for a text-only result. |
| `reading_order` | no | Region IDs in reading order. Absent means the engine gave **no explicit reading order**, and regions are read in array order. |
| `conversion` | no | `{"from": "markdown", "version": 1}` when the prediction was converted on the host from a native output (see below). |
| `extra` | no | Engine-specific data. Allowed on the prediction, regions and lines; never used by metrics. |

**Region:** `id` (required, unique among regions), `type`, `native_type` (the engine's own label), `polygon`, `text`, `lines`, `confidence`, `extra`.

**Line:** `text` (required), `id` (optional, unique among lines), `type` (free label, for example `handwritten`), `polygon`, `baseline`, `confidence`, `extra`.

Unknown fields are errors. Confidence values are engine-specific numbers; metrics do not use them.

### Region types

The recommended types are `text`, `heading`, `list-item`, `caption`, `footnote`, `page-header`, `page-footer`, `page-number`, `table`, `formula`, `figure` and `other`. The field is optional. Any other value is allowed and reported as info, so the vocabulary can grow without a new format version.

### Text

- **One text source per region.** A region's text comes from its lines when it has any; otherwise from its own `text`. A region with both gets a warning, and its own `text` is dropped. A region may have no text at all, such as a figure.
- **Lines are optional.** A formula, a table or a paragraph from an image-to-text model can be a region with its own text and no lines. A line is whatever the engine reports; there is no marker for detected versus derived lines.
- **Page text agrees with the structure.** When any region carries text, the **structure text** is built in reading order: lines of a region joined by `\n`, regions joined by a blank line (`\n\n`). The page text must equal it up to whitespace: the comparison treats any run of whitespace as one separator and ignores leading and trailing whitespace.
  - If they agree, the page text is kept as the engine wrote it.
  - If not, there is a `text-mismatch` warning and the **line/region text has priority**: the standardized page text is replaced by the structure text.
- **No normalization.** Text is kept exactly as produced. Normalization and the way lines are joined for scoring belong to the evaluator ([EVALUATION.md](EVALUATION.md#text-preparation)). The joining above is used only for the agreement check and the rebuilt page text.

### Positions

- **Positions are optional** on regions and lines and are never fabricated. A text-only model has no regions, or regions without positions.
- A **`polygon`** is a list of at least 3 `[x, y]` points, in either direction; an axis-aligned box is the 4-point case. A line's **`baseline`** is a polyline of at least 2 points.
- **Coordinates** are float pixel coordinates of the **original input image** as stored (EXIF orientation is not applied), measured from pixel edges: pixel `i` spans `[i, i+1)`, so the image spans `[0, W] × [0, H]`. An engine that resizes, rotates or tiles the image maps positions back before writing the prediction.
- `image` must be present when any position is given. If it differs from the actual image, that is an **error** (`image-size-mismatch`): the positions belong to a different image.
- **Out-of-bounds points are clipped** to `[0, W] × [0, H]` with one `points-out-of-bounds` warning per prediction. A polygon with zero area gets a `degenerate-polygon` warning. Neither is an error.

## Host-side conversion of native output

Engines keep their native output (TXT, Markdown, HTML, hOCR, PAGE XML, …) **unchanged**. Common formats are converted into predictions by shared framework code (`ocrbench.convert`) on the host, after inference, so a conversion fix means re-converting stored output, never re-running a model. Every converted prediction records `conversion.from` and the converter `version`; any rule change increments that version.

Converted predictions contain plain text and typed regions **without positions** (`r1`, `r2`, … in document order, with no explicit reading order). Blocks without text are dropped, and the page text is the region texts joined by blank lines.

| Native element | Region type | Text |
| --- | --- | --- |
| Plain text (`.txt`) | none | The whole file, unchanged, as page text |
| Heading (`#`, `<h1>`–`<h6>`) | `heading` | Inline text |
| Paragraph, loose text in a container | `text` | Inline text; soft and hard line breaks and `<br>` become `\n` |
| List item | `list-item` | An ordered item keeps its number (`3. `); a bullet marker is dropped. Nested items are separate regions. |
| Table (Markdown or HTML) | `table` | One row per line, cells separated by a tab; a nested table's text stays in its cell |
| `<caption>`, `<figcaption>` | `caption` | Inline text |
| `$$…$$`, block `<math>` | `formula` | TeX source; for MathML the TeX annotation when present, otherwise its text |
| Code block, `<pre>` | `other` | Verbatim |

Inline rules: emphasis, links and other markup keep only their text; inline `$…$` math keeps its TeX source; images (including alt text), horizontal rules, `<script>`, `<style>` and `<head>` add nothing; HTML entities and Markdown escapes are decoded. Whitespace is collapsed as a browser would render it (runs of spaces become one space), except in code blocks and `<pre>`. HTML embedded in Markdown uses the HTML rules: an HTML block is converted as HTML, and a paragraph containing inline HTML has its text converted with the HTML inline rules, so the content of inline `<script>` or `<style>` adds nothing while text in tags such as `<span>` is kept.

Known limitations of converter version 1: `\[…\]` and `\(…\)` math delimiters are treated as ordinary Markdown escapes, and a whole page wrapped in a code fence stays one `other` region.

## Validation findings

| Severity | Meaning | Examples (`code`) |
| --- | --- | --- |
| error | The prediction is invalid and must not be treated as a successful result | `prediction-unreadable`, `prediction-json-error` (including `NaN`/`Infinity`), `schema-violation`, `invalid-unicode`, `sample-id-mismatch`, `image-size-mismatch`, `missing-image-size`, `invalid-coordinate`, `duplicate-id`, `unknown-reading-order-ref`, `duplicate-reading-order-ref` |
| warning | Usable; the standardized result was fixed | `text-mismatch`, `region-text-ignored`, `region-not-in-reading-order` (appended in array order), `points-out-of-bounds`, `degenerate-polygon` |
| info | Noteworthy only | `unknown-region-type`, `empty-prediction` |

Each finding names the sample (when known), the file and a JSON pointer to the element, for example `pred.json#/regions/3/lines/0/polygon`.
