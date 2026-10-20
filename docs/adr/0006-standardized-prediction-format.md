# 0006: Standardized prediction format and host-side conversion

- Status: proposed
- Date: 2026-10-20
- Related issue: [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6) (maintainer decisions: [comment](https://github.com/michal-hradis/ocr_benchmarking/issues/6#issuecomment-6299290606))

## Context

Engines range from line-based OCR with exact polygons to image-to-text models that emit Markdown or HTML without any positions. ADR 0003 requires plain page text, a versioned JSON prediction and preserved native output, but leaves the fields to #6. The evaluator (#8, #9) and reports (#10) need one format that every engine can fill honestly: no fabricated lines or coordinates, no markup in the page text, and no ambiguity about which text is scored.

## Decision

- A prediction is versioned JSON (`ocrbench.prediction`, version 1) with a required **page text**, a **flat** list of regions and an optional **reading order** of region IDs. Without a reading order, regions are read in array order and the prediction has no explicit order.
- **Lines are optional.** A region has one text source: its lines when present, otherwise its own text. Having both is a warning, and the lines win. There is no marker for detected versus derived lines.
- **Region type** is optional and drawn from a small recommended vocabulary; other values are allowed. The engine's own label is kept as `native_type`.
- **Page text must agree** with the line/region text joined in reading order (`\n` between lines, a blank line between regions), ignoring whitespace-only differences. On disagreement the validator warns and rebuilds the page text from the structure. How text is joined for scoring stays with #8.
- **Positions are optional** bounding polygons (≥ 3 points) and baselines in float pixel-edge coordinates of the original stored image. The prediction declares the image size, which must match the real image. Out-of-bounds points are **clipped with a warning**, never an error; degenerate polygons are warnings.
- **Common native formats** (plain text, Markdown, HTML) are converted by **shared framework code on the host**, from the stored native output, after inference. The conversion is versioned and recorded in the prediction; native output is never modified.
- Structure is defined by a JSON Schema shipped with the package; cross-field rules are checked by `ocrbench.prediction`. Any error makes the prediction invalid, and the runner (#7) records that as a failure, never as a success.
- Execution status, failures and provenance stay with the runner (ADR 0003). Confidence values are optional and not used by metrics.

## Alternatives considered

- *Nested regions (as in PAGE XML).* Rejected: few engines produce them, and nesting complicates text precedence and matching. Tables and other inner structure stay in the native output.
- *Requiring lines in every region.* Rejected: it would force fake line splits onto formulas, tables and image-to-text paragraphs.
- *Rejecting predictions with out-of-bounds coordinates.* Rejected: overshoot is common and harmless after clipping.
- *Axis-aligned boxes only.* Rejected: polygons cover skewed and curved lines, and a box is the 4-point case.
- *Per-engine Markdown/HTML converters inside the containers.* Rejected: inconsistent rules across engines, and every fix would require re-running models.
- *Page text and structure text as independent outputs.* Rejected: it leaves it unclear which text is scored.

## Consequences and verification

- The format is documented in [PREDICTIONS.md](../PREDICTIONS.md) and implemented in `ocrbench.prediction`, `ocrbench.convert` and `src/ocrbench/schemas/prediction-v1.schema.json`. `ocrbench prediction validate|convert` expose them.
- Fixtures in `tests/fixtures/predictions/` cover a positioned multi-column newspaper page with out-of-bounds points, a text-only result, and Markdown/HTML outputs with headings, lists, tables, formulas and embedded HTML.
- #9 must handle regions and lines without positions and line segmentation that differs from the reference.
- Changing the conversion rules increments the converter version; changing the format increments the prediction version. Both must be considered when comparing evaluation results.
