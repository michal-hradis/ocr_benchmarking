# 0011: Region markup converted on the host

- Status: accepted
- Date: 2026-10-25
- Related issue: [#67](https://github.com/michal-hradis/ocr_benchmarking/issues/67)

## Context

Layout models such as Surya 2 and Chandra OCR 2 return positioned blocks whose text is HTML or Markdown. The shared text-conversion rules ([CONTRIBUTING.md](../../CONTRIBUTING.md#shared-text-conversion-for-engine-outputs)) require their plain text to follow `ocrbench.convert` and their boxes and reading order to be kept. The converter works only on whole documents and builds regions without positions, and it runs on the host: an adapter in its container cannot import it. Chandra OCR 2 therefore kept the shared conversion and lost the boxes; dots.mocr keeps the boxes and repeats the converter's inline Markdown rules in the adapter.

## Decision

- Text conversion happens only on the host, in shared code (`ocrbench.convert`). Engines never convert markup in their containers, and no engine has its own conversion code.
- A prediction region may carry `markup` (`{"format": "markdown" | "html", "content": "…"}`) instead of `text` and `lines`.
- When the host validates a prediction (the runner, `ocrbench prediction validate`), it converts every region's markup with `ocrbench.convert.fragment_text`. That function applies the rules of `convert_native` to one fragment and joins its blocks by line breaks. The standardized prediction holds the plain text and records `conversion` with the converter version.
- The engine keeps its native output and writes the markup unchanged; an empty page text is rebuilt from the converted regions.
- Rules are in [PREDICTIONS.md](../PREDICTIONS.md#markup-in-positioned-regions). The field is optional and additive; prediction format version 1 is kept.

### Amendment: line markup (2026-11-06, [#91](https://github.com/michal-hradis/ocr_benchmarking/issues/91))

- A positioned **line** may carry `markup` instead of `text`, for engines whose recognizer writes inline markup into line text (Surya's `<b>`, `<sup>`, `<br>`, entities). The host converts it with `ocrbench.convert.line_text`: `fragment_text` with its line breaks joined by single spaces, so a line stays one line.
- A line whose converted text is empty is dropped from the standardized prediction on the host; the other lines keep their IDs and order. A line with both `markup` and `text` is a `markup-conflict` error, and the one-format rule covers region and line markup together.
- dots.mocr writes every element's text as `markdown` region markup: its HTML tables are HTML blocks and its `$$…$$` formulas math blocks of the Markdown, so one format per prediction suffices. Its `Picture` elements get no markup, because the prompt asks for no text there (engine semantics, not conversion).
- The field is optional and additive; prediction format version 1 is kept.

## Alternatives considered

- *Whole-document host conversion (Chandra OCR 2).* Keeps the shared rules but drops the boxes from the prediction.
- *A copy of the converter in each engine image.* Keeps the boxes, but conversion happens in the container: a converter fix needs an image rebuild and a re-run, and copies can drift.
- *Engine-specific conversion code (dots.mocr).* Rejected by the shared text-conversion rule.

## Consequences and verification

- Engines with positioned markup need no conversion code. Surya 2 ([#67](https://github.com/michal-hradis/ocr_benchmarking/issues/67)) is the first; dots.mocr (region Markdown) and Surya (line markup) followed in [#91](https://github.com/michal-hradis/ocr_benchmarking/issues/91). Chandra OCR 2 already converts on the host and can keep its boxes the same way.
- A converter rule change re-converts stored predictions when they are validated again; `conversion.version` identifies the rules used.
- `tests/test_convert.py` checks `fragment_text` against the document rules, and `tests/test_prediction.py` checks the conversion, the kept positions and reading order, the rebuilt page text and the conflicts.
