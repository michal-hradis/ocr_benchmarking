# 0011: Region markup converted on the host

- Status: proposed
- Date: 2026-10-25
- Related issue: [#67](https://github.com/michal-hradis/ocr_benchmarking/issues/67)

## Context

Layout models such as Surya 2 and Chandra OCR 2 return positioned blocks whose text is HTML or Markdown. The shared text-conversion rules ([CONTRIBUTING.md](../../CONTRIBUTING.md#shared-text-conversion-for-engine-outputs)) require their plain text to follow `ocrbench.convert` and their boxes and reading order to be kept. The converter works only on whole documents and builds regions without positions, and it runs on the host: an adapter in its container cannot import it. Chandra OCR 2 therefore kept the shared conversion and lost the boxes; dots.mocr keeps the boxes and repeats the converter's inline Markdown rules in the adapter.

## Decision

- A prediction region may carry `markup` (`{"format": "markdown" | "html", "content": "…"}`) instead of `text` and `lines`.
- When the host validates a prediction (the runner, `ocrbench prediction validate`), it converts every region's markup with `ocrbench.convert.fragment_text`. That function applies the rules of `convert_native` to one fragment and joins its blocks by line breaks. The standardized prediction holds the plain text and records `conversion` with the converter version.
- The engine keeps its native output and writes the markup unchanged; an empty page text is rebuilt from the converted regions.
- Rules are in [PREDICTIONS.md](../PREDICTIONS.md#markup-in-positioned-regions). The field is optional and additive; prediction format version 1 is kept.

## Alternatives considered

- *Whole-document host conversion (Chandra OCR 2).* Keeps the shared rules but drops the boxes from the prediction.
- *A copy of the converter in each engine image.* Keeps the boxes, but conversion happens in the container: a converter fix needs an image rebuild and a re-run, and copies can drift.
- *Engine-specific conversion code (dots.mocr).* Rejected by the shared text-conversion rule.

## Consequences and verification

- Engines with positioned markup need no conversion code. Surya 2 ([#67](https://github.com/michal-hradis/ocr_benchmarking/issues/67)) is the first; Chandra OCR 2 and dots.mocr could follow in separate changes.
- A converter rule change re-converts stored predictions when they are validated again; `conversion.version` identifies the rules used.
- `tests/test_convert.py` checks `fragment_text` against the document rules, and `tests/test_prediction.py` checks the conversion, the kept positions and reading order, the rebuilt page text and the conflicts.
