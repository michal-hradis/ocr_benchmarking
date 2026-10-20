# 0008: Strict text metrics and failure accounting

- Status: proposed
- Date: 2026-10-20
- Related issue: [#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8) (maintainer decisions: [comment](https://github.com/michal-hradis/ocr_benchmarking/issues/8#issuecomment-6302335401))

## Context

Every engine is compared through the same scores, so small choices in how text is prepared and counted change the ranking: Unicode form, how line breaks count, what a character and a word are, how empty pages and failed invocations are scored, and how per-sample scores are averaged. ARCHITECTURE.md and EVALUATION.md fix the families (ordered CER/WER, bag-of-words, ROUGE-2) and the empty-output failure default, but leave these definitions to #8. Newspaper pages with 2,000+ lines need a full-page alignment of about 100,000 characters.

## Decision

- **Text preparation.** The reference is the PAGE XML reference lines in reading order; the prediction is the standardized page text, never native markup. Both are NFC-normalized (NFKC as a named variant). Every whitespace run, including line and region breaks, becomes one space, and the ends are stripped. Case, diacritics and punctuation are kept.
- **Units.** Characters are code points after normalization, not grapheme clusters. Words are whitespace tokens with punctuation attached. All word metrics use the same tokens.
- **CER/WER** come from a unit-cost Levenshtein alignment of the whole texts, computed with the pinned `rapidfuzz` library, with stored substitution, deletion and insertion counts. The rate is errors per reference unit, not capped. For an empty reference it is `null`, while its insertions stay in the counts.
- **Bag-of-words and ROUGE-2** are multiset overlaps of words and word bigrams with precision, recall and F1 = 2·matched / (reference + predicted). ROUGE-1 is not reported.
- **Failures** are scored as empty predictions by default (`empty-output`). `exclude` leaves them out of the aggregates and reports coverage. `partial` and `truncated` outputs are scored as produced. The evaluator never creates a prediction for a failure.
- **Aggregates.** Micro (summed counts) and macro (mean of defined per-sample values, with the number left out) are computed overall, per unit, and per document type within a unit.
- **Identity.** Scoring refuses runs with pending samples or with GT that differs from the GT frozen in `run.json`. Each evaluation is written to a new directory with the configuration and its hash, the metrics version, the text rules and the library versions.

## Alternatives considered

- *Grapheme clusters as characters.* Closer to what a reader sees for combining marks in historic prints, but it needs another dependency and differs from common OCR tools. It can be added later as a new metrics version.
- *Newlines as characters, or ignoring all whitespace.* The first penalizes line segmentation that the reference does not prescribe. The second hides word splits and merges in CER.
- *A defined rate of 0 or 1 for empty references.* Every sample would have a rate, but the value would be arbitrary. A `null` rate with counts keeps hallucinations visible in the micro score.
- *Pure-Python edit distance.* No dependency, but far too slow for full newspaper pages.
- *Splitting punctuation into separate tokens.* Deferred; it would be a named variant.

## Consequences and verification

- The definitions are documented in [EVALUATION.md](../EVALUATION.md#strict-metrics-metrics-version-1) and implemented in `ocrbench.metrics`, `ocrbench.evaluate` and `ocrbench evaluate`. Any rule change increments `METRICS_VERSION`; the configuration has its own version.
- Tests cover hand-computed S/D/I and denominators, hallucination, missing text, repetition, split and merged words, line breaks, empty pages, Unicode normalization, a 2,500-line page, comparison against a pure-Python edit distance, micro versus macro averages, strata, both failure policies with coverage, truncated output, Markdown output, and refusal of changed GT and pending samples.
- Flexible alignment (#9) and reports (#10) build on these counts and must use the same text preparation.
