# 0009: Flexible line alignment for CER/WER

- Status: proposed
- Date: 2026-10-21
- Related issue: [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9) (maintainer decisions recorded in the implementing PR)

## Context

Ordered CER/WER ([ADR 0008](0008-strict-text-metrics.md)) penalize a newspaper read in a different column order as heavily as wrong text, and engines segment text differently: physical lines, paragraphs from image-to-text models, or a single page text. Pages have 2,000+ reference lines, so comparing every reference line with every predicted line (about 4 million edit distances) is not acceptable as the default. Content must never be credited twice, and failures must keep the #8 accounting.

## Decision

- **Units and groups.** Reference lines are matched with prediction units, the lines of the prediction's structure text (prediction lines, or the lines of a region's or the page's own text). A group pairs a run of reference lines consecutive in reading order with a run of consecutive prediction units and costs their Levenshtein distance. Groups can appear in any order at no cost; order inside a group counts. A unit belongs to at most one group.
- **Accounting.** An unmatched unit costs its length plus one separator as deletions or insertions; separators between groups are matched. The denominator is the ordered reference length, so flexible and ordered rates are comparable, and the ordered alignment is always a candidate (flexible ≤ ordered). Without a group, the counts equal ordered CER/WER, so a failure scored as empty output deletes the whole reference.
- **Words** use the same rules without separators and are optimized separately.
- **Search.** Exhaustive for samples with at most 8 units on each side; otherwise q-gram anchors, blocks of consistently anchored units in reference order, line-boundary cuts with local shifts, and exact group costs. The cheaper of this and the ordered alignment is kept. Parameters are constants of the metrics version, not evaluation configuration.
- **Approximation is reported.** Each result stores a lower bound (`max(|ref|, |hyp|)` minus the multiset overlap), `exact` (proven optimal), the method, groups, order breaks, unmatched units and retrieval coverage. Non-exact samples stay in the aggregates and are counted.
- **Versions.** `METRICS_VERSION` 2 and `ocrbench.scores` version 2 add `flexible_chars`, `flexible_words` and `flexible_units`; ordered metrics are unchanged.

## Alternatives considered

- *A GT line matched to one contiguous span of the flat prediction text.* Independent of prediction segmentation, but it ignores the engine's own units; the maintainer chose prediction lines as units.
- *Any GT lines, in any order, inside one prediction unit.* More lenient for whole-page text, but it ignores order inside a unit, where the engine did commit to an order.
- *A cost per order break* (as in CDER/TER). It mixes reading order into recognition scores; order breaks are reported instead.
- *Ignoring the spaces between lines.* Simpler, but not comparable with ordered CER.
- *Excluding non-exact samples from aggregates.* It would hide the hardest pages; they are flagged instead.
- *All-pairs edit distances with optimal assignment.* Exact for 1:1 matching, but dense and without N:M groups.

## Consequences and verification

- Documented in [EVALUATION.md](../EVALUATION.md#flexible-cer-and-wer-metrics-version-2), implemented in `ocrbench.alignment` and used by `ocrbench evaluate`.
- Tests cover hand-computed reordering, missing and extra lines, empty sides, splits and merges, order inside a unit, repeated identical lines, word-level alignment, Unicode forms, the exhaustive search against an independent enumeration, the scalable search against the exhaustive optimum and the ordered score, flagged inexact results, structure-text units, text-only paragraphs and a 2,200-line page with a bounded number of edit-distance computations.
- The lower bound is loose when missing text sits next to unrelated extra text or when confusions cancel out in the character counts, so `exact` can be false although the result is optimal; the reported value is still a valid, never optimistic score.
