# Evaluation

This document defines the **initial** metric families and how they complement each other. Text preparation, ordered CER/WER, bag-of-words, ROUGE-2, aggregates and the failure policies are defined [below](#strict-metrics-metrics-version-1) ([#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8), [ADR 0008](adr/0008-strict-text-metrics.md)) and implemented. Flexible CER/WER are defined in [Flexible CER and WER](#flexible-cer-and-wer-metrics-version-2) ([#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9), [ADR 0009](adr/0009-flexible-line-alignment.md)) and implemented. System boundaries are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Inputs

The evaluator and reports run **locally on the same Linux machine** as the OCR runner. The evaluator compares a stored UTF-8 prediction (when present) against reference text from PAGE XML. The OCR container never receives GT. Evaluation uses stored execution records, predictions and reference GT; it does **not** rerun an engine. Scores use an independently **versioned evaluation configuration** for normalization, tokenization, alignment and failure policy.

## Initial metric families

| Family | What it measures | Order-sensitive |
| --- | --- | --- |
| Ordered CER/WER | Character and word edit distance (substitutions, deletions, insertions) between the full reference and the full prediction in reading order | Yes |
| Flexible CER/WER | The same edit costs after order-independent alignment of reference lines with prediction chunks, which tolerates differing column order and segmentation | No (within chunks: yes) |
| Bag-of-words P/R/F1 | Multiset word overlap: precision penalizes extra or hallucinated words, recall penalizes missing words | No |
| ROUGE-2 F1 | Word-bigram overlap, which captures local word order without global order | Locally |

How the families complement each other:

- **Ordered and flexible CER/WER.** A large gap between them indicates reading-order or segmentation differences rather than recognition errors. This is typical for multi-column newspapers.
- **Bag-of-words.** It is a cheap, alignment-free check on content: what was missed and what was added.
- **ROUGE-2.** It sits between bag-of-words and ordered scores. Its value drops when text is fragmented or locally scrambled.

## Accounting rules

The following hold for all families. #8 and #9 define the details.

- **All text counts.** For CER/WER, missing reference text contributes deletion costs and extra predicted text (including hallucinations or repetitions) contributes insertion costs. For bag-of-words and ROUGE, omissions and extraneous text affect their respective overlap/precision/recall calculations rather than edit operations. Flexible alignment must not credit content twice.
- Scores are kept **per sample** and **aggregated**. #8 defines the aggregates, for example micro and macro averages and separate strata for full pages and crops. Every score is identifiable by sample, run and evaluation version.
- Strict scoring preserves capitalization, diacritics and punctuation. Any normalized variants are named explicitly in the evaluation config.
- Markup in native outputs, such as Markdown or HTML, is not scored as words. Only the standardized plain transcription is evaluated.
- Text-only predictions without geometry are fully scorable.
- The design must scale to newspaper pages with 2,000+ reference lines, so a dense all-pairs comparison is not acceptable as the default.

## Strict metrics (metrics version 1)

Implemented in `ocrbench.metrics` and `ocrbench.evaluate`. Any change to these rules increments the metrics version, which is stored with every score. Metrics version 2 adds [flexible CER/WER](#flexible-cer-and-wer-metrics-version-2) and leaves these definitions unchanged.

### Text preparation

- **Reference:** the PAGE XML reference lines in reading order ([DATASETS.md](DATASETS.md#reference-extraction-from-page-xml)), joined by spaces.
- **Prediction:** the standardized page text of the stored `prediction.json` ([PREDICTIONS.md](PREDICTIONS.md)), never the native output. Markdown or HTML markup is removed by the host-side conversion before scoring and is never scored as words.
- **Unicode:** NFC by default. `NFKC` is a configured, explicitly named variant; it folds compatibility characters such as `ſ` → `s` and `ﬁ` → `fi`.
- **Whitespace:** every run of whitespace (spaces, tabs, line and region breaks, no-break spaces) becomes one space, and leading and trailing whitespace is removed. A line break and a space are therefore equal, and different line segmentation costs nothing, but spaces between words still count as characters.
- **Strict:** capitalization, diacritics, punctuation and historic characters are kept.
- **Characters** are Unicode code points after normalization. A combining mark without a precomposed form counts as its own character.
- **Words** are the space-separated tokens of the prepared text, with punctuation attached (`"Zeitung,"` is one word). CER/WER, bag-of-words and ROUGE-2 all use the same tokens.

### Ordered CER and WER

- Unit-cost Levenshtein alignment of the whole reference and the whole prediction, on characters (CER) or words (WER), computed with the pinned `rapidfuzz` library. It scales to newspaper pages with 2,000+ lines.
- Each sample stores the reference and hypothesis lengths, substitutions `S`, deletions `D` (reference text missing from the prediction) and insertions `I` (extra text, including hallucinations and repetitions).
- **Rate** = (S + D + I) / reference length. It is not capped at 1, since insertions can exceed the reference.
- The total is the edit distance. When several minimal alignments exist, one substitution can be traded for a deletion plus an insertion; the pinned library chooses deterministically.
- **Empty reference:** the rate is undefined (`null`), but the insertions stay in the counts and in the micro aggregate. Both empty: zero counts and `null`.

### Bag-of-words and ROUGE-2

- **Bag-of-words:** multiset overlap of words. `matched` = Σ min(reference count, predicted count) per word. Precision = matched / predicted words (penalizes extra words), recall = matched / reference words (penalizes missing words).
- **ROUGE-2:** the same overlap on word bigrams of the prepared text, so bigrams span line and region breaks. A text with fewer than two words has no bigrams. ROUGE-1 is not reported, as it equals bag-of-words.
- **F1** = 2·matched / (reference + predicted items). It equals the harmonic mean when precision and recall are defined, is 0 when exactly one side is empty, and is `null` when both are empty. Precision is `null` without predicted items, recall is `null` without reference items.

### Flexible CER and WER (metrics version 2)

Implemented in `ocrbench.alignment`. Flexible scores use the same text preparation and edit costs as ordered CER/WER, but tolerate a different column order and segmentation.

- **Units.** Reference units are the PAGE XML reference lines. Prediction units are the lines of the prediction's structure text: the prediction lines, the lines of a region's own text (such as a paragraph from an image-to-text model) and, when no region carries text, the lines of the page text. Units are normalized like the whole text; empty units are dropped. A paragraph is never split further, so a page without any line break is a single unit and is scored in order.
- **Groups.** A group pairs a run of reference lines that are consecutive in reading order with a run of consecutive prediction units (1:1, 1:N, N:1 or N:M). It costs the Levenshtein distance of the two runs, joined by spaces. Inside a group, order counts as in ordered CER; the groups themselves can appear in any order, at no cost. A unit belongs to at most one group, so no content is credited twice.
- **Unmatched units.** A reference line in no group costs its characters plus one separator as deletions; a prediction unit in no group costs the same as insertions. The separators between groups are matched. Without any group, the counts equal ordered CER/WER: a failure scored as an empty prediction deletes the whole reference.
- **Denominator.** The reference length is the same as for ordered CER (lines plus the spaces joining them), so the rates are directly comparable. The ordered alignment, all units in one group, is always a candidate: flexible CER/WER is never higher than ordered CER/WER. A large gap indicates reading-order or segmentation differences.
- **Words.** Flexible WER uses the same units and rules on words (without separators). It is optimized separately from flexible CER, so the two can use different groups.
- **Search.** Samples with at most 8 units on both sides are solved exhaustively. Larger samples avoid a dense comparison of every reference line with every prediction unit: prediction units are anchored to reference lines by shared character 3-grams; consecutive units with consistent anchors form blocks; the blocks are put in reference order and aligned with the reference; the reference is cut into groups at line boundaries, and each cut is shifted by up to two lines where that lowers the cost. The cheaper of this result and the ordered alignment is kept. A 2,200-line synthetic page takes a few seconds.
- **Approximation.** Every result is a valid grouping, so a non-optimal search can only overstate errors. Each sample stores a **lower bound**, `max(|reference|, |prediction|)` minus their multiset overlap of characters (or words), and **`exact`**, which is true when the result is proven optimal (exhaustive search, or errors equal to the bound). Missing text next to unrelated extra text typically leaves a gap between the bound and the result, and `exact` is false.
- **Diagnostics** per sample and metric: `method` (`exhaustive`, `ordered` or `reordered`), the number of `groups`, the groups themselves as `group_lines`, `order_breaks` (consecutive groups in prediction order that are not consecutive in reference order), and the unmatched reference and prediction units. `flexible_units` holds the reference and prediction unit counts and how many prediction units were anchored (retrieval coverage).
- **Stored groups.** `group_lines` lists every group, in prediction order, as `[reference start, reference stop, prediction start, prediction stop]`: half-open indices of the PAGE XML reference lines (the order of `ocrbench dataset text`) and of the prediction lines defined under *Units* (`ocrbench.alignment.prediction_lines`). Empty lines inside a range belong to no unit; a non-empty line outside every range is unmatched. Reports can show the matches and omissions from these ranges without recomputing the alignment.

### Aggregates

Aggregates are computed for all samples, for each unit (`page`, `paragraph`) and for each document type within a unit (`doc_type` from the manifest, frozen in `run.json`; `null` when not given, or for runs created before it was frozen).

- **Micro:** the counts summed over the group, then the rates, precision, recall and F1 from the sums. Long pages weigh more. Flexible CER/WER also sum their lower bounds (`lower_bound`, `lower_bound_rate`) and count the samples whose result is not proven optimal (`not_exact`); those samples stay in the aggregates.
- **Macro:** the mean of the per-sample values. Samples whose value is `null` (for example an empty reference) are left out, and their number is reported as `undefined`.
- Each group also reports its number of samples, scored samples, **coverage** (scored / all samples), status counts and failure reasons.

### Command and stored scores

```sh
uv run ocrbench evaluate runs/<run-id> [--config eval.toml]
```

It logs every non-`success` sample on stderr with its status, reason and how it was scored, and prints the micro scores per group, including flexible CER/WER (`fCER`, `fWER`). A flexible score marked `*` includes samples whose result is not proven optimal; a closing line gives their number for fCER and fWER and points to `exact` and `lower_bound` in `scores.json`. The command writes `runs/<run-id>/evaluations/<n>/scores.json` in a new numbered directory; earlier evaluations are never overwritten. The command exits with status 1, and writes nothing, when the run cannot be scored:

- a sample is still **pending** (finish the run with `run resume` first);
- a GT file **changed**, cannot be read, or was not pinned when the run was created (`ocrbench.runner.verify_ground_truth`), so predictions are only ever compared with the GT frozen with the run;
- the PAGE XML reader reports an error for the GT of **any** sample, whatever the failure policy (`exclude` only decides which scores are aggregated);
- a stored prediction no longer validates, including against the image size frozen in `run.json` (`image-size-mismatch`).

The configuration is a TOML file; every field is optional:

```toml
name = "strict"                  # label for the evaluation
unicode = "NFC"                  # NFC (default) or NFKC
failure_policy = "empty-output"  # empty-output (default) or exclude
```

`scores.json` (`ocrbench.scores`, version 2; version 1 had no flexible scores) records the run ID and the SHA-256 of `run.json`, the configuration and its hash, the metrics version, the fixed text rules, and the ocrbench and rapidfuzz versions. Per sample it holds the status, failure reason, attempt, GT hash, the SHA-256 of the scored `prediction.json` (`null` when no prediction was scored), `scored_as` (`prediction`, `empty` or `excluded`) and the counts and values of every metric, including `flexible_chars`, `flexible_words` and `flexible_units`; then the aggregates per group.

## Failed, partial and truncated samples

- **Primary default: `empty-output`.** Any failed invocation (timeout, crash, invalid prediction, etc.) is treated **by the evaluator** as an empty prediction for scoring. For CER/WER, its reference text contributes deletions, so failure on a difficult page cannot artificially improve overall accuracy. For bag-of-words and ROUGE-2, recall and F1 are 0 and precision is undefined (see above).
- **Explicit override: `exclude`.** A user may configure an alternate calculation that excludes failed samples from its metric aggregate; this is **not the default primary score**. Any excluded-sample score must be clearly labelled and shown with coverage (scored samples / all benchmark samples), with counts and reasons for failures. Results computed with different failure policies must not be silently compared.
- **`partial` and `truncated`** samples are scored using the output actually produced (including omissions); their status is retained. If the output cannot be validated as a prediction, record a `failure` instead.
- **Execution failure is not a successful empty transcription.** The runner records each sample's execution status/reason/provenance even if no valid OCR prediction exists. The evaluator may synthesize an **empty scoring input** for a failure; it must never silently invent or persist a successful OCR result.
- Every non-`success` sample is **logged** by `ocrbench evaluate`, and **counted and enumerated** in reports under both policies ([REPORTS.md](REPORTS.md#failures)). The policy is stored in the evaluation configuration and versioned score, and runs scored under different policies are never put in one report.

## Later extensions

Layout or region-detection scores, reading-order metrics and other metric families can be added later as new evaluation versions. None is required now.
