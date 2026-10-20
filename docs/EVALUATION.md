# Evaluation

This document defines the **initial** metric families and how they complement each other. Text preparation, ordered CER/WER, bag-of-words, ROUGE-2, aggregates and the failure policies are defined [below](#strict-metrics-metrics-version-1) ([#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8), [ADR 0008](adr/0008-strict-text-metrics.md)) and implemented. Flexible alignment is owned by [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9) and not implemented yet. System boundaries are in [ARCHITECTURE.md](ARCHITECTURE.md).

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

Implemented in `ocrbench.metrics` and `ocrbench.evaluate`. Any change to these rules increments the metrics version, which is stored with every score.

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

### Aggregates

Aggregates are computed for all samples, for each unit (`page`, `paragraph`) and for each document type within a unit (`doc_type` from the manifest, frozen in `run.json`; `null` when not given, or for runs created before it was frozen).

- **Micro:** the counts summed over the group, then the rates, precision, recall and F1 from the sums. Long pages weigh more.
- **Macro:** the mean of the per-sample values. Samples whose value is `null` (for example an empty reference) are left out, and their number is reported as `undefined`.
- Each group also reports its number of samples, scored samples, **coverage** (scored / all samples), status counts and failure reasons.

### Command and stored scores

```sh
uv run ocrbench evaluate runs/<run-id> [--config eval.toml]
```

It prints the micro scores per group and writes `runs/<run-id>/evaluations/<n>/scores.json` in a new numbered directory; earlier evaluations are never overwritten. The command exits with status 1, and writes nothing, when the run cannot be scored:

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

`scores.json` (`ocrbench.scores`, version 1) records the run ID and the SHA-256 of `run.json`, the configuration and its hash, the metrics version, the fixed text rules, and the ocrbench and rapidfuzz versions. Per sample it holds the status, failure reason, attempt, GT hash, the SHA-256 of the scored `prediction.json` (`null` when no prediction was scored), `scored_as` (`prediction`, `empty` or `excluded`) and the counts and values of every metric; then the aggregates per group.

## Failed, partial and truncated samples

- **Primary default: `empty-output`.** Any failed invocation (timeout, crash, invalid prediction, etc.) is treated **by the evaluator** as an empty prediction for scoring. For CER/WER, its reference text contributes deletions, so failure on a difficult page cannot artificially improve overall accuracy. For bag-of-words and ROUGE-2, recall and F1 are 0 and precision is undefined (see above).
- **Explicit override: `exclude`.** A user may configure an alternate calculation that excludes failed samples from its metric aggregate; this is **not the default primary score**. Any excluded-sample score must be clearly labelled and shown with coverage (scored samples / all benchmark samples), with counts and reasons for failures. Results computed with different failure policies must not be silently compared.
- **`partial` and `truncated`** samples are scored using the output actually produced (including omissions); their status is retained. If the output cannot be validated as a prediction, record a `failure` instead.
- **Execution failure is not a successful empty transcription.** The runner records each sample's execution status/reason/provenance even if no valid OCR prediction exists. The evaluator may synthesize an **empty scoring input** for a failure; it must never silently invent or persist a successful OCR result.
- Every non-`success` sample is **logged, counted and enumerated** in reports. The default/override is stored in the evaluation configuration and versioned score ([#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10)).

## Later extensions

Layout or region-detection scores, reading-order metrics and other metric families can be added later as new evaluation versions. None is required now.
