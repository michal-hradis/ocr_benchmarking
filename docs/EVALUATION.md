# Evaluation

This document defines the **initial** metric families and how they complement each other. Exact definitions are owned by [#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8) (normalization, tokenization, edit counts) and [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9) (flexible alignment). System boundaries are in [ARCHITECTURE.md](ARCHITECTURE.md).

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

## Failed, partial and truncated samples

- **Primary default: `empty-output`.** Any failed invocation (timeout, crash, invalid prediction, etc.) is treated **by the evaluator** as an empty prediction for scoring. For CER/WER, its reference text contributes deletions, so failure on a difficult page cannot artificially improve overall accuracy. Other metrics apply their normal empty-prediction conventions, detailed in #8.
- **Explicit override: `exclude`.** A user may configure an alternate calculation that excludes failed samples from its metric aggregate; this is **not the default primary score**. Any excluded-sample score must be clearly labelled and shown with coverage (scored samples / all benchmark samples), with counts and reasons for failures. Results computed with different failure policies must not be silently compared.
- **`partial` and `truncated`** samples are scored using the output actually produced (including omissions); their status is retained. If the output cannot be validated as a prediction, record a `failure` instead.
- **Execution failure is not a successful empty transcription.** The runner records each sample's execution status/reason/provenance even if no valid OCR prediction exists. The evaluator may synthesize an **empty scoring input** for a failure; it must never silently invent or persist a successful OCR result.
- Every non-`success` sample is **logged, counted and enumerated** in reports. The default/override is stored in the evaluation configuration and versioned score ([#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10)).

## Later extensions

Layout or region-detection scores, reading-order metrics and other metric families can be added later as new evaluation versions. None is required now.
