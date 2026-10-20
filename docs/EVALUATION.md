# Evaluation

This document defines the **initial** metric families and how they complement each other. Exact definitions are owned by [#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8) (normalization, tokenization, edit counts) and [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9) (flexible alignment). System boundaries are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Inputs

The evaluator compares the standardized UTF-8 prediction of one sample with the reference text extracted from its PAGE XML. It reads only stored run outputs, so it never runs an engine. Its scores depend on a **versioned evaluation configuration** covering normalization, tokenization, alignment and failure policy, which is stored with every result.

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

- **All text counts.** Missing reference text counts as deletions. Extra predicted text, including hallucinations and repetitions, counts as insertions. Flexible alignment must not credit any content twice.
- Scores are kept **per sample** and **aggregated**. #8 defines the aggregates, for example micro and macro averages and separate strata for full pages and crops. Every score is identifiable by sample, run and evaluation version.
- Strict scoring preserves capitalization, diacritics and punctuation. Any normalized variants are named explicitly in the evaluation config.
- Markup in native outputs, such as Markdown or HTML, is not scored as words. Only the standardized plain transcription is evaluated.
- Text-only predictions without geometry are fully scorable.
- The design must scale to newspaper pages with 2,000+ reference lines, so a dense all-pairs comparison is not acceptable as the default.

## Failed, partial and truncated samples

- **`partial` and `truncated`** samples are scored as produced, and their status is kept with the score.
- **`failure`** samples (timeouts, crashes, invalid output) are handled by a **configurable policy** in the evaluation config:
  - *empty-output*: the sample is scored as an empty prediction, so all its reference text counts as deletions;
  - *exclude*: the sample is left out of metric aggregates, and coverage (scored samples / total samples) is reported with each aggregate.

  An evaluation can compute both variants. The chosen policy is part of the evaluation version.
- Whatever the policy, every non-`success` sample is **logged, counted and enumerated** in generated reports, with its status and reason ([#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10)).

## Later extensions

Layout or region-detection scores, reading-order metrics and other metric families can be added later as new evaluation versions. None is required now.
