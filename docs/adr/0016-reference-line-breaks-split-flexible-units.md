# 0016: Reference line breaks split flexible units

- Status: proposed
- Date: 2026-11-13
- Related: [ADR 0009](0009-flexible-line-alignment.md), [ADR 0014](0014-archer-import-and-text-view.md)

## Context

Flexible CER/WER ([ADR 0009](0009-flexible-line-alignment.md)) group runs of consecutive reference lines with runs of consecutive prediction units; a unit is never split. Reference units were the PAGE XML reference lines. A region-level `TextEquiv` without `TextLine`s is one such line, even when its text has many line breaks. The ARCHER import ([ADR 0014](0014-archer-import-and-text-view.md)) writes every block this way, because ARCHER has no line coordinates. So a page of three table blocks had three reference units against 100+ prediction lines, and flexible CER equalled ordered CER whenever an engine read the blocks' lines in a different order or grouping. Prediction text is already split at line breaks (`ocrbench.alignment.prediction_lines`).

On the 50 ARCHER English pages, splitting the reference at line breaks lowers micro fCER from 24.7 to 13.6 % (PERO), 50.8 to 7.5 % (PaddleOCR), 11.9 to 5.1 % (PaddleOCR-VL) and 9.1 to 6.0 % (dots.mocr), with ordered CER unchanged.

## Decision

- Reference lines are split at `\n` inside their text before scoring (`ocrbench.evaluate.reference_lines`), as prediction text is. This is **metrics version 3**.
- Ordered CER/WER, bag-of-words and ROUGE-2 do not change: they join the reference lines with spaces and collapse whitespace, so a line break and a space are equivalent.
- `group_lines` indexes the split lines, which are the lines `ocrbench dataset text` prints without `--ids`. Reports' inspection pages use the same split.
- No line geometry is created; PAGE files and the ARCHER import are unchanged.

## Alternatives considered

- *Write `TextLine`s in the ARCHER import.* PAGE requires line `Coords`, which ARCHER does not have; inventing them is not allowed.
- *Split only ARCHER references.* The rule is about the text, not the source: any region-level reference with line breaks has the same problem.

## Consequences and verification

- Scores of metrics versions 2 and 3 cannot be reported together (reports require one metrics version); runs must be re-evaluated. Scores of PAGE references whose lines have no line breaks are identical.
- A reference line that a prediction splits across non-adjacent places (for example a table row `61 Name` against separate number and name columns) is still not split further; bag-of-words reflects that case.
- `tests/test_evaluate.py` checks that a block with a line break scores exactly like the same lines as `TextLine`s.
