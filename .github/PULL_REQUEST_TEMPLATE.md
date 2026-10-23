## Issue

Closes #

## Summary

- Changed behavior/files:
- Out-of-scope follow-ups:

## Verification

See [CONTRIBUTING.md](https://github.com/michal-hradis/ocr_benchmarking/blob/main/CONTRIBUTING.md#verification-and-tests). Record **only checks actually run**.

| Check or command | Result |
| --- | --- |
| | |

Checks not run and reasons:


## Development-set results (new/changed engines or configurations only)

For **each new or inference-affecting changed configuration**, test **all seven pages** in `datasets/dev-01.json` and paste **two Markdown tables here**: (1) overall metrics, one row per preset (CER, WER, fCER, fWER, BoW F1, ROUGE-2 F1, engine time), and (2) per-page results for all seven pages, with CER (and fCER when different) for each preset. Use the table templates and reporting requirements in [CONTRIBUTING.md](https://github.com/michal-hradis/ocr_benchmarking/blob/main/CONTRIBUTING.md#development-set-results-for-new-engines-and-configurations); [PR #69](https://github.com/michal-hradis/ocr_benchmarking/pull/69) is a filled-in example.

Record run IDs, preset/model and evaluation settings, hardware, number of successful/failed pages, and notable limitations. Mark failed pages, do not omit them. If access prevents the runs, explain why and leave the PR draft until results can be added. **For other kinds of PR, remove this section.**

## Risks and handoff

- Data/model/GPU access and reproducibility notes (if applicable):
- Documentation/ADR changes:
- Remaining risks, decisions and follow-ups:
- Reviewer requested:
