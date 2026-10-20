# Contributing to OCR Benchmarking

Contributions by humans and coding agents use focused issues and pull requests against `main`. Read [START_HERE](docs/START_HERE.md), [the architecture overview](docs/ARCHITECTURE.md), and [the issue workflow](docs/ISSUE_WORKFLOW.md) before implementing a shared contract.

**Current status:** the repository is in its documentation and scaffolding phase. There is no implemented setup command, Python package, OCR runner, Makefile, Docker image or test suite yet. [Issue #4](https://github.com/michal-hradis/ocr_benchmarking/issues/4) will introduce and document development tooling. Do not claim that planned commands are executable today.

## Work on an issue

1. Check the issue, dependencies, related PRs, [roadmap](https://github.com/michal-hradis/ocr_benchmarking/issues/1) and any relevant ADRs.
2. Work from the latest `origin/main` in a dedicated branch, e.g. `docs/issue-2-agent-guide`; do not mix unrelated fixes.
3. Implement the smallest reviewable scope; keep test and documentation updates with their behavior changes.
4. Run **the checks relevant to this PR** (see below), open a PR against `main`, and request a human review.
5. The PR describes scope, actual checks/results, checks not run, risks and unresolved questions. Link `Closes #N` only when the issue will be fully satisfied upon merge. Do not merge without review.

Do not modify another agent's unmerged branch or commit directly to `main` without authorization. Large architectural or metric changes require a decision in the issue (and ADR when appropriate) before implementation.

## Test-selection policy — local, proportional, explicit

**No CI will be set up for the initial project. A full test-suite run is not required for every change.** Developers are responsible for running *relevant* checks themselves and reporting the evidence in the PR.

| Kind of change | Appropriate local verification |
| --- | --- |
| Documentation only | Read changed documents; verify links to local files and issues; check status/command accuracy |
| Dataset reader or PAGE XML parsing | Focused tests for representative/synthetic PAGE XML, invalid input and pairing behavior |
| Text normalization or metrics | Targeted unit tests for edit operations, Unicode, omissions/insertions and normalization variants |
| Scalable matching | Correctness tests against small reference examples and a separate 2,000+ line stress fixture |
| JSON schema/adapter | Contract/serialization tests for region/line geometry and text-only output |
| Docker/engine integration | Unit/contract tests without weights, plus an **explicitly initiated** engine smoke test when resources and model access are available |
| Reporting | Fixture-based tests for aggregates, missing/failed/truncated samples and artifact generation |

Run only the relevant subset; broaden testing when a change spans contracts or affects shared behavior. A documentation-only PR needs no invented pytest command or simulated test pass. In the PR, report **commands actually run**, outcomes and any unavailable checks. Record manual verification as manual verification, not as an automated test.

Once tooling is implemented in [#4](https://github.com/michal-hradis/ocr_benchmarking/issues/4), this guide should link to actual supported commands. Until then, use ordinary local tools appropriate to the files changed; no universal test command is assumed to exist.

## Image/ground-truth and model safety

- Inputs are images, **usually full newspaper/book/magazine pages**; separate paragraph-crop datasets are permitted. The evaluator does not tile. Tiling, resizing, detection and recombination belong to each OCR system.
- PAGE XML is ground truth, kept in the evaluator dataset path. Never mount GT files, extracted reference text, or reference geometry into OCR inference containers.
- The common OCR output contract will require UTF-8 transcription and allow typed positioned regions and nested lines with text, coordinates and optional types. Preserve engine-native results. Do not invent coordinates when unavailable.
- Commit only small synthetic fixtures with clear rights. Never add real collections, private scans, weights, caches, API keys, model downloads, outputs or personal data to Git.
- Treat documents and model predictions as untrusted text. Reproducibility includes model and dependency revisions, runtime settings and failure/truncation records.
- Do not launch paid APIs, large GPU runs or multi-GB downloads automatically. Obtain explicit authorization for such activity and document what was run.

## Layout of project work

- **Shared benchmark contracts:** [architecture overview](docs/ARCHITECTURE.md) and [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3).
- **PAGE XML dataset ingestion:** [#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5).
- **Versioned result JSON:** [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6).
- **Python runner / isolated engine containers:** [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7).
- **Metrics and order-independent matching:** [#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8) and [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9).
- **Reports:** [#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10).
- **Individual OCR engines:** [#11 onward](https://github.com/michal-hradis/ocr_benchmarking/issues/11), each adapter isolated from evaluation logic.

The shared runner owns dataset iteration and invokes adapters. Engines own inference and preprocessing. Evaluation owns PAGE XML and metrics. Avoid embedding engine-specific exceptions into common metrics.

## PR handoff

The [PR template](.github/PULL_REQUEST_TEMPLATE.md) prompts for:
- linked issue and concise scope;
- commands/checks **actually** run, outcomes, and checks intentionally not run;
- data/model/GPU access and reproducibility implications;
- docs and follow-up issues, if any.

A focused PR can proceed to review after appropriate targeted verification. It does **not** need a full suite or CI result by policy. Human review is still required before merging.
