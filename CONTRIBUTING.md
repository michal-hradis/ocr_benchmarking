# Contributing to OCR Benchmarking

Start with [README](README.md), the assigned issue, and [architecture](docs/ARCHITECTURE.md). See [docs/ISSUE_WORKFLOW.md](docs/ISSUE_WORKFLOW.md) for preparing and refining issues; coding agents also follow [AGENTS.md](AGENTS.md).

## Work on an issue

1. Confirm dependencies, relevant ADRs, existing issues/PRs and the current `main` branch.
2. Create a focused topic branch from `origin/main`. Keep each PR to one reviewable purpose.
3. Implement the agreed scope, including relevant tests and documentation.
4. Run checks appropriate to the files and behavior changed, then open a PR against `main` and request human review.
5. Report what changed, exact verification performed, what was not run, limitations and follow-ups. Use `Closes #N` only when the PR fully addresses that issue; merge only after review.

Do not push directly to `main` or silently change an agreed metric, schema, or cross-engine contract.

## Verification and tests

**There is no CI at the start of the project. The developer or agent runs relevant local checks for the PR; running the full test suite is not generally required.** Report commands and actual results in the PR, including any material checks skipped or unavailable.

| Change | Relevant local checks |
| --- | --- |
| Documentation | Read changed pages; verify relative links and claims about commands |
| PAGE XML / dataset loader | Focused parsing and validation fixtures, including malformed input |
| Normalization / CER / WER / ROUGE | Targeted numerical regression and Unicode/tokenization tests |
| Flexible line matching | Small exact-reference tests; representative 2,000+ line stress case |
| JSON schema / adapters | Contract tests for positioned and text-only output |
| Engine Docker/inference | Adapter tests; an explicitly requested local CPU/GPU smoke run if feasible |
| Reporting | Fixture-based aggregate, failure and truncation checks |

Add regression tests for changed behavior. Normal developer tests should use synthetic deterministic fixtures without downloading models or requiring GPUs, paid APIs or external data. Real inference and the separately held six-image development set require deliberate, authorized invocation; if unavailable, report that limitation instead of claiming a pass.

Do not add CI workflows or a universal full-suite PR gate as part of unrelated changes. Once [#4](https://github.com/michal-hradis/ocr_benchmarking/issues/4) implements tooling, document actual supported commands here.

## Data and runtime hygiene

Respect [the engine/evaluator boundary](docs/ARCHITECTURE.md): images may be provided to OCR engines; PAGE XML ground truth must remain isolated from inference. Commit only expressly redistributable synthetic fixtures, never real scans/GT, secrets, model weights, caches or run outputs. Pin versions and capture relevant inference configuration and failures for reproducibility.

For Docker, GPU or model-dependent changes, do not silently install huge models or start costly work. Record which configuration and resources were actually tested.

## PR handoff

Use [the PR template](.github/PULL_REQUEST_TEMPLATE.md) to link the issue and list verification, risks and remaining work. A documentation-only PR may require only manual document/link checks; it must not invent a passing test command.
