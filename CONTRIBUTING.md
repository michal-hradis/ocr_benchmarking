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

Add regression tests for changed behavior. Normal developer tests should use synthetic deterministic fixtures without downloading models or requiring GPUs, paid APIs or external data. Real inference and the separately held development set (`dev_data/`, see [docs/DATASETS.md](docs/DATASETS.md)) require deliberate, authorized invocation; if unavailable, report that limitation instead of claiming a pass.

Do not add CI workflows or a universal full-suite PR gate as part of unrelated changes.

## Development setup and commands

Requirements: Linux, `make`, and [uv](https://docs.astral.sh/uv/) ≥ 0.13 (for example `pip install --user uv`). uv downloads Python 3.12 if the host lacks it; the host Python is not modified.

```sh
make setup   # create .venv from uv.lock
make check   # ruff lint + format check, then offline unit tests
```

| Command | Purpose |
| --- | --- |
| `make lint` / `make format` | Ruff check and format check / apply fixes and formatting |
| `make test` | Offline unit tests; select tests with `ARGS`, e.g. `make test ARGS="tests/test_package.py -k version"` |
| `uv run ocrbench dataset validate <manifest>` | Validate a dataset manifest and its PAGE XML, see [docs/DATASETS.md](docs/DATASETS.md) |
| `uv run ocrbench prediction validate\|convert …` | Validate a prediction JSON or convert native text/Markdown/HTML, see [docs/PREDICTIONS.md](docs/PREDICTIONS.md) |
| `uv run ocrbench run start\|resume\|status …` | Run an engine image over a dataset and record every sample, see [docs/RUNNER.md](docs/RUNNER.md) |
| `uv run ocrbench evaluate <run-dir> [--config eval.toml]` | Score a finished run against its frozen GT, see [docs/EVALUATION.md](docs/EVALUATION.md#command-and-stored-scores) |
| `uv run ocrbench report <run-dir>... -o <new-dir> [--inspect]` | Compare evaluated runs in CSV, JSON and HTML, see [docs/REPORTS.md](docs/REPORTS.md) |
| `make docker-cpu`, `make docker-cu130-torch`, `make docker-cu130-paddle` | Build base images |
| `make docker-tesseract`, `make tesseract-smoke` | Build the Tesseract engine image / opt-in: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/tesseract](engines/tesseract/README.md) |
| `make docker-paddleocr`, `make paddleocr-smoke` | Build the PaddleOCR engine image / opt-in, GPU: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/paddleocr](engines/paddleocr/README.md) |
| `make docker-kraken`, `make kraken-smoke` | Build the Kraken engine image / opt-in, GPU: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/kraken](engines/kraken/README.md) |
| `make docker-smoke`, `make gpu-preflight`, `make gpu-smoke` | Opt-in Docker and GPU checks (including the fake-engine runner tests), see [docs/DOCKER.md](docs/DOCKER.md) |

`make help` lists all targets. Unit tests run with network sockets disabled (`pytest-socket`). Tests that need Docker or a GPU are marked `docker` / `gpu` and run only with `--run-docker` / `--run-gpu`. Add dependencies with `uv add` (or `uv add --dev`) so that `uv.lock` is updated in the same PR. Fixture rules are in [tests/fixtures/README.md](tests/fixtures/README.md).

## Data and runtime hygiene

Respect [the engine/evaluator boundary](docs/ARCHITECTURE.md): images may be provided to OCR engines; PAGE XML ground truth must remain isolated from inference. Commit only expressly redistributable synthetic fixtures, never real scans/GT, secrets, model weights, caches or run outputs. Pin versions and capture relevant inference configuration and failures for reproducibility.

For Docker, GPU or model-dependent changes, do not silently install huge models or start costly work. Record which configuration and resources were actually tested.

## PR handoff

Use [the PR template](.github/PULL_REQUEST_TEMPLATE.md) to link the issue and list verification, risks and remaining work. A documentation-only PR may require only manual document/link checks; it must not invent a passing test command.
