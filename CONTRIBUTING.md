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


### Shared text conversion for engine outputs

**Use the shared conversion rules for recognized text, not engine-specific markup cleaners.** Engines must keep native output unchanged. When the model returns Markdown or HTML, the conversion happens on the host, in the shared `ocrbench.convert` converter ([prediction rules](docs/PREDICTIONS.md#host-side-conversion-of-native-output), [ADR 0011](docs/adr/0011-region-markup-converted-on-the-host.md)), never inside the engine container. Do not copy its heading, emphasis, link, list, table, formula, entity, escape or whitespace handling into an adapter. The standardized prediction must contain recognized **plain text**, not formatting characters such as Markdown `*italic*` markers. This is **format extraction**, not permission to correct or normalize the model's spelling, characters or wording.

For **positioned structured output** (for example, a JSON array of regions with bounding boxes and Markdown text), preserve the original regions, polygons, IDs and reading order. The existing `convert_native()` interface converts a **whole document to unpositioned regions**, so do **not** feed the entire positioned result to it. Instead, write each region's markup unchanged into the region's `markup` field ([rules](docs/PREDICTIONS.md#markup-in-positioned-regions)); the host converts it with the shared rules and keeps the structure. If the shared code does not support a case, extend `ocrbench.convert` with focused tests. Do not maintain an independent Markdown/HTML parser or partial formatting-regex implementation in each engine.

For a new conversion path, add deterministic fixtures for the markup that engine can actually emit (as applicable: bold/italic, links, headings, lists, HTML entities and inline tags, tables, formulas, escapes, whitespace). Check that native output remains unchanged, markup is not counted as OCR text, and any genuine positions and reading order survive. Shared conversion-rule changes must follow the documented converter-versioning rules; do not silently change cross-engine scoring semantics.

### Development-set results for new engines and configurations

Every PR that **adds a new OCR/HTR engine or predefined model/configuration/preset**, or **changes an existing configuration in a way that affects inference**, must evaluate **each new or affected configuration on all seven development pages** in [`datasets/dev-01.json`](datasets/dev-01.json). Synthetic fixtures and smoke tests remain necessary but do not replace this evaluation. If a PR adds two presets, run and report both; unchanged configurations do not need to be rerun.

Use the frozen engine configuration and the normal `ocrbench run start` → `ocrbench evaluate` workflow (see [runner](docs/RUNNER.md) and [evaluation](docs/EVALUATION.md)). Use the default evaluation configuration for comparability, or state and justify any changes. The seven page images and PAGE XML GT remain in the separately held `dev_data/` directory: do not commit them, their predictions, or generated run/report artifacts. Real inference still requires authorized access to the data, models and hardware.

**Post the results in the PR description as two Markdown tables**, following [PR #69's dev-01 results](https://github.com/michal-hradis/ocr_benchmarking/pull/69): an overall table with **one row per configuration** and a per-page table with **all seven distinct pages for every configuration**. Do not select only the best pages or report one preset in place of another.

**1. Overall results (micro-averaged unless identified otherwise).** Include strict CER/WER, flexible CER/WER (fCER/fWER), bag-of-words F1, ROUGE-2 F1 and engine time. For example, replace the placeholders:

| Preset / configuration | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `<preset>` | … | … | … | … | … | … | … |

**2. Per-page results.** Include the per-page CER for each configuration and fCER in parentheses when different; line counts are useful for diagnosing layout/detection failures. For multiple presets, add one score column per preset as in PR #69; for one preset, keep just its column. Replace every placeholder, including for failures (mark failures explicitly, rather than silently dropping a page):

| Page | Content | Lines (preset A / B) | Preset A CER (fCER) | Preset B CER (fCER) |
| --- | --- | ---: | ---: | ---: |
| `3e6309d0…` | Czech newspaper | … | … | … |
| `bd497a62…` | Czech newspaper | … | … | … |
| `CkXII61-0017` | Glagolitic print | … | … | … |
| `dta-346672…` | Latin, Antiqua and italic | … | … | … |
| `dta-laube…` | German Fraktur (1833) | … | … | … |
| `dta-martzi…` | German Fraktur (1692) | … | … | … |
| `dta-steinen…` | German Antiqua (1894) | … | … | … |

Alongside the tables, identify the run IDs, engine/preset and model versions, relevant inference parameters, image digest (if available), hardware, evaluation configuration, and whether engine time includes model loading. Give the success/failure count **per configuration**, describe failed pages and major accuracy or reading-order limitations, and distinguish measured facts from speculation. The development set is small and not representative; do not present these numbers as general model rankings.

If the seven-page data, model access or hardware are unavailable, state exactly what is blocked, keep the PR **draft/not ready to merge** until the results are supplied, and never invent results or silently substitute synthetic fixtures. This is a required handoff for new/changed engine configurations, not permission to access private data or run costly inference without authorization.

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
| `make docker-tesseract`, `make tesseract-best`, `make tesseract-smoke` | Build the Tesseract engine image / opt-in: download the pinned `tessdata_best` models (48 MB) for the best presets into the cache / opt-in: run all presets through `run`, `evaluate` and `report` on synthetic samples, see [engines/tesseract](engines/tesseract/README.md) |
| `make docker-paddleocr`, `make paddleocr-smoke` | Build the PaddleOCR engine image / opt-in, GPU: run both presets through `run`, `evaluate` and `report` on synthetic samples, see [engines/paddleocr](engines/paddleocr/README.md) |
| `make docker-ppstructurev3`, `make ppstructurev3-smoke` | Build the PP-StructureV3 layout pipeline image / opt-in, GPU: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/ppstructurev3](engines/ppstructurev3/README.md) |
| `make docker-paddleocr-vl`, `make paddleocr-vl-model`, `make paddleocr-vl-smoke` | Build the PaddleOCR-VL-1.6 document parsing image / opt-in: download its pinned 2 GB of models into `OCRBENCH_CACHE` / opt-in, GPU: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/paddleocr-vl](engines/paddleocr-vl/README.md) |
| `make docker-ocrd-calamari`, `make ocrd-calamari-smoke` | Build the standalone OCR-D/Calamari CPU image / opt-in: run it and its presets through `run`, `evaluate` and `report` on synthetic samples, see [engines/ocrd-calamari](engines/ocrd-calamari/README.md) |
| `make docker-surya`, `make surya-model`, `make surya-smoke` | Build the Surya OCR engine image / opt-in: download its pinned 3 GB of checkpoints into `OCRBENCH_CACHE` / opt-in, GPU: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/surya](engines/surya/README.md) |
| `make docker-doctr`, `make doctr-model`, `make doctr-smoke` | Build the docTR engine image / opt-in: download its pinned 129 MB of checkpoints into `OCRBENCH_CACHE` / opt-in, GPU: run both presets through `run`, `evaluate` and `report` on synthetic samples, see [engines/doctr](engines/doctr/README.md) |
| `make docker-kraken`, `make kraken-smoke` | Build the Kraken engine image / opt-in, GPU: run it and its presets through `run`, `evaluate` and `report` on synthetic samples, see [engines/kraken](engines/kraken/README.md) |
| `make docker-pylaia`, `make pylaia-smoke` | Build the PyLaia engine image (Kraken segmentation, PyLaia recognition in a second Python 3.10 venv, with its model) / opt-in, GPU: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/pylaia](engines/pylaia/README.md) |
| `make docker-lightonocr`, `make lightonocr-model`, `make lightonocr-smoke` | Build the LightOnOCR-2 engine image / opt-in: download its pinned 2 GB weights into `OCRBENCH_CACHE` (default `~/.cache/ocrbench`) / opt-in, GPU: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/lightonocr](engines/lightonocr/README.md) |
| `make docker-fireredocr`, `make fireredocr-model`, `make fireredocr-smoke` | The same for FireRed-OCR (4.3 GB of weights), see [engines/fireredocr](engines/fireredocr/README.md) |
| `make docker-ovisocr2`, `make ovisocr2-model`, `make ovisocr2-smoke` | The same for OvisOCR2 (1.7 GB of weights), see [engines/ovisocr2](engines/ovisocr2/README.md) |
| `make docker-glm-ocr`, `make glm-ocr-model`, `make glm-ocr-smoke` | The same for GLM-OCR (2.7 GB of weights), see [engines/glm-ocr](engines/glm-ocr/README.md) |
| `make docker-mineru25-pro`, `make mineru25-pro-model`, `make mineru25-pro-smoke` | The same for MinerU2.5-Pro (2.3 GB of weights), see [engines/mineru25-pro](engines/mineru25-pro/README.md) |
| `make docker-chandra-ocr2`, `make chandra-ocr2-model`, `make chandra-ocr2-smoke` | The same for Chandra OCR 2 (10.6 GB of weights under a modified OpenRAIL-M license), see [engines/chandra-ocr2](engines/chandra-ocr2/README.md) |
| `make docker-olmocr2`, `make olmocr2-model`, `make olmocr2-smoke` | The same for olmOCR-2 (16.6 GB of weights), see [engines/olmocr2](engines/olmocr2/README.md) |
| `make docker-deepseek-ocr2`, `make deepseek-ocr2-model`, `make deepseek-ocr2-smoke` | The same for DeepSeek-OCR-2 (6.8 GB of weights and the model's own code), see [engines/deepseek-ocr2](engines/deepseek-ocr2/README.md) |
| `make docker-hunyuanocr`, `make hunyuanocr-model`, `make hunyuanocr-smoke` | The same for HunyuanOCR-1.5 (2.2 GB of weights under the Tencent Hunyuan Community License), see [engines/hunyuanocr](engines/hunyuanocr/README.md) |
| `make docker-dots-mocr`, `make dots-mocr-model`, `make dots-mocr-smoke` | The same for dots.mocr (6.1 GB of weights and the model's own code; positioned layout output), see [engines/dots-mocr](engines/dots-mocr/README.md) |
| `make docker-cu130-vllm`, `make docker-surya2`, `make surya2-model`, `make surya2-smoke` | Build the vLLM base image / the same for Surya 2 (1.4 GB of weights under a modified OpenRAIL-M license, served by vLLM inside the container; positioned layout output), see [engines/surya2](engines/surya2/README.md) |
| `make docker-cu130-vllm026`, `make docker-nemotron-parse`, `make nemotron-parse-model`, `make nemotron-parse-smoke` | Build the vLLM 0.26 base image / the same for Nemotron 3.5 Parse (2.5 GB of weights under the OpenMDW license, served by vLLM with NVIDIA's plugin inside the container; positioned layout output; single- and double-pass presets), see [engines/nemotron-parse](engines/nemotron-parse/README.md) |
| `make docker-unlimited-ocr`, `make unlimited-ocr-model`, `make unlimited-ocr-smoke` | The same for Unlimited-OCR 1.0 (6.7 GB of MIT-licensed weights, served by vLLM's built-in model inside the container on the vLLM 0.26 base; positioned layout output), see [engines/unlimited-ocr](engines/unlimited-ocr/README.md) |
| `make docker-cu130-vllm030`, `make docker-lightonocr3`, `make lightonocr3-model`, `make lightonocr3-smoke` | Build the vLLM 0.30 base / the same for LightOnOCR-3 (0.8B, 1B and 4B; 12.8 GB of Apache-2.0 weights, or one checkpoint with `CHECKPOINT=`, served by vLLM's built-in models inside the container; six presets, plain Markdown and positioned grounding output per checkpoint), see [engines/lightonocr3](engines/lightonocr3/README.md) |
| `make docker-infinityparser2`, `make infinityparser2-model`, `make infinityparser2-smoke` | The same for Infinity-Parser2-Flash (4.4 GB of weights), see [engines/infinityparser2](engines/infinityparser2/README.md) |
| `make docker-mistral-ocr`, `make mistral-ocr-smoke` | Build the Mistral OCR 4.1 hosted-API engine image / opt-in, paid: run both presets on synthetic samples against the real API (needs `MISTRAL_API_KEY`; runs need `--allow-network`), see [engines/mistral-ocr](engines/mistral-ocr/README.md) |
| `make docker-openrouter`, `make openrouter-smoke` | Build the OpenRouter hosted vision-model engine image / opt-in, paid: run its preset (Gemini 3.8 Flash) on synthetic samples against the real API (needs `OPENROUTER_API_KEY`; runs need `--allow-network`), see [engines/openrouter](engines/openrouter/README.md) |
| `make docker-pero`, `make pero-bundles`, `make pero-smoke` | Build the PERO OCR engine image / copy the user-supplied PERO model bundles from `PERO_BUNDLES` (default `dev_data/pero_ocr_models`) into `OCRBENCH_CACHE`, checked against `bundles.sha256` / opt-in, GPU: run it through `run`, `evaluate` and `report` on synthetic samples, see [engines/pero](engines/pero/README.md) |
| `make docker-smoke`, `make gpu-preflight`, `make gpu-smoke` | Opt-in Docker and GPU checks (including the fake-engine runner tests), see [docs/DOCKER.md](docs/DOCKER.md) |

`make help` lists all targets. Unit tests run with network sockets disabled (`pytest-socket`). Tests that need Docker or a GPU are marked `docker` / `gpu` and run only with `--run-docker` / `--run-gpu`. Tests that send paid requests to a hosted OCR API are marked `api` and run only with `--run-api`. Add dependencies with `uv add` (or `uv add --dev`) so that `uv.lock` is updated in the same PR. Fixture rules are in [tests/fixtures/README.md](tests/fixtures/README.md).

## Data and runtime hygiene

Respect [the engine/evaluator boundary](docs/ARCHITECTURE.md): images may be provided to OCR engines; PAGE XML ground truth must remain isolated from inference. Commit only expressly redistributable synthetic fixtures, never real scans/GT, secrets, model weights, caches or run outputs. Pin versions and capture relevant inference configuration and failures for reproducibility.

For Docker, GPU or model-dependent changes, do not silently install huge models or start costly work. Record which configuration and resources were actually tested.

## PR handoff

Use [the PR template](.github/PULL_REQUEST_TEMPLATE.md) to link the issue and list verification, risks and remaining work. A documentation-only PR may require only manual document/link checks; it must not invent a passing test command.
