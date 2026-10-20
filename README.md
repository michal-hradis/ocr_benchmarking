# OCR Benchmarking

An image-only, reproducible benchmark for OCR and handwritten text recognition (HTR) on **books, magazines and newspapers**, including difficult historical scans.

**Project status:** planning/documentation stage. There is not yet an installable Python package, working benchmark CLI, Docker engine image, evaluation implementation, or published dataset in this repository. These are tracked as implementation issues. Do not treat planned commands or interfaces as implemented features.

## Intended benchmark

- **Inputs:** raster page images (e.g. PNG, TIFF, JPEG); whole pages are the default. A dataset may instead contain independently prepared paragraph crops. PDF text layers are not used.
- **Ground truth:** PAGE XML, paired with each image by a dataset manifest.
- **Engines:** modular OCR/HTR pipelines and end-to-end image-to-text models, each isolated in its own Linux Docker environment. Engines own their preprocessing, resizing, internal tiling and reconstruction; the evaluation framework does **not** tile input pages.
- **Output:** required UTF-8 page transcription plus a versioned JSON representation. The planned representation supports typed regions with coordinates in the original image, and nested lines with text, positions and optional type. Systems unable to localize content retain their text without invented boxes. Raw/native OCR outputs are preserved.
- **Evaluation:** initially CER/WER with flexible, order-independent text alignment, conventional ordered diagnostics, bag-of-words precision/recall/F1, and ROUGE-2. Newspaper images can contain over 2,000 text lines; large-page matching must be scalable. Exact metrics and schema semantics will be specified and implemented in separate issues.
- **Environment:** Linux; NVIDIA RTX 4090 (24 GB) is a primary GPU target, while offline deterministic developer tests must not depend on a GPU. Model-specific Docker environments should reuse common, pinned CUDA/CPU base layers **when dependencies are compatible**.
- **Reproducibility:** record model/runtime versions, configuration, languages, prompts, preprocessing, runtime and resources, truncation and failures. Keep real data, weights and generated runs outside Git.

**Ground-truth isolation is essential:** inference engines must never receive PAGE XML ground truth or evaluation-only annotations.

## Current state and next work

This repository currently has a planning backlog, not runnable benchmarking software. The implementation sequence is tracked by [roadmap issue #1](https://github.com/michal-hradis/ocr_benchmarking/issues/1):

1. [#2: documentation and agent workflow](https://github.com/michal-hradis/ocr_benchmarking/issues/2).
2. [#3: architecture and benchmark contracts](https://github.com/michal-hradis/ocr_benchmarking/issues/3).
3. [#4: Linux development tooling and reusable Docker base images](https://github.com/michal-hradis/ocr_benchmarking/issues/4).
4. [#5–#10: datasets, result schema, runner, metrics, alignment and reporting](https://github.com/michal-hradis/ocr_benchmarking/issues).
5. [#11–#17: initial OCR systems](https://github.com/michal-hradis/ocr_benchmarking/issues), beginning with Tesseract, followed by PaddleOCR, Kraken and image-to-text models.

There is **no CI at this stage**. Developers and coding agents run checks relevant to their PR changes and record actual results. Running the entire future test suite is not a universal requirement.

## Documentation

| Read | Purpose |
| --- | --- |
| [Start here](docs/START_HERE.md) | Status, where to begin, and what does not yet exist |
| [Contributing](CONTRIBUTING.md) | Branch/PR workflow, selective testing and safety rules |
| [Agent instructions](AGENTS.md) | Rules for coding agents working in this repository |
| [Issue workflow](docs/ISSUE_WORKFLOW.md) | From draft to implementation-ready issues |
| [Architecture overview](docs/ARCHITECTURE.md) | Current boundaries vs. planned components |
| [ADR index](docs/adr/README.md) | How architectural decisions will be recorded |
| [Issue index](docs/TODO.md) | Ordered work and links to GitHub issues |

Development commands, setup instructions and GPU Docker steps will be documented *when implemented* under [#4](https://github.com/michal-hradis/ocr_benchmarking/issues/4).

## Attribution

The agent-friendly issue, PR and documentation workflow takes inspiration from [DCGM/semant-demo](https://github.com/DCGM/semant-demo), notably its [AGENTS.md](https://github.com/DCGM/semant-demo/blob/main/AGENTS.md) and [issue workflow](https://github.com/DCGM/semant-demo/blob/main/docs/ISSUE_WORKFLOW.md). This project intentionally does **not** adopt semant-demo's application architecture, CI requirements, or blanket full-suite verification policy.
