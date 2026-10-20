# OCR Benchmarking

Compare OCR and handwritten text recognition (HTR) systems on **raster images of books, magazines and newspapers**, particularly complex and historical prints.

**Project status:** documentation and initial implementation planning. The Python runner, evaluation code, Docker images and CLI commands described below are *not yet implemented*.

## Use cases

- **Newspapers:** full-page, multi-column, densely printed or degraded scans, potentially with 2,000+ text lines.
- **Books:** printed pages and historic typography, including text-dense scans.
- **Magazines:** pages with mixed text, headings, captions and complex layout.
- **Paragraph crops:** pre-prepared image crops, as an alternative dataset unit to the default **complete page**.

Input is always an **image**, never a PDF text layer. Ground truth uses **PAGE XML**. The benchmark will compare conventional OCR pipelines with page-image-to-text models; engine-specific resizing, segmentation and tiling belong to the engine. See [architecture](docs/ARCHITECTURE.md) for the separation between engines, reference data and evaluation.

For initial development, there is a **small, separately held set of six images with PAGE XML ground truth**. It is not committed to this repository. Selecting or collecting additional evaluation datasets is **separate work, not part of the present backlog**.

## Planned outputs and scoring

Engines provide UTF-8 transcription plus versioned JSON with optional typed, positioned regions and nested text lines; native OCR outputs are retained. Initial metrics will include CER/WER (including flexible line/chunk alignment), bag-of-words precision/recall/F1 and ROUGE-2. Details are to be finalized in [contract issues #3–#9](https://github.com/michal-hradis/ocr_benchmarking/issues/1).

## Quick start (planned)

The following is a **proposed workflow, not runnable commands**; the CLI and flags may change during [runner implementation #7](https://github.com/michal-hradis/ocr_benchmarking/issues/7):

```sh
ocrbench dataset validate <dataset-manifest>
ocrbench run --engine tesseract --dataset <dataset-manifest>
ocrbench evaluate --run <run-id>
ocrbench report --run <run-id>
```

Linux is the target platform, with per-engine Docker environments (shared CUDA/CPU bases where compatible) and an RTX 4090 as a primary GPU test device. Development setup will be introduced in [#4](https://github.com/michal-hradis/ocr_benchmarking/issues/4). Consult [CONTRIBUTING.md](CONTRIBUTING.md) for the testing and PR policy.

## Documentation and work

- [Start here](docs/START_HERE.md) — short guide to repository documents and next tasks.
- [Contributing](CONTRIBUTING.md) — development, selective verification and PR rules.
- [Agent instructions](AGENTS.md) — additional guidance for coding agents.
- [Issue workflow](docs/ISSUE_WORKFLOW.md) — define and refine implementation-ready work.
- [Architecture](docs/ARCHITECTURE.md) and [ADR index](docs/adr/README.md) — component boundaries and recorded decisions.
- [Work index](docs/TODO.md) and [roadmap #1](https://github.com/michal-hradis/ocr_benchmarking/issues/1) — issue dependencies and priorities.

The development workflow is inspired by [DCGM/semant-demo](https://github.com/DCGM/semant-demo), especially its [AGENTS.md](https://github.com/DCGM/semant-demo/blob/main/AGENTS.md) and [issue workflow](https://github.com/DCGM/semant-demo/blob/main/docs/ISSUE_WORKFLOW.md); its application architecture and CI workflow are not copied.
