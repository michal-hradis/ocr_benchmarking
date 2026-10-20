# Start here

This is an early-stage benchmark for comparing OCR/HTR on raster images from books, magazines and newspapers. **It is currently a documentation-only repository.** There is no executable benchmark or test harness yet; do not assume a Python package, `ocrbench` command, `make setup`, or Docker images already exist.

## Which document should I read?

| Task | Start with |
| --- | --- |
| Understand the goal, inputs and metrics | [README](../README.md), then [architecture overview](ARCHITECTURE.md) |
| Implement an issue | [Contributing](../CONTRIBUTING.md); agents also read [AGENTS.md](../AGENTS.md) |
| Refine an issue | [Issue workflow](ISSUE_WORKFLOW.md), [roadmap](https://github.com/michal-hradis/ocr_benchmarking/issues/1) |
| Find planned work | [Issue index](TODO.md) |
| Make or revisit a binding design decision | [ADR index](adr/README.md), and the corresponding GitHub issue |
| Work on development tooling | [#4](https://github.com/michal-hradis/ocr_benchmarking/issues/4) |
| Implement an OCR engine | Shared schema/runner issues [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6), [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7), then a model-specific issue |

## Architecture at a glance

Planned flow:

```text
raster page or paragraph-crop image
  -> common Python runner + image-only engine input
  -> engine-specific Docker environment (optional internal tiling)
  -> native output + normalized UTF-8 / JSON result
  -> host-side PAGE XML reference + evaluation metrics
  -> reproducible per-image and aggregate reports
```

The PAGE XML reference must **never** be available to engine inference. Region and line locations refer to coordinates in the **original image** when available; text-only systems are supported without invented geometry.

## What can be worked on now?

The first steps are [#2 documentation](https://github.com/michal-hradis/ocr_benchmarking/issues/2), [#3 architecture/contracts](https://github.com/michal-hradis/ocr_benchmarking/issues/3) and [#4 Linux development environment](https://github.com/michal-hradis/ocr_benchmarking/issues/4). After shared contracts, the first executable end-to-end path is Tesseract [#11](https://github.com/michal-hradis/ocr_benchmarking/issues/11); pipeline and page-VLM engines follow.

**Verification policy:** no CI at the beginning. Contributors run the tests relevant to their PR and list actual results; there is no standing requirement to run the whole test suite. See [CONTRIBUTING.md](../CONTRIBUTING.md#test-selection-policy--local-proportional-explicit).
