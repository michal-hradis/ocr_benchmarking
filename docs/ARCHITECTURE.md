# Benchmark architecture

This document defines the system boundaries and high-level contracts of the benchmark. Durable decisions are recorded in [ADRs](adr/README.md); metric families are described in [EVALUATION.md](EVALUATION.md). Detailed formats and algorithms are owned by the implementation issues linked below. The [README](../README.md) states what is currently implemented.

## Data and execution flow

There are two separate paths. The **inference path** carries only images to the engines. The **reference path** carries ground truth (GT) only to the evaluator.

```text
                      dataset manifest (sample id, image path, GT path, metadata)
                                              |
                       +----------------------+----------------------+
                       |                                             |
              INFERENCE PATH                                 REFERENCE PATH
           (image + allowed metadata)                     (host-side only)
                       |                                             |
                       v                                             v
                 host runner                               PAGE XML reader
                       |                                             |
               image only, no GT                          reference text, lines,
                       v                                   regions, positions
          +------------------------+                                 |
          | engine Docker container|                                 |
          |  resize / segment /    |                                 |
          |  tile / OCR / stitch   |                                 |
          +------------------------+                                 |
                       |                                             |
          native output + UTF-8 text + JSON                          |
                       v                                             |
           run store (frozen config, status,                         |
           provenance, raw + standardized output)                    |
                       |                                             |
                       +--------------------+------------------------+
                                            v
                                   host-side evaluator
                                 (versioned eval config)
                                            |
                                            v
                                   metrics -> reporter
```

Ground truth never enters the inference path. The evaluator is the first and only component that sees both predictions and GT.

## Responsibilities and interface owners

| Component | Responsibility | Does not | Owner |
| --- | --- | --- | --- |
| Dataset manifest and PAGE XML reader | Pair images with GT; validate files, image sizes and geometry; extract reference text, lines, regions and reading order | Pass GT to the runner or engines | [#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5) |
| Engine adapter (container) | Every preprocessing step: resizing, region/line detection, tiling and stitching, plus OCR/HTR and prompting. Emits native output, UTF-8 text and standardized JSON | Read GT; invent geometry | [#11–#17](https://github.com/michal-hradis/ocr_benchmarking/issues/1) |
| Output contract | Versioned JSON for page text and optional positioned regions and lines; validator | Prescribe how engines segment | [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6) |
| Runner and CLI | Launch isolated containers with the image and allowed metadata; record status, provenance and resources; persist outputs | Tile, crop or resize images; mount GT | [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7) |
| Evaluator | Compute ordered and flexible metrics against GT from stored outputs, using a versioned evaluation config | Run engines | [#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8), [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9) |
| Reporter | Per-sample and aggregate results, diagnostics, enumerated failures and truncations, provenance | Recompute or alter scores | [#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10) |
| Development tooling and base images | Linux Python environment; shared CPU/CUDA base image families | Add CI | [#4](https://github.com/michal-hradis/ocr_benchmarking/issues/4) |

## Contracts

### Benchmark samples ([ADR 0001](adr/0001-image-only-samples-engine-owned-preprocessing.md))

- One sample is one raster image. By default this is a **full page**. An independently prepared **paragraph crop** is also a valid sample type, with its own GT.
- Samples never use PDF input or a PDF text layer.
- The framework neither tiles nor crops images. Engines own all preprocessing, including tiling and stitching for large newspaper pages with 2,000+ lines.

### Ground-truth isolation ([ADR 0002](adr/0002-ground-truth-isolation.md))

PAGE XML and anything derived from it, such as reference text, line or region geometry and reading order, stays host-side. A container receives only the image and **allowed inference metadata**, for example language hints and the sample type. It never receives GT paths, content or statistics derived from GT. #7 defines the exact allowed-metadata list.

### Standardized prediction ([ADR 0003](adr/0003-engine-run-output-contract-boundaries.md))

Each engine produces, for every sample:

1. A required **UTF-8 page transcription** in plain text, with no Markdown or HTML markup.
2. A **versioned standardized JSON** result. When the engine genuinely localizes text, the JSON contains **typed regions** with positions, and the regions contain **lines** with text, a position and an optional type. Positions refer to pixels of the **original input image**.
3. The **native outputs**, preserved unchanged, for example TXT, Markdown, HTML, hOCR, PAGE XML and metadata.

Image-to-text models without reliable segmentation are first-class participants. They provide page text only and must not invent regions, physical lines or coordinates.

### Execution and reproducibility

- A common Python orchestrator runs on **Linux** and starts one isolated Docker environment per engine. Engines may share a pinned CPU or CUDA base image family **only when their dependencies are compatible**.
- Both CPU and NVIDIA GPU execution are supported. A single RTX 4090 is a primary target.
- Engine code, image digests and model revisions are pinned.
- Each run records a **frozen run configuration**: engine and model versions, prompts and language hints, preprocessing settings, hardware, token and output limits, and timeouts.

### Versioning, status and re-evaluation

- **Separate versions.** The output schema, the run configuration and the evaluation configuration (normalization, tokenization, alignment and failure policy) are versioned independently. Every score is identifiable by sample, run and evaluation version.
- **Raw and normalized output.** The run store keeps native output unchanged, next to the standardized text and JSON derived from it. Evaluator-side normalization is applied at scoring time and is never written back over predictions.
- **Explicit status per sample:** `success`, `partial`, `truncated` or `failure`, with failure reasons such as a timeout, a crash, empty output or invalid output. Output that fails schema validation is never treated as `success`. #7 finalizes the exact status values and reasons.
- **Re-evaluation without rerunning models.** The evaluator works only from stored outputs and GT. Changing the evaluation config produces new scores under a new evaluation version, without new inference.
- **Failure accounting.** [EVALUATION.md](EVALUATION.md#failed-partial-and-truncated-samples) describes how failed, partial and truncated samples are scored.

## Data

- The initial real development set is **six page images with PAGE XML GT**. It is held **outside this repository** and is referenced by local manifests only.
- Automated tests use small, redistributable synthetic fixtures.
- Selecting, acquiring, reviewing licenses for or splitting additional datasets is **not part of the current backlog**. A manifest may carry split labels as metadata, but they are not required.

## Engine integration

Every engine adapter requires the common contracts in #4–#7. Evaluation and reporting (#8–#10) are needed for complete benchmarking. **Tesseract ([#11](https://github.com/michal-hradis/ocr_benchmarking/issues/11)) is a reference adapter, not a prerequisite** for any other engine.

## Open implementation choices

The issues below own these choices. The architecture deliberately leaves them undecided.

| Choice | Owner |
| --- | --- |
| Manifest format and fields; precedence between PAGE XML line and region text; handling of missing text | [#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5) |
| JSON field names, nullability, bounding-box and polygon convention, encoding of unlocalized results, aggregation of page text versus line text | [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6) |
| Container invocation, I/O mount protocol, per-sample versus batch lifecycle, allowed-metadata list, final status vocabulary | [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7) |
| Unicode normalization, grapheme handling, word tokenization, edit-count and empty-case definitions | [#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8) |
| Flexible matching algorithm, split/merge accounting, candidate retrieval, treatment of unmatched predictions | [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9) |
| Report layout and formats; default failure-scoring variant shown first | [#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10) |

Development and testing rules are in [CONTRIBUTING.md](../CONTRIBUTING.md).
