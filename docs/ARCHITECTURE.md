# Benchmark architecture

This document defines the system boundaries and high-level contracts of the benchmark. Durable decisions are recorded in [ADRs](adr/README.md); metric families are described in [EVALUATION.md](EVALUATION.md). Detailed formats and algorithms are owned by the implementation issues linked below. The [README](../README.md) states what is currently implemented.

## Data and execution flow

Everything runs **locally on one Linux machine**. The OCR inference component runs in an isolated Docker container without network access. The exception is a hosted-API engine, which may send the page image, and nothing else, to its provider when the run explicitly allows it ([ADR 0013](adr/0013-opt-in-network-for-hosted-api-engines.md)). The surrounding local application manages images and PAGE XML reference data, runs OCR, evaluates predictions, and produces reports. In the diagram, only the image and permitted inference metadata cross into the OCR container.

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

**Ground truth never enters an OCR container.** The local dataset reader, evaluator and report generator can access PAGE XML and its derived reference text/geometry as needed. The distinction is an **inference-container boundary**, not separate hosts, servers or permissions between local evaluation and reporting.

## Responsibilities and interface owners

| Component | Responsibility | Does not | Owner |
| --- | --- | --- | --- |
| Dataset manifest and PAGE XML reader | Pair images with GT; validate files, image sizes and geometry; extract reference text, lines, regions and reading order ([DATASETS.md](DATASETS.md), [ADR 0005](adr/0005-page-xml-reference-extraction.md)) | Expose GT to OCR inference containers | [#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5) |
| Engine adapter (container) | Every preprocessing step: resizing, region/line detection, tiling and stitching, plus OCR/HTR and prompting. Emits native output, UTF-8 text and standardized JSON | Read GT; invent geometry | [#11–#17](https://github.com/michal-hradis/ocr_benchmarking/issues/1) |
| Output contract | Versioned JSON for page text, flat optional regions with optional lines and positions; validator; shared host-side conversion of plain-text, Markdown and HTML output ([PREDICTIONS.md](PREDICTIONS.md), [ADR 0006](adr/0006-standardized-prediction-format.md)) | Prescribe how engines segment | [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6) |
| Runner and CLI | Launch isolated containers with the image and allowed metadata; record status, provenance and resources; persist outputs ([RUNNER.md](RUNNER.md), [ADR 0007](adr/0007-runner-and-engine-protocol.md)) | Tile, crop or resize images; expose GT to an engine container | [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7) |
| Evaluator | Compute ordered and flexible metrics against GT from stored outputs, using a versioned evaluation config ([EVALUATION.md](EVALUATION.md), [ADRs 0008](adr/0008-strict-text-metrics.md), [0009](adr/0009-flexible-line-alignment.md)) | Run engines | [#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8), [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9) |
| Reporter | Per-sample and aggregate results, GT-versus-prediction inspection, diagnostics, enumerated failures and truncations, provenance ([REPORTS.md](REPORTS.md), [ADR 0010](adr/0010-reports.md)) | Recompute or alter scores | [#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10) |
| Development tooling and base images | Linux Python environment; shared CPU/CUDA base image families | Add CI | [#4](https://github.com/michal-hradis/ocr_benchmarking/issues/4) |

## Contracts

### Benchmark samples ([ADR 0001](adr/0001-image-only-samples-engine-owned-preprocessing.md))

- One sample is one raster image. By default this is a **full page**. An independently prepared **paragraph crop** is also a valid sample type, with its own GT.
- Samples never use PDF input or a PDF text layer.
- The framework neither tiles nor crops images. Engines own all preprocessing, including tiling and stitching for large newspaper pages with 2,000+ lines.

### Ground-truth isolation ([ADR 0002](adr/0002-ground-truth-isolation.md))

PAGE XML and derived reference text, region/line geometry and reading order are available to the **local dataset reader, evaluator and reports**. An OCR inference container receives only the image and **allowed inference metadata** (for example language hints and sample type); it never receives GT paths, reference content or statistics derived from GT. The allowed metadata are the sample ID and unit, plus `language`, `doc_type` and `year` when the engine configuration enables them ([RUNNER.md](RUNNER.md#engine-protocol)).

### Standardized prediction ([ADR 0003](adr/0003-engine-run-output-contract-boundaries.md))

For an invocation that **produces a valid OCR prediction**, the engine adapter provides:

1. Required **UTF-8 page transcription** in plain text, without Markdown or HTML markup.
2. A **versioned standardized JSON** prediction. If the engine genuinely localizes text, it contains **typed regions** with positions and, where the engine reports them, nested **lines** with text, positions and optional type. Positions refer to pixels of the **original input image**.
3. The **native outputs** that were produced, preserved unchanged (e.g. TXT, Markdown, HTML, hOCR or PAGE XML).

The format is specified in [PREDICTIONS.md](PREDICTIONS.md) ([ADR 0006](adr/0006-standardized-prediction-format.md)): regions are a flat list, lines and positions are optional, and page text must agree with the line/region text. For engines whose native output is plain text, Markdown or HTML, items 1 and 2 are produced on the host by shared, versioned conversion code from the stored native output, after inference.

Image-to-text models without reliable segmentation are first-class participants: keep their page text without inventing regions, physical lines or coordinates. **When inference fails, a separate per-sample execution record with status, reason and provenance is required even if no valid OCR text/JSON result exists.** The evaluator, not the engine, may interpret that failure as an empty prediction for scoring.

### Execution and reproducibility

- A common Python orchestrator runs on **Linux** and starts one isolated Docker environment per engine. Engines may share a pinned CPU or CUDA base image family **only when their dependencies are compatible**.
- Both CPU and NVIDIA GPU execution are supported. A single RTX 4090 is a primary target.
- Engine code, image digests and model revisions are pinned. Base image families and engine image conventions are in [DOCKER.md](DOCKER.md) ([ADR 0004](adr/0004-docker-base-image-families.md)).
- Each run records a **frozen run configuration**: engine and model versions, prompts and language hints, preprocessing settings, hardware, token and output limits, and timeouts.

### Versioning, status and re-evaluation

- **Separate versions.** The output schema, the run configuration and the evaluation configuration (normalization, tokenization, alignment and failure policy) are versioned independently. Every score is identifiable by sample, run and evaluation version.
- **Raw and normalized output.** The run store keeps native output unchanged, next to the standardized text and JSON derived from it. Evaluator-side normalization is applied at scoring time and is never written back over predictions.
- **Execution records and prediction results are separate:** the runner records a status for every sample (`success`, `partial`, `truncated` or `failure`) and a reason for failures, such as a crash, timeout or invalid output. A failure does **not** require fabricating a valid OCR text/JSON result. Output that fails schema validation is never treated as `success`. Status values and failure reasons are defined in [RUNNER.md](RUNNER.md#statuses-and-failures).
- **Re-evaluation without rerunning models.** Local evaluation reads stored execution records, available predictions, and PAGE XML ground truth. Changing the evaluation config produces new scores under a new version, without new inference.
- **Failure accounting.** The primary default is **empty-output**: a failed OCR invocation is scored as an empty prediction, retaining its missing GT text in the score. Explicit configuration may instead **exclude** failures from an alternate metric calculation, always with coverage reported. See [EVALUATION.md](EVALUATION.md#failed-partial-and-truncated-samples).

## Data

- The currently available development set contains a handful of full-page images with PAGE XML GT (seven at present), held outside this repository and referenced by [`datasets/dev-01.json`](../datasets/dev-01.json). The architecture and ordinary automated tests do **not** assume its size, page-only content, or access to these files.
- Automated tests use small, redistributable synthetic fixtures.
- Selecting, acquiring, reviewing licenses for or splitting additional datasets is **not part of the current backlog**. A manifest may carry split labels as metadata, but they are not required.

## Engine integration

Every engine adapter requires the common contracts in #4–#7. Evaluation and reporting (#8–#10) are needed for complete benchmarking. **Tesseract ([#11](https://github.com/michal-hradis/ocr_benchmarking/issues/11)) is a reference adapter, not a prerequisite** for any other engine.

## Open implementation choices

The issues below own these choices. The architecture deliberately leaves them undecided. Normalization, tokenization, edit counts and empty cases were decided in [#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8) ([EVALUATION.md](EVALUATION.md#strict-metrics-metrics-version-1)). Flexible matching, split/merge accounting, candidate retrieval and unmatched predictions were decided in [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9) ([EVALUATION.md](EVALUATION.md#flexible-cer-and-wer-metrics-version-2)). Report layout, formats and the failure and coverage displays were decided in [#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10) ([REPORTS.md](REPORTS.md)). No open choice is currently assigned.

Development and testing rules are in [CONTRIBUTING.md](../CONTRIBUTING.md).
