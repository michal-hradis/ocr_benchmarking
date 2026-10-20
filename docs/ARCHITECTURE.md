# Benchmark architecture (provisional)

This document records agreed system boundaries. [Issue #3](https://github.com/michal-hradis/ocr_benchmarking/issues/3) will expand and finalize the architecture; [#5–#10](https://github.com/michal-hradis/ocr_benchmarking/issues/1) implement the contracts.

## Data and execution flow

```text
raster image + manifest metadata
             |
             v
       host-side runner ---- image only ---> engine Docker container
             |                              (resizing / tiling / OCR)
             |                                         |
             |                           native output + UTF-8 / JSON
             |                                         |
             +-----------------------------------------+
             |
             v
     host-side evaluator <--------- PAGE XML ground truth
             |
             v
       metrics / reports
```

**GT isolation:** PAGE XML, reference text and reference positions remain host-side and must never be exposed to an engine container. Only allowed image inputs and metadata cross that boundary.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| Dataset manifest / PAGE reader ([#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5)) | Pair raster images and PAGE XML GT; validate image and annotation metadata |
| Runner ([#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7)) | Launch isolated engine containers, retain outputs, capture status and provenance |
| Engine adapter ([#11–#17](https://github.com/michal-hradis/ocr_benchmarking/issues/1)) | Image preprocessing, any region detection/tiling/stitching, OCR/HTR and native output |
| Output contract ([#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6)) | Required UTF-8 transcription; versioned JSON regions with optional original-image coordinates, types, nested text lines and optional line types; native outputs preserved |
| Evaluator ([#8–#9](https://github.com/michal-hradis/ocr_benchmarking/issues/1)) | Parse GT, compute ordered and aligned CER/WER, bag-of-words metrics and ROUGE-2; count omissions/extra text |
| Reporter ([#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10)) | Per-image and aggregate metrics, diagnostics, failures and configuration provenance |

## Stable boundaries and open details

- **Benchmark unit:** a full-page raster image by default; separately supplied paragraph crops are also valid. No PDF extraction and no evaluator-side tiling.
- **Geometry:** regions/lines must use coordinates in the original input image when genuinely available. Text-only predictions are valid without invented coordinates.
- **Reproducibility:** Linux Python orchestration, engine-isolated Docker images, compatible shared CPU/CUDA bases and pinned model/runtime revisions. Record language hints, prompts, preprocessing, resources, truncation and failures.
- **Scoring:** initial CER/WER (including scalable flexible alignment for 2,000+ lines per newspaper page), lexical overlap and ordered diagnostics. Exact normalization, split/merge rules, bounding-box convention and JSON property details remain for [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3), [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6) and [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9).

Use [ADRs](adr/README.md) for agreed durable decisions. Individual OCR engines need the common image/dataset, runner and output contracts; Tesseract is an example adapter, **not** a prerequisite for other engines.
