# Architecture overview (provisional)

**Status:** this describes agreed boundaries and a proposed flow. The actual implementation architecture, CLI, runtime packaging, schema shapes and evaluator design belong to [issue #3](https://github.com/michal-hradis/ocr_benchmarking/issues/3) and subsequent implementation issues. As of this documentation milestone, only the README and workflow documentation exist; there is no functioning runner or model adapter.

## Purpose and boundaries

The benchmark compares transcription quality of OCR/HTR pipelines and page-image VLMs on raster page images of books, magazines and newspapers. Independently prepared paragraph crops are also a supported sample type. Evaluation ground truth is PAGE XML.

```text
Dataset manifest
    | image path + image metadata
    v
Common host-side orchestration  ---> engine image/container
    |                               (preprocessing, tiling, OCR/HTR)
    |                                      |
    |                               text + normalized JSON
    |                               + native engine output
    v                                      |
Host-side evaluation <---------------------+
    ^
    | PAGE XML GT (never mounted/passed to engine)
    v
Metrics + reproducible reports
```

## Responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| Dataset loader ([#5](https://github.com/michal-hradis/ocr_benchmarking/issues/5)) | Image/GT pairing, metadata, PAGE XML parsing/validation | Engine inference and benchmark-scoring decisions |
| Runner ([#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7)) | Iteration, launching isolated Docker engine containers, version/config/failure records | OCR model internals, reference-text access |
| Each OCR engine ([#11+](https://github.com/michal-hradis/ocr_benchmarking/issues/11)) | OCR/HTR inference, resizing, optional tiling/detection/stitching, adapter to result contract | Scoring, GT access, external evaluator tiling |
| Result contract ([#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6)) | UTF-8 text, versioned JSON, optional typed regions + nested lines and positions, native output preservation | Invented locations for text-only systems |
| Evaluator ([#8](https://github.com/michal-hradis/ocr_benchmarking/issues/8), [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9)) | PAGE XML reference, CER/WER, scalable flexible alignment, lexical overlap, omission/hallucination accounting | Inference or changing source images |
| Reporting ([#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10)) | Per-image/aggregate metrics, versions, failures, inspectable artifacts | Rerunning/altering model predictions |

## Interface principles — agreed, detailed contracts pending

- **Image-only input:** complete raster pages by default; paragraph crops as explicit benchmark samples. No PDF text extraction.
- **One output per sample:** required UTF-8 transcription; normalized JSON with typed regions and original-image positions, nested lines with text/positions and optional type; geometry absent when genuinely unavailable.
- **Native output preservation:** preserve generated TXT, Markdown, HTML, PAGE/hOCR or JSON without treating Markdown syntax as transcription.
- **Text evaluation:** strict baseline CER/WER, flexible order-independent matching, bag-of-words precision/recall/F1 and ROUGE-2; separate configuration/versioning for normalizations. Routine alignment must scale beyond 2,000 lines per newspaper image.
- **No ground-truth leakage:** do not pass/mount PAGE XML into engine containers.
- **Runtime:** Linux Python orchestration, isolated Docker engines. Share pinned CPU/CUDA base layers only where compatible; document NVIDIA dependencies. Run metadata includes weights/code/runtime versions, prompts, preprocessing, language, token limits, resources, truncation and failures.
- **Testing:** deterministic local tests with synthetic fixtures; no CI initially, no forced full suite. Real GPU/model inference explicitly opted into.

The specific JSON property names, bbox convention, line concatenation and alignment algorithm are **not specified by this document**; they must be agreed and tested in [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3), [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6) and [#9](https://github.com/michal-hradis/ocr_benchmarking/issues/9).

## Decisions and backlog

Use the [ADR index](adr/README.md) for durable decisions. Keep the [issue index](TODO.md) linked to the GitHub roadmap, which is the source of truth for planning/status. Do not label future functionality as implemented before its corresponding PR is merged.
