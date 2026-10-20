# 0001: Image-only samples and engine-owned preprocessing

- Status: accepted
- Date: 2026-10-20
- Related issue: [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3)

## Context

The benchmark compares conventional OCR pipelines with page-image-to-text models on books, magazines and newspapers. Some newspaper pages have 2,000+ lines. Engines differ greatly in input resolution limits, segmentation and tiling strategy. If the framework preprocessed images, it would favor some engines and hide others' real behavior.

## Decision

- A benchmark sample is one raster image: a **full page** by default, or an independently prepared **paragraph crop** with its own ground truth.
- PDF input and PDF text layers are not used.
- All resizing, region detection, tiling and stitching happen **inside the engine adapter**. The runner and evaluator never tile, crop or resize sample images.

## Alternatives considered

- *Framework-side tiling or line cropping.* Rejected: it imposes one segmentation on all engines and measures the tiler rather than the engine.
- *PDF input.* Rejected: a text layer would bypass recognition entirely.

## Consequences and verification

- Adapters for resolution-limited models must implement their own tiling. That work is part of the engine's result and cost.
- Runner tests ([#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7)) should verify that the image passed to a container is byte-identical to the sample image.
