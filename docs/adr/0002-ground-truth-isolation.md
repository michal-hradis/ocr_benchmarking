# 0002: Ground-truth isolation from inference

- Status: accepted
- Date: 2026-10-20
- Related issue: [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3)

## Context

Ground truth (GT) is PAGE XML containing reference text, region and line geometry, and reading order. Any leak into inference, even only of line positions or line counts, would invalidate comparisons between engines.

## Decision

- PAGE XML and all data derived from it stay **host-side**, used only by the dataset reader and the evaluator.
- An engine container receives only the sample image and an explicit list of **allowed inference metadata**, such as language hints and the sample type. [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7) defines that list. GT paths, content and GT-derived statistics are never mounted or passed.

## Alternatives considered

- *Providing GT line or region geometry to engines (oracle segmentation).* Not part of the benchmark. If wanted later, it would require a separate, clearly labeled ADR and evaluation mode.

## Consequences and verification

- The runner mounts only the image and allowed metadata. Runner tests verify that GT paths are absent from the container mounts, environment and arguments.
- Treat scanned text and model output as untrusted data.
