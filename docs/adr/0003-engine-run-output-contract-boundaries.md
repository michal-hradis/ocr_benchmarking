# 0003: Engine, run and output contract boundaries

- Status: accepted
- Date: 2026-10-20
- Related issue: [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3)

## Context

The engines range from line-based OCR with exact geometry to image-to-text models that emit unlocalized page text. Results must be comparable, reproducible and re-scorable without repeating costly GPU inference.

## Decision

- **Output.** Every engine emits a required UTF-8 plain-text page transcription and a versioned standardized JSON, and its native outputs are preserved unchanged. When the engine genuinely localizes text, the JSON holds typed regions with positions that contain lines with text, a position and an optional type. Positions refer to original-image pixels. Text-only engines provide no regions, lines or coordinates, rather than fabricated ones. Exact fields are defined in [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6).
- **Run.** A Linux Python orchestrator runs each engine in its own Docker image. Shared CPU/CUDA base image families are used only when dependencies are compatible. Engine, image digest and model revisions are pinned. Each run stores a frozen configuration, per-sample status (success, partial, truncated or failure with a reason), timings, resources and limits. Details are defined in [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7).
- **Evaluation.** Scoring reads stored outputs only, under an independently versioned evaluation configuration. Re-evaluation never requires rerunning a model.

## Alternatives considered

- *Requiring line geometry from all engines.* Rejected: it would exclude, or force fake geometry onto, image-to-text models.
- *Scoring inside the run.* Rejected: any change to scoring would require new inference.

## Consequences and verification

- The schema and validator in #6 must accept text-only results and reject invalid output. Invalid output is never recorded as success.
- Output schema, run configuration and evaluation configuration change independently and are versioned separately.
