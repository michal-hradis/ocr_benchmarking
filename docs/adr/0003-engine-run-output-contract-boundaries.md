# 0003: Engine, run and output contract boundaries

- Status: accepted
- Date: 2026-10-20
- Related issue: [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3)

## Context

The engines range from line-based OCR with exact geometry to image-to-text models that emit unlocalized page text. Results must be comparable, reproducible and re-scorable without repeating costly GPU inference.

## Decision

- **Output.** A valid OCR prediction has required UTF-8 plain-text transcription and versioned standardized JSON, with native outputs preserved where generated. If the engine genuinely localizes text, JSON holds typed positioned regions and nested positioned lines with text and optional type in original-image pixel coordinates; text-only models are valid without invented geometry. **A failed inference need not produce valid prediction text/JSON; it must produce a separate execution failure record through the runner.** Exact fields are owned by [#6](https://github.com/michal-hradis/ocr_benchmarking/issues/6).
- **Run.** A Linux Python orchestrator runs each engine in its own Docker image on the local machine. Shared CPU/CUDA bases are used only when dependency versions are compatible. Engine, image digest and model revisions are pinned. A run preserves frozen configuration and per-sample **execution records** (status, failure reason, timings, resources, limits and provenance) independently of any valid OCR prediction. Details belong to [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7).
- **Evaluation.** Local scoring reads stored execution records, valid predictions (when available) **and PAGE XML ground truth**; reports may inspect both GT and predictions. Re-evaluation never reruns a model. A failed execution is scored as empty prediction **by default**; a clearly labelled, explicitly configured exclude-failures score is optional and must report coverage.

## Alternatives considered

- *Requiring line geometry from all engines.* Rejected: it would exclude, or force fake geometry onto, image-to-text models.
- *Scoring inside the run.* Rejected: any change to scoring would require new inference.

## Consequences and verification

- The schema and validator in #6 must accept text-only predictions and reject invalid prediction output. Invalid predictions are never recorded as successful OCR; #7 retains failure records without fabricating schema-valid text.
- Output schema, run configuration and evaluation configuration change independently and are versioned separately.
