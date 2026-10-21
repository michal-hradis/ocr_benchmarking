# 0010: Reports from stored evaluations

- Status: proposed
- Date: 2026-10-21
- Related issue: [#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10) (maintainer decisions recorded in the implementing PR)

## Context

Runs (#7) and their scores (#8, #9) are stored per run. Engines must be compared without rerunning OCR or rescoring, and without silently mixing scores that were computed differently: another failure policy, another metrics version, other GT, or scores computed before a failed sample was retried. Failures must stay visible under every failure policy. OCR output is untrusted text, and GT may be read by reports but never by OCR containers.

## Decision

- **Input.** `ocrbench report <run-or-evaluation>... -o <new dir>` reads one `scores.json` per run, by default the newest evaluation. The report copies the stored per-sample and aggregate scores and never recomputes them.
- **Staleness and comparability are refused, not labelled.** An evaluation is refused if:
  - `run.json` changed after it;
  - a sample has a newer attempt or a different status;
  - a scored prediction's SHA-256 changed;
  - the GT differs from the GT frozen with the run;
  - the scores format is old.

  Runs in one report must share the metrics version and the evaluation configuration hash (including the failure policy). Each sample ID must have the same frozen image and GT hashes, unit and document type in every run, and inputs that were not pinned are refused when runs are compared. The output directory must be new.
- **Outputs.** The report writes:
  - `report.json` with provenance (engine, model, image, run and evaluation versions and hashes);
  - `aggregates.csv` with one row per run and group, with micro and macro values, denominators, coverage and status counts;
  - `samples.csv` with one row per run and sample;
  - `index.html`.

  The groups are those of the evaluation (all, unit, document type within a unit); each run is one engine/config.
- **Failures.** Every non-`success` sample is listed with its status, reason, the engine's message and a link to its execution record, together with per-status counts, under both failure policies. `exclude` reports carry a warning that they are not the primary score.
- **No split grouping.** Dataset splits are not frozen in `run.json` and not grouped; they can be added when a dataset uses them.
- **Inspection pages are optional** (`--inspect`). They show text differences only: the ordered character diff and the flexible groups from the stored `group_lines`, with unmatched lines. Image and position overlays are deferred.
- **Escaping.** All text in HTML is escaped, and the pages contain no scripts. Engine messages are not written to CSV, and CSV text cells that start with a formula character get a leading `'`.

## Alternatives considered

- *Per-run reports inside the run directory.* Simpler, but they do not compare engines, which is the purpose of the benchmark.
- *Labelling stale or differently configured evaluations instead of refusing them.* More permissive, but such scores are easy to compare by mistake; `ocrbench evaluate` is cheap and fixes it.
- *Freezing `split` in `run.json` and grouping by it.* No dataset has splits yet; deferred by the maintainer.
- *Reading the split from the current manifest at report time.* It could silently disagree with the run.
- *Image overlays of GT and prediction geometry.* Useful for layout inspection, but deferred to keep the first version simple.
- *Recomputing aggregates per engine in the reporter.* Not needed: each run already has its own aggregates.

## Consequences and verification

- Documented in [REPORTS.md](../REPORTS.md), implemented in `ocrbench.report` and `ocrbench report`. `ocrbench evaluate` also logs every non-`success` sample on stderr.
- Tests cover:
  - provenance and copied aggregates of two runs, and golden CSV values;
  - listing of failed and truncated samples with messages under both failure policies, and a run without any success;
  - optional inspection pages and their differences, and flexible groups from stored ranges;
  - escaping of markup in predictions, GT and engine messages;
  - refusal of newer attempts, changed predictions, changed GT, changed `run.json`, old scores, missing evaluations, different configurations, different samples, different images or unit/document type for the same sample, unpinned inputs, duplicate runs and existing output directories;
  - group texts that follow the stored input-line ranges with blank GT and prediction lines;
  - CSV cells that could become spreadsheet formulas;
  - the CLI.
- A report format change increments `REPORT_VERSION`.
