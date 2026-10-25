# Architecture decision records (ADRs)

Create short ADRs for agreed cross-engine decisions, including output schema, coordinate conventions, scoring normalization/alignment, and Docker engine interfaces.

## Records

| ADR | Status |
| --- | --- |
| [0001: Image-only samples and engine-owned preprocessing](0001-image-only-samples-engine-owned-preprocessing.md) | accepted |
| [0002: Ground-truth isolation from inference](0002-ground-truth-isolation.md) | accepted |
| [0003: Engine, run and output contract boundaries](0003-engine-run-output-contract-boundaries.md) | accepted |
| [0004: Docker base image families](0004-docker-base-image-families.md) | proposed |
| [0005: PAGE XML reference extraction and dataset manifests](0005-page-xml-reference-extraction.md) | proposed |
| [0006: Standardized prediction format and host-side conversion](0006-standardized-prediction-format.md) | proposed |
| [0007: Runner lifecycle and engine container protocol](0007-runner-and-engine-protocol.md) | proposed |
| [0008: Strict text metrics and failure accounting](0008-strict-text-metrics.md) | proposed |
| [0009: Flexible line alignment for CER/WER](0009-flexible-line-alignment.md) | proposed |
| [0010: Reports from stored evaluations](0010-reports.md) | proposed |
| [0011: Region markup converted on the host](0011-region-markup-converted-on-the-host.md) | accepted |
| [0012: Inference servers inside engine containers](0012-inference-servers-inside-engine-containers.md) | accepted |

## Conventions

- Name records `0001-short-decision.md`, `0002-next-decision.md`, etc.
- Include **status** (proposed/accepted/superseded), date, context, decision, alternatives, consequences, and links to the motivating issues/PRs.
- An ADR marked `proposed` is not binding. Mark accepted only after an explicit decision.
- Preserve history; replace a superseded decision with a new linked ADR rather than rewriting its rationale.

## Template

```markdown
# 0001: Decision title

- Status: proposed
- Date: YYYY-MM-DD
- Related issue: #N

## Context
## Decision
## Alternatives considered
## Consequences and verification
```
