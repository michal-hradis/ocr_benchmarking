# Architecture decision records (ADRs)

Create short ADRs for agreed cross-engine decisions, including output schema, coordinate conventions, scoring normalization/alignment, and Docker engine interfaces. [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3) will establish the first formal decisions.

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
