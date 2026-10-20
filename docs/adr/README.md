# Architecture decision records (ADRs)

Use short, versioned Markdown ADRs to record **agreed** changes that multiple components/engines need to honor: e.g. dataset and GT isolation, JSON region/line semantics, coordinate convention, metric normalization/alignment, engine Docker protocol or reproducibility policy.

**No numbered ADRs exist yet.** [#3](https://github.com/michal-hradis/ocr_benchmarking/issues/3) begins formalizing architecture and may add the first accepted records. This index defines the convention, not decisions that have not been made.

## Convention

- Filename: `0001-short-decision-name.md`, increasing sequential number.
- Include title, **status** (proposed / accepted / superseded), date, context, decision, alternatives, consequences, and links to motivating issues/PRs.
- A proposed ADR is not binding. Mark accepted only after an explicit project decision; do not silently introduce new interfaces because they seem convenient for one model.
- If a decision changes, append a new ADR or mark the old one superseded and link the successor. Do not rewrite the reasoning history.
- Link relevant ADRs in implementation issues and [docs/ARCHITECTURE.md](../ARCHITECTURE.md).

## Template

```markdown
# 0001: Short decision title

- Status: proposed
- Date: YYYY-MM-DD
- Related issue: #N

## Context
...

## Decision
...

## Alternatives considered
...

## Consequences and verification
...
```
