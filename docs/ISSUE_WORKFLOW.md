# Issue workflow

Adapted from [semant-demo's issue workflow](https://github.com/DCGM/semant-demo/blob/main/docs/ISSUE_WORKFLOW.md). Issues define verifiable implementation work; [CONTRIBUTING.md](../CONTRIBUTING.md) governs branches, testing and PRs.

## Lifecycle

1. **Draft:** record an idea/problem. Prefix with `[Draft]` while important design decisions are unresolved.
2. **Refine:** check the current repository, issues/PRs and relevant architecture/ADRs. Add expected behavior, scope, dependencies and verification.
3. **Decide:** ask the maintainer about material contract or evaluation choices, record the decision in the issue, and add an [ADR](adr/README.md) if durable.
4. **Implement:** once ready, create a focused branch from `main`; submit a reviewable PR linked to the issue.
5. **Verify and review:** follow the relevant-checks policy in [CONTRIBUTING.md](../CONTRIBUTING.md#verification-and-tests), request human review and close only when acceptance criteria are met.

## Ready issue checklist

- [ ] Goal, required inputs/outputs and observable behavior are clear.
- [ ] Scope and non-goals are explicit.
- [ ] Prerequisites and existing related work are linked.
- [ ] Acceptance criteria and appropriate local verification are checkable.
- [ ] Unresolved choices that would change shared behavior are decided or the issue stays `[Draft]`.

Implementation-ready issues may specify an approach where it is an agreed constraint; otherwise avoid micromanaging internal code choices. Link a parent initiative such as [roadmap #1](https://github.com/michal-hradis/ocr_benchmarking/issues/1) instead of duplicating its contents.

## Consult before changing shared behavior

Get an explicit decision before changing the result schema, dataset/GT isolation, coordinate conventions, normalization/alignment, OCR input unit, shared runtime contract, or permitted dataset use. State alternatives/trade-offs when asking.

When preparing an issue, agents should distinguish implemented code from proposed functionality, inspect affected tests and contracts, and refine the original issue rather than replacing it. Check [AGENTS.md](../AGENTS.md) for agent-specific handoff rules.
