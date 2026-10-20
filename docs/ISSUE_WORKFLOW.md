# Issue workflow: proposal to implementable task

GitHub issues are the source for proposed behavior and implementation requirements. This guide is adapted from [semant-demo's issue workflow](https://github.com/DCGM/semant-demo/blob/main/docs/ISSUE_WORKFLOW.md) for a smaller, image-only OCR benchmark. For implementing, testing and PRs use [CONTRIBUTING.md](../CONTRIBUTING.md); agents also follow [AGENTS.md](../AGENTS.md).

**A ready issue states a testable outcome and resolves decisions material to that outcome.** It need not prescribe routine internal code organization, but it may define necessary interfaces, algorithm constraints or Docker compatibility.

## Lifecycle

1. **Draft:** record the goal/problem with enough context, even if decisions are unresolved. Prefix the title `[Draft]` when it is not implementation-ready.
2. **Refine:** inspect the repository and existing issues/PRs; distinguish implemented behavior from proposals; add dependencies, acceptance criteria, non-goals, constraints, and verification.
3. **Decide:** consult the maintainer on significant open questions; record the decision in the original issue. For a durable architectural contract use the [ADR convention](adr/README.md).
4. **Implement:** once ready, work on a dedicated branch from current `main`. Keep each PR focused and link it to its issue.
5. **Verify and review:** run **relevant local checks** and report what actually ran. Request human review; merge only after review. Close the issue only when its acceptance criteria are met.

The roadmap [#1](https://github.com/michal-hradis/ocr_benchmarking/issues/1) links the initial dependency-ordered backlog. A draft issue does not imply implementation approval or a fixed priority.

## Minimum content for a ready issue

- **Goal / motivation:** what problem is solved and for whom.
- **Requirements:** inputs, outputs and observable behavior, including failures/partial outputs if applicable.
- **Scope and non-goals:** explicit boundary with neighboring features.
- **Dependencies:** issue links for shared contracts, earlier components, related PRs and any engine constraints.
- **Acceptance criteria:** concise, checkable behavior, not a generic list of development tasks.
- **Verification:** what focused tests or manual checks can demonstrate correctness. List GPU/model/manual tests separately from deterministic checks.
- **Decisions / constraints:** relevant PAGE XML, JSON region/line schema, evaluation metrics, permissions, image tiling or Docker conventions.

An issue can be brief when the change is small; do not reproduce every common rule from AGENTS/CONTRIBUTING. Use parent issues for larger themes and subissues for independently reviewable deliverables.

## When to consult a maintainer

Resolve substantial ambiguity before coding, especially for:

- canonical dataset manifests, JSON schema and coordinate conventions;
- text extraction, normalization, CER/WER alignment or other leaderboard-affecting changes;
- allowing page crops/tiling, GT exposure or non-image inputs;
- major engine runtime/platform changes, external services, expensive GPU runs or external data licensing;
- adding real datasets, making destructive changes, or expanding an issue materially.

Document the options and recommendation; do not quietly choose an assumption that affects future adapters or scores.

## Agent-assisted issue refinement

Agents should check related issues/PRs, current files, decisions and known tests, then suggest a minimally scoped implementation outline, observable acceptance criteria and targeted verification. Treat external model documentation as evidence to verify for pinning, not as permission to download or run it. Clearly separate **verified current implementation** from roadmap intentions. Prefer refining an existing issue over replacing it.

## Verification and PR handoff

There is **no initial CI** and **no mandatory full-suite run**. The contributor selects the tests that cover the changed behavior, performs them locally, and reports exact commands/results. A docs-only change can be validated by reading and checking links. A parser/matcher change warrants targeted regression and edge-case tests; Docker/model inference may need separately authorized smoke testing.

The PR should link `Closes #N` only if it fully completes that issue. Include behavior/files changed, relevant test evidence, checks not run, open risks and any newly discovered follow-ups. Human review is required before merge. See the [PR template](../.github/PULL_REQUEST_TEMPLATE.md).

## Readiness checklist

- [ ] Goal, observable requirements and acceptance criteria are clear.
- [ ] Scope and non-goals are explicit.
- [ ] Dependencies and cross-issue contracts are identified.
- [ ] Important unresolved decisions were made or the issue remains `[Draft]`.
- [ ] Relevant local verification can be performed, with GPU/live checks clearly separated.
