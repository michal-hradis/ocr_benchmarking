# 0007: Runner lifecycle and engine container protocol

- Status: proposed
- Date: 2026-10-20
- Related issue: [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7) (maintainer decisions: [comment](https://github.com/michal-hradis/ocr_benchmarking/issues/7#issuecomment-6300080767))

## Context

ADR 0003 requires per-sample execution records independent of predictions, and ADR 0002 limits engine containers to the image and allowed metadata. Engines range from fast CPU OCR to VLMs that need tens of seconds to load a model and can hang or run out of memory on a single large newspaper page. GPU inference is expensive, so a crash must not lose finished work, and a rerun must never silently replace a valid result.

## Decision

- **One container per run.** A container processes all pending samples in job order and stops. The engine writes each sample's output directory with `result.json` last. If no sample finishes within the configured timeout (plus a startup allowance), or the container exits early, the sample in progress fails (`timeout` or `crash`) and a new container continues with the rest.
- **Input.** A read-only mount holds only `job.json` and byte-identical sample images. Every sample has its ID and unit. `language`, `doc_type` and `year` are passed only when the engine configuration enables them, and the selection is frozen with the run. GT paths and content, splits and checksums are never passed.
- **Output protocol.** Native files plus a small `result.json` with the status the engine claims (`ok`, `truncated`, `partial`, `failed`) and the primary output file. The runner decides the final status (`success`, `truncated`, `partial`, `failure` with a reason). Timeouts, crashes, missing and invalid output override the claim, and invalid output is never `success`. Text, Markdown and HTML primary outputs are converted on the host (ADR 0006).
- **Run store.** Each run has a directory with a frozen configuration (engine configuration and its hash, resolved image ID, dataset, host). Every attempt for a sample gets its own numbered record directory. Nothing is overwritten. Continuing a run skips samples that have a record and retries failed ones only on request. It is refused if the image tag resolves to a different image.
- **Isolation.** Containers run with no network, a read-only root, the host UID/GID and the configured resource limits. The runner never builds or pulls images.
- Engine configurations are TOML files.

## Alternatives considered

- *One container per sample.* Rejected: it repeats model loading for every page, which is costly for VLMs. Exact per-sample timeouts and crash attribution are kept by the progress check instead.
- *Fixed-size batches of samples.* Rejected as unnecessary: restarting after a failure already limits the damage of one bad page.
- *A long-lived server container.* Rejected for now: it needs an IPC or network protocol inside `--network none` and is more complex for adapters.
- *Native files only, without `result.json`.* Rejected: engines could not report truncation or their own failures.
- *Rerunning a sample in place.* Rejected: it can silently replace a valid result.

## Consequences and verification

- The protocol, statuses and run directory are documented in [RUNNER.md](../RUNNER.md) and implemented in `ocrbench.runner`, `ocrbench.engine` and `ocrbench run start|resume|status`.
- Engines must process samples in job order and write `result.json` last and atomically. A sample that makes the engine crash is attributed to the first unfinished sample.
- Offline tests run a fake engine as a host process. They cover every status, restarts after crashes and timeouts, interruption and continuation, retries as new attempts, GT absence from the job and inputs, and byte-identical images. Opt-in Docker and GPU tests run the same engine in a container through the CLI.
- If the runner process itself is killed (not interrupted), a running container is not stopped. Its later output is ignored, and continuing the run starts a new container.
- Evaluation (#8–#10) reads `record.json` and `prediction.json` and must treat pending samples as not run.
