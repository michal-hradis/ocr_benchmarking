# 0013: Opt-in network access for hosted-API engines

- Status: accepted
- Date: 2026-10-30
- Related issues: [#103](https://github.com/michal-hradis/ocr_benchmarking/issues/103#issuecomment-6524446229) (decision), [#102](https://github.com/michal-hradis/ocr_benchmarking/issues/102), [#57](https://github.com/michal-hradis/ocr_benchmarking/issues/57)

## Context

Engine containers run with `--network none` ([ADR 0007](0007-runner-and-engine-protocol.md), [ADR 0012](0012-inference-servers-inside-engine-containers.md)). Hosted OCR services such as Mistral OCR 4.1 ([#103](https://github.com/michal-hradis/ocr_benchmarking/issues/103)), and later Nanonets OCR-3 (#104), Transkribus (#118) and OpenRouter vision models (#57), only run on the provider's servers. They need outbound access and an API key. Each request is paid, and the page image leaves the machine.

## Decision

- An engine configuration may declare **`network = "egress"`** (default `"none"`) and **`secrets`**, the names of host environment variables it needs, for example `["MISTRAL_API_KEY"]`. Secrets are allowed only with egress.
- `ocrbench run start` and `run resume` **refuse such an engine unless `--allow-network` is passed**. They also refuse it when a named secret is unset or empty. Both checks happen before a run directory is created.
- The container then runs on Docker's default bridge network. Each secret is passed **by name** (`docker run --env NAME`), so Docker copies its value from the runner's environment. The value never appears in the command, `run.json`, `container.json`, logs or reports. The frozen configuration records only the names.
- Everything else is unchanged: the read-only root, the host UID/GID, image-only input without GT ([ADR 0002](0002-ground-truth-isolation.md)), and the same job and result protocol. The adapter sends only the image and its frozen request settings to the provider. It does not send the sample ID, metadata or file names.
- Hosted-API adapters pin an exact model identifier, never a `latest` alias, and check the model the response names. They keep the response unchanged and record the HTTP attempts, rate limiting, latency, usage and an estimated cost per sample. An API failure, or throttling that outlasts the retries, fails the sample (`engine-error`). A successful empty transcription is a result.
- Offline tests use a fake API on the loopback interface. Real requests are opt-in, through pytest's `--run-api` marker or an explicit `run start --allow-network`.
- For offline engines, `network` and `secrets` are omitted from the frozen configuration, so their configuration hashes, and existing runs, are unchanged.

## Alternatives considered

- *An allowlisting egress proxy*, with the container on an internal network that can reach only a host-side proxy for the provider's domain. This enforces the destination, but it needs a proxy service and per-provider setup. It was deferred. Docker's bridge network does not restrict destinations; the adapter code decides what is contacted.
- *A host-process adapter outside Docker.* This is simplest, but it gives up the container isolation and resource limits that every other engine has.

## Consequences and verification

- A hosted engine's results depend on the provider's service at request time. The pinned model ID and the recorded response model, request ID and timestamps are the provenance; exact reproduction later is not guaranteed.
- Engine time includes network and provider queueing. It is not comparable with on-device speed.
- `tests/test_runner.py` checks the refusal without `--allow-network`, the missing-secret refusal, the bridge network, the by-name secret and unchanged hashes for offline engines. `tests/test_mistral_ocr.py` runs the first such adapter against the fake API.
