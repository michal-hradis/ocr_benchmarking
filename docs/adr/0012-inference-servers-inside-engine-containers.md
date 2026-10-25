# 0012: Inference servers inside engine containers

- Status: accepted
- Date: 2026-10-25
- Related issue: [#67](https://github.com/michal-hradis/ocr_benchmarking/issues/67)

## Context

Several VLM pipelines are built around an inference server, usually vLLM, reached over an OpenAI-compatible API. Surya 2 has no other backend for its model: by default it starts a sibling `vllm/vllm-openai` Docker container. The engines so far ran their models in-process with transformers, and their READMEs name the server and its own CUDA stack as the reason not to use vLLM. [ADR 0007](0007-runner-and-engine-protocol.md) rejects a long-lived server *container* outside the engine protocol.

## Decision

- An engine may start an inference server as a **child process of its adapter inside the engine container**, listening on the loopback interface only. The runner protocol is unchanged: the adapter still reads `job.json`, processes samples in order and writes `result.json`. The server runs under `--network none`, and the adapter stops it at the end.
- An engine never starts other containers and never uses the Docker socket.
- The server reserves at most 0.6 of the GPU's memory (vLLM `gpu_memory_utilization` ≤ 0.6); the adapter rejects a larger share.
- Server and model versions, the full server command and the GPU memory share are pinned and recorded per sample.
- vLLM pins its own torch, so it gets its own base family `base-cu130-vllm` ([ADR 0004](0004-docker-base-image-families.md), [DOCKER.md](../DOCKER.md)).
- Kernels compiled at run time (Triton, torch.compile, FlashInfer) are cached under `/cache`, because the runner's `/tmp` is mounted `noexec`.

## Alternatives considered

- *Sibling server containers (Surya's default).* Rejected: they need the Docker socket, escape the runner's isolation and resource limits, and are not recorded with the run.
- *transformers in-process with Surya's prompt and parser.* No server, but not Surya's supported inference path; chosen against for Surya 2.
- *llama.cpp with GGUF weights.* Different weights and quantization than the reference model.

## Consequences and verification

- Start-up is slower (CUDA graph capture and compilation; about 1 minute with a warm cache on an RTX 4090), and vLLM reserves a fixed share of GPU memory (`gpu_memory_utilization`) that must fit beside other processes on the GPU.
- Output can depend on server features such as speculative decoding; engine READMEs state them.
- `tests/test_surya2.py` runs the adapter with a fake server executable offline, including start-up failure and a server that exits; `make surya2-smoke` runs the real server.
