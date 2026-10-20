# 0004: Docker base image families

- Status: proposed
- Date: 2026-10-20
- Related issue: [#4](https://github.com/michal-hradis/ocr_benchmarking/issues/4)

## Context

Engines (#11–#17) need conflicting stacks: classic CPU OCR, PyTorch VLMs, Paddle, vLLM-style servers. Rebuilding CUDA and PyTorch for every engine is slow and wastes disk, but forcing all engines onto one set of versions would break some of them. Results must stay reproducible after base images change.

## Decision

- Shared bases are grouped in **families**, each pinning one Python version and, for GPU families, one CUDA and one framework version. The initial families are `base-cpu` (Python 3.12) and `base-cu130-torch` (CUDA 13.0, torch 2.14.1+cu130, Python 3.12).
- An engine uses a family only if its dependencies are compatible with it. Different requirements get a **new family**; an existing family is never upgraded in place. Changes to a family produce a new `-vN` tag.
- Upstream images are pinned by digest, and the framework packages by version and hash.
- Shared bases are built locally (no registry). Engine images reference them by an immutable versioned tag `ocrbench/base-<family>:<versions>-vN`, which is never reassigned to different content; the resolved image ID is what identifies a run's image.
- Images run as a non-root user, keep model weights in a separate `/cache` volume, and must run with `--network none` once weights are present.
- The host needs only the NVIDIA driver and NVIDIA Container Toolkit; no host CUDA toolkit.

## Alternatives considered

- *One universal GPU image.* Rejected: incompatible engines would force version compromises or break.
- *Fully independent images per engine.* Rejected as the default: it repeats multi-GB CUDA/framework layers. It remains allowed when no family fits.
- *Host CUDA installations.* Rejected: they couple engines to host state and conflict across versions.
- *Publishing bases to a registry and pinning them by digest.* Deferred: valid, but needs a push/pull workflow and registry access that a single local machine does not need yet.

## Consequences and verification

- Families are defined in `docker/`; conventions and commands are in [DOCKER.md](../DOCKER.md).
- `make test` checks parent pins (external digest or local versioned base tag), non-root users and hashed requirements statically for every Dockerfile. `make docker-smoke` (including a derived CPU image build) and `make gpu-smoke` are opt-in runtime checks.
- The runner (#7) records the resolved image ID actually used, so recorded runs remain identifiable when families change.
