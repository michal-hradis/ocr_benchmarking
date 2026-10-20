# Docker host setup and base images

Engines run in per-engine Docker images on one Linux machine ([architecture](ARCHITECTURE.md#execution-and-reproducibility)). This page covers host setup, the shared base images and the conventions for engine images. How the runner invokes containers (mounts, metadata, lifecycle) is owned by [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7).

## Host setup (Ubuntu)

1. **Docker Engine.** Install from Docker's apt repository following the [official Ubuntu guide](https://docs.docker.com/engine/install/ubuntu/), then add yourself to the `docker` group (`sudo usermod -aG docker $USER`, log in again). Docker Desktop is not needed. **Warning:** membership in the `docker` group is root-equivalent on the host; grant it only on a trusted single-user development machine, or run Docker commands with `sudo` instead.
2. **NVIDIA driver** (GPU only). Install a driver supporting the CUDA version of the image you run; CUDA 13.0 images need driver ≥ 580. A host CUDA toolkit is **not** needed: CUDA user-space libraries come from the images.
3. **NVIDIA Container Toolkit** (GPU only). Follow the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run `sudo nvidia-ctk runtime configure --runtime=docker` and `sudo systemctl restart docker`.
4. **Preflight.** Run `make gpu-preflight`. It checks the host driver, the Docker daemon, the registered `nvidia` runtime, and runs `nvidia-smi -L` in a pinned CUDA base container (about 100 MB pulled once). Each failed step prints a hint.

**Without a GPU**, skip steps 2–4. Development checks (`make check`), the CPU base image and CPU engines work without a GPU. GPU engines are then unavailable rather than silently run on the CPU.

## Base image families

| Family | Directory | Default tag | Contents |
| --- | --- | --- | --- |
| CPU | `docker/base-cpu` | `ocrbench/base-cpu:py3.12-v1` | Debian bookworm slim, Python 3.12.15, uv, empty venv |
| CUDA 13.0 + PyTorch | `docker/base-cu130-torch` | `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` | Ubuntu 24.04 `nvidia/cuda` base, Python 3.12.15, torch 2.14.1+cu130 from a hash-locked requirements file |

Both images provide:

- a virtual environment at `/opt/venv`, already on `PATH`, and `uv` for installing engine dependencies;
- a non-root user `ocr` (UID 10001) as the final `USER`;
- a writable model cache at `/cache` (`XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME` point inside it);
- no model weights.

**Sharing rule.** An engine image starts `FROM` a base only if its dependencies are compatible with what the base pins (Python, CUDA major/minor and torch version). An engine needing different versions gets a **new family directory** (for example `base-cu126-torch` or a Paddle family) rather than an upgrade of the existing one; existing families are never changed in place. A changed family gets a new `-vN` tag so that recorded runs keep pointing at the image they used.

**Referencing shared bases (local-first).** The bases are built locally and are not published to a registry. A locally built image has an image ID but no `repository@sha256` registry digest, so engine images cannot pin bases by digest. The rule therefore is:

- **External** images (`FROM`, `COPY --from`) are pinned by registry digest.
- **Shared bases** are referenced by their versioned local tag, e.g. `FROM ocrbench/base-cpu:py3.12-v1`. No other local tags (such as `latest`) are allowed.
- A released `-vN` tag is **never reassigned** to different content: any change to a family's Dockerfile or lock bumps `BASE_VERSION`. Rebuilding the same version from the same pinned inputs is allowed.
- Identity is recorded by **image ID**: `make docker-*` prints the ID of each built base, and the runner ([#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7)) records the resolved image ID of every engine image it runs.

Publishing bases to a registry and pinning them by digest would also work, but would need a documented push/pull workflow; it is not set up now.

## Building and testing

| Command | What it does |
| --- | --- |
| `make docker-cpu` | Build the CPU base (small pull) |
| `make docker-cu130-torch` | Build the CUDA/torch base (several GB pulled; cached afterwards) |
| `make docker-lock` | Re-lock `docker/base-cu130-torch/requirements.txt` with hashes after editing `requirements.in` |
| `make docker-smoke` | Opt-in: run the built images with `--network none --read-only` and check Python, the non-root user and that torch imports without a GPU; also build a minimal derived image (`tests/fixtures/derived-cpu`) from the CPU base and check it inherits the base layers |
| `make gpu-preflight` | Opt-in: host/Docker/toolkit GPU check described above |
| `make gpu-smoke` | Opt-in: check that torch in the CUDA image sees the GPU and runs a matrix product |

The default `make test` only checks the conventions of every `Dockerfile` in the repository statically (pinned parents per the rule above, non-root final user, hashed requirements); it never builds or runs images.

## Engine image conventions

Engine adapters ([#11–#17](https://github.com/michal-hradis/ocr_benchmarking/issues/1)) follow these rules:

- **Pin everything.** External `FROM` and `COPY --from` images by digest, shared bases by their versioned `ocrbench/base-*:...-vN` tag; Python packages by version with hashes (uv lock or `uv pip compile --generate-hashes`); engine source by commit or release; model weights by revision and checksum.
- **Install as root, run as `ocr`.** Switch to `USER root` for system packages and back to `USER ocr` at the end.
- **No weights in layers unless small.** Large weights go to the `/cache` volume, downloaded by an explicit, separately invoked step, never during an ordinary test or benchmark run. Record their revision and checksum.
- **Inference runs offline.** Containers must work with `--network none` once weights are cached.
- **Read-only inputs.** Images and other inputs are mounted read-only; the container writes only to its output location and `/cache`. The root filesystem should work with `--read-only` (use `--tmpfs /tmp` if the engine needs scratch space).
- **Resource limits come from the runner** (`--cpus`, `--memory`, `--gpus`, timeouts) and are recorded in the run configuration ([#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7)); images must not assume unlimited resources.
- **Ground truth never enters an engine image or container** ([ADR 0002](adr/0002-ground-truth-isolation.md)).
