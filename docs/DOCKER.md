# Docker host setup and base images

Engines run in per-engine Docker images on one Linux machine ([architecture](ARCHITECTURE.md#execution-and-reproducibility)). This page covers host setup, the shared base images and the conventions for engine images. How the runner invokes containers (mounts, metadata, lifecycle) is described in [RUNNER.md](RUNNER.md).

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
| CUDA 13.0 + PaddlePaddle | `docker/base-cu130-paddle` | `ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1` | The same `nvidia/cuda` base, Python 3.12.15, `libgomp1`, paddlepaddle-gpu 3.4.0 (cu130) from a hash-locked requirements file. A separate family because Paddle pins other cuDNN and nvJitLink wheels than torch; numpy is held below 2.4 for PaddleX 3.7 |

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
- Identity is recorded by **image ID**: `make docker-*` prints the ID of each built base, and the runner ([RUNNER.md](RUNNER.md)) records the resolved image ID of every engine image it runs.

Publishing bases to a registry and pinning them by digest would also work, but would need a documented push/pull workflow; it is not set up now.

## Building and testing

| Command | What it does |
| --- | --- |
| `make docker-cpu` | Build the CPU base (small pull) |
| `make docker-cu130-torch` | Build the CUDA/torch base (several GB pulled; cached afterwards) |
| `make docker-cu130-paddle` | Build the CUDA/Paddle base (several GB pulled) |
| `make docker-lock` | Re-lock the hashed `requirements.txt` of both CUDA bases and of the PaddleOCR (against the Paddle base) and Kraken, LightOnOCR, FireRed-OCR, OvisOCR2, Infinity-Parser2-Flash and PERO OCR (against the torch base) engines after editing a `requirements.in` |
| `make docker-smoke` | Opt-in: run the built images with `--network none --read-only` and check Python, the non-root user and that torch imports without a GPU; also build a minimal derived image (`tests/fixtures/derived-cpu`) from the CPU base and check it inherits the base layers; build the fake engine (`tests/fixtures/fake-engine`) and run it with `ocrbench run start` ([RUNNER.md](RUNNER.md)) |
| `make docker-tesseract` | Build the Tesseract engine image from the CPU base ([engines/tesseract](../engines/tesseract/README.md)) |
| `make tesseract-best` | Opt-in: download the pinned `tessdata_best` models (48 MB) for the Tesseract best presets into `$(OCRBENCH_CACHE)` |
| `make tesseract-smoke` | Opt-in: build the Tesseract image and run all its presets on synthetic samples through the runner, evaluation and reports (the best presets need `make tesseract-best`) |
| `make docker-paddleocr` | Build the PaddleOCR PP-OCRv6 GPU engine image from the Paddle base ([engines/paddleocr](../engines/paddleocr/README.md)) |
| `make paddleocr-smoke` | Opt-in, GPU: build the PaddleOCR image and run it on synthetic samples through the runner, evaluation and reports |
| `make docker-kraken` | Build the Kraken GPU engine image from the torch base ([engines/kraken](../engines/kraken/README.md)) |
| `make kraken-smoke` | Opt-in, GPU: build the Kraken image and run it on synthetic samples through the runner, evaluation and reports |
| `make docker-lightonocr` | Build the LightOnOCR-2 GPU engine image from the torch base, without weights ([engines/lightonocr](../engines/lightonocr/README.md)) |
| `make lightonocr-model` | Opt-in: download the pinned LightOnOCR-2 weights (2 GB) into the cache directory `OCRBENCH_CACHE` (default `~/.cache/ocrbench`), checked against `models.sha256` |
| `make lightonocr-smoke` | Opt-in, GPU: build the LightOnOCR-2 image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-fireredocr` | Build the FireRed-OCR GPU engine image from the torch base, without weights ([engines/fireredocr](../engines/fireredocr/README.md)) |
| `make fireredocr-model` | Opt-in: download the pinned FireRed-OCR weights (4.3 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make fireredocr-smoke` | Opt-in, GPU: build the FireRed-OCR image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-ovisocr2` | Build the OvisOCR2 GPU engine image from the torch base, without weights ([engines/ovisocr2](../engines/ovisocr2/README.md)) |
| `make ovisocr2-model` | Opt-in: download the pinned OvisOCR2 weights (1.7 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make ovisocr2-smoke` | Opt-in, GPU: build the OvisOCR2 image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-infinityparser2` | Build the Infinity-Parser2-Flash GPU engine image from the torch base, without weights ([engines/infinityparser2](../engines/infinityparser2/README.md)) |
| `make infinityparser2-model` | Opt-in: download the pinned Infinity-Parser2-Flash weights (4.4 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make infinityparser2-smoke` | Opt-in, GPU: build the Infinity-Parser2-Flash image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-pero` | Build the PERO OCR GPU engine image from the torch base, without models ([engines/pero](../engines/pero/README.md)) |
| `make pero-bundles` | Opt-in: copy the user-supplied PERO model bundles from `PERO_BUNDLES` into `OCRBENCH_CACHE`, checked against `bundles.sha256`; nothing is downloaded |
| `make pero-smoke` | Opt-in, GPU: build the PERO OCR image and run it with the cached bundles on synthetic samples through the runner, evaluation and reports |
| `make gpu-preflight` | Opt-in: host/Docker/toolkit GPU check described above |
| `make gpu-smoke` | Opt-in: check that torch in the CUDA image sees the GPU and runs a matrix product, and that the runner passes `resources.gpus` to an engine container |

The default `make test` only checks the conventions of every `Dockerfile` in the repository statically (pinned parents per the rule above, non-root final user, hashed requirements); it never builds or runs images.

## Engine image conventions

Engine adapters ([#11–#17](https://github.com/michal-hradis/ocr_benchmarking/issues/1)) live in `engines/<name>/` and follow these rules; [`engines/tesseract`](../engines/tesseract/README.md) (CPU), [`engines/paddleocr`](../engines/paddleocr/README.md) and [`engines/kraken`](../engines/kraken/README.md) (GPU, Python dependencies locked against a base) are examples, and [`engines/lightonocr`](../engines/lightonocr/README.md), [`engines/fireredocr`](../engines/fireredocr/README.md), [`engines/ovisocr2`](../engines/ovisocr2/README.md) and [`engines/infinityparser2`](../engines/infinityparser2/README.md) of large weights in `/cache`, and [`engines/pero`](../engines/pero/README.md) of user-supplied model bundles in `/cache`:

- **Pin everything.** External `FROM` and `COPY --from` images by digest, shared bases by their versioned `ocrbench/base-*:...-vN` tag; Python packages by version with hashes (uv lock or `uv pip compile --generate-hashes`); engine source by commit or release; model weights by revision and checksum.
- **Install as root, run as `ocr`.** Switch to `USER root` for system packages and back to `USER ocr` at the end.
- **No weights in layers unless small.** Large weights go to the `/cache` volume, downloaded by an explicit, separately invoked step, never during an ordinary test or benchmark run. Record their revision and checksum.
- **Inference runs offline.** Containers must work with `--network none` once weights are cached.
- **Read-only inputs.** Images and other inputs are mounted read-only; the container writes only to its output location and `/cache`. The root filesystem should work with `--read-only` (use `--tmpfs /tmp` if the engine needs scratch space).
- **Resource limits come from the runner** (`--cpus`, `--memory`, `--gpus`, timeouts) and are recorded in the run configuration ([RUNNER.md](RUNNER.md#engine-configuration)); images must not assume unlimited resources.
- **Runner user.** The runner starts containers with the host user's UID/GID and `HOME=/tmp`, not as `ocr`, so files in `/output` and `/cache` belong to the host user. Do not depend on files writable only by `ocr`.
- **Ground truth never enters an engine image or container** ([ADR 0002](adr/0002-ground-truth-isolation.md)).
