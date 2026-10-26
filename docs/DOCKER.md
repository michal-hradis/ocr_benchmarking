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
| CUDA 13.0 + vLLM | `docker/base-cu130-vllm` | `ocrbench/base-cu130-vllm:vllm0.20.1-py3.12-v1` | The same `nvidia/cuda` base, Python 3.12.15, `gcc` (Triton compiles kernel launchers at run time), vLLM 0.20.1 with the torch 2.11.0 (CUDA 13.0) it pins and transformers 5.12.1, from a hash-locked requirements file. A separate family because vLLM pins its own torch. For engines that serve their model with vLLM inside the container ([ADR 0012](adr/0012-inference-servers-inside-engine-containers.md)) |

All base images provide:

- a virtual environment at `/opt/venv`, already on `PATH`, and `uv` for installing engine dependencies;
- a non-root user `ocr` (UID 10001) as the final `USER`;
- a writable model cache at `/cache` (`XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME` point inside it);
- no model weights.

**Sharing rule.** An engine image starts `FROM` a base only if its dependencies are compatible with what the base pins (Python, CUDA major/minor and torch version). An engine needing different versions gets a **new family directory** (for example `base-cu126-torch` or a Paddle family) rather than an upgrade of the existing one; existing families are never changed in place. A changed family gets a new `-vN` tag so that recorded runs keep pointing at the image they used. An engine whose dependencies fit no family and would gain nothing from a new shared one may build a **standalone image** from a digest-pinned upstream image with the same conventions ([ADR 0004](adr/0004-docker-base-image-families.md)); [`engines/ocrd-calamari`](../engines/ocrd-calamari/README.md) (Python 3.11 for TensorFlow 2.15) is one. An engine whose main dependencies fit a family but one component does not may instead add a **second venv** for that component inside its image; [`engines/pylaia`](../engines/pylaia/README.md) runs PyLaia (Python 3.10, torch 1.13) beside Kraken on the torch base.

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
| `make docker-cu130-vllm` | Build the CUDA/vLLM base (several GB pulled; 8.7 GB image) |
| `make docker-lock` | Re-lock the hashed `requirements.txt` of the CUDA bases and of the Surya 2 (against the vLLM base), PaddleOCR, PP-StructureV3 and PaddleOCR-VL (against the Paddle base) and Kraken, LightOnOCR, FireRed-OCR, OvisOCR2, GLM-OCR, MinerU2.5-Pro, Chandra OCR 2, olmOCR-2, DeepSeek-OCR-2, Infinity-Parser2-Flash, PERO OCR, Surya and docTR (against the torch base) engines, and of the standalone OCR-D/Calamari image (Python 3.11) and the PyLaia venv of the PyLaia image (Python 3.10, packages as of PyLaia 1.1.2's release), after editing a `requirements.in` |
| `make docker-smoke` | Opt-in: run the built images with `--network none --read-only` and check Python, the non-root user and that torch imports without a GPU; also build a minimal derived image (`tests/fixtures/derived-cpu`) from the CPU base and check it inherits the base layers; build the fake engine (`tests/fixtures/fake-engine`) and run it with `ocrbench run start` ([RUNNER.md](RUNNER.md)) |
| `make docker-tesseract` | Build the Tesseract engine image from the CPU base ([engines/tesseract](../engines/tesseract/README.md)) |
| `make tesseract-best` | Opt-in: download the pinned `tessdata_best` models (48 MB) for the Tesseract best presets into `$(OCRBENCH_CACHE)` |
| `make tesseract-smoke` | Opt-in: build the Tesseract image and run all its presets on synthetic samples through the runner, evaluation and reports (the best presets need `make tesseract-best`) |
| `make docker-paddleocr` | Build the PaddleOCR PP-OCRv6 GPU engine image from the Paddle base ([engines/paddleocr](../engines/paddleocr/README.md)) |
| `make paddleocr-smoke` | Opt-in, GPU: build the PaddleOCR image and run both presets (`det4000`, `det8000`) on synthetic samples, including a 5400-pixel page, through the runner, evaluation and reports |
| `make docker-ppstructurev3` | Build the PP-StructureV3 GPU engine image from the Paddle base, with its models ([engines/ppstructurev3](../engines/ppstructurev3/README.md)) |
| `make ppstructurev3-smoke` | Opt-in, GPU: build the PP-StructureV3 image and run it on synthetic samples, including two-column and newspaper pages, through the runner, evaluation and reports |
| `make docker-paddleocr-vl` | Build the PaddleOCR-VL-1.6 GPU engine image from the Paddle base, without models ([engines/paddleocr-vl](../engines/paddleocr-vl/README.md)) |
| `make paddleocr-vl-model` | Opt-in: download the pinned PaddleOCR-VL-1.6 and PP-DocLayoutV3 models (2 GB) into `$(OCRBENCH_CACHE)` |
| `make paddleocr-vl-smoke` | Opt-in, GPU: build the PaddleOCR-VL image and run it on synthetic samples, including two-column and newspaper pages, through the runner, evaluation and reports |
| `make docker-ocrd-calamari` | Build the standalone OCR-D/Calamari CPU engine image (Python 3.11, no shared base), with its models ([engines/ocrd-calamari](../engines/ocrd-calamari/README.md)) |
| `make ocrd-calamari-smoke` | Opt-in: build the OCR-D/Calamari image and run its presets on synthetic samples, including two-column and newspaper pages, through the runner, evaluation and reports |
| `make docker-surya` | Build the Surya OCR GPU engine image from the torch base, without weights ([engines/surya](../engines/surya/README.md)) |
| `make surya-model` | Opt-in: download the pinned Surya checkpoints (3 GB) into `$(OCRBENCH_CACHE)` |
| `make surya-smoke` | Opt-in, GPU: build the Surya image and run it on synthetic samples, including two-column and newspaper pages, through the runner, evaluation and reports |
| `make docker-doctr` | Build the docTR GPU engine image from the torch base, without weights ([engines/doctr](../engines/doctr/README.md)) |
| `make doctr-model` | Opt-in: download the pinned docTR checkpoints (129 MB) into `$(OCRBENCH_CACHE)` |
| `make doctr-smoke` | Opt-in, GPU: build the docTR image and run both presets on synthetic samples, including two-column and newspaper pages, through the runner, evaluation and reports |
| `make docker-kraken` | Build the Kraken GPU engine image from the torch base ([engines/kraken](../engines/kraken/README.md)) |
| `make kraken-smoke` | Opt-in, GPU: build the Kraken image and run it on synthetic samples through the runner, evaluation and reports |
| `make docker-pylaia` | Build the PyLaia GPU engine image from the torch base, with its model; it adds a second venv with Python 3.10 and torch 1.13.1 for PyLaia, and its build context is `engines/` ([engines/pylaia](../engines/pylaia/README.md)) |
| `make pylaia-smoke` | Opt-in, GPU: build the PyLaia image and run it on synthetic samples through the runner, evaluation and reports |
| `make docker-lightonocr` | Build the LightOnOCR-2 GPU engine image from the torch base, without weights ([engines/lightonocr](../engines/lightonocr/README.md)) |
| `make lightonocr-model` | Opt-in: download the pinned LightOnOCR-2 weights (2 GB) into the cache directory `OCRBENCH_CACHE` (default `~/.cache/ocrbench`), checked against `models.sha256` |
| `make lightonocr-smoke` | Opt-in, GPU: build the LightOnOCR-2 image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-fireredocr` | Build the FireRed-OCR GPU engine image from the torch base, without weights ([engines/fireredocr](../engines/fireredocr/README.md)) |
| `make fireredocr-model` | Opt-in: download the pinned FireRed-OCR weights (4.3 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make fireredocr-smoke` | Opt-in, GPU: build the FireRed-OCR image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-ovisocr2` | Build the OvisOCR2 GPU engine image from the torch base, without weights ([engines/ovisocr2](../engines/ovisocr2/README.md)) |
| `make ovisocr2-model` | Opt-in: download the pinned OvisOCR2 weights (1.7 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make ovisocr2-smoke` | Opt-in, GPU: build the OvisOCR2 image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-glm-ocr` | Build the GLM-OCR GPU engine image from the torch base, without weights ([engines/glm-ocr](../engines/glm-ocr/README.md)) |
| `make glm-ocr-model` | Opt-in: download the pinned GLM-OCR weights (2.7 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make glm-ocr-smoke` | Opt-in, GPU: build the GLM-OCR image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-mineru25-pro` | Build the MinerU2.5-Pro GPU engine image from the torch base, without weights ([engines/mineru25-pro](../engines/mineru25-pro/README.md)) |
| `make mineru25-pro-model` | Opt-in: download the pinned MinerU2.5-Pro weights (2.3 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make mineru25-pro-smoke` | Opt-in, GPU: build the MinerU2.5-Pro image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-chandra-ocr2` | Build the Chandra OCR 2 GPU engine image from the torch base, without weights ([engines/chandra-ocr2](../engines/chandra-ocr2/README.md)) |
| `make chandra-ocr2-model` | Opt-in: download the pinned Chandra OCR 2 weights (10.6 GB, modified OpenRAIL-M license) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make chandra-ocr2-smoke` | Opt-in, GPU: build the Chandra OCR 2 image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-olmocr2` | Build the olmOCR-2 GPU engine image from the torch base, without weights ([engines/olmocr2](../engines/olmocr2/README.md)) |
| `make olmocr2-model` | Opt-in: download the pinned olmOCR-2 weights (16.6 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make olmocr2-smoke` | Opt-in, GPU: build the olmOCR-2 image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-deepseek-ocr2` | Build the DeepSeek-OCR-2 GPU engine image from the torch base, without weights ([engines/deepseek-ocr2](../engines/deepseek-ocr2/README.md)) |
| `make deepseek-ocr2-model` | Opt-in: download the pinned DeepSeek-OCR-2 weights and model code (6.8 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make deepseek-ocr2-smoke` | Opt-in, GPU: build the DeepSeek-OCR-2 image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-hunyuanocr` | Build the HunyuanOCR-1.5 GPU engine image from the torch base, without weights ([engines/hunyuanocr](../engines/hunyuanocr/README.md)) |
| `make hunyuanocr-model` | Opt-in: download the pinned HunyuanOCR-1.5 weights (2.2 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make hunyuanocr-smoke` | Opt-in, GPU: build the HunyuanOCR-1.5 image and run both presets with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-dots-mocr` | Build the dots.mocr GPU engine image from the torch base, without weights ([engines/dots-mocr](../engines/dots-mocr/README.md)) |
| `make dots-mocr-model` | Opt-in: download the pinned dots.mocr weights and model code (6.1 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make dots-mocr-smoke` | Opt-in, GPU: build the dots.mocr image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
| `make docker-surya2` | Build the Surya 2 GPU engine image from the vLLM base, without weights ([engines/surya2](../engines/surya2/README.md)) |
| `make surya2-model` | Opt-in: download the pinned surya-ocr-2 weights (1.4 GB) into `OCRBENCH_CACHE`, checked against `models.sha256` |
| `make surya2-smoke` | Opt-in, GPU: build the Surya 2 image and run it with the cached weights on synthetic samples through the runner, evaluation and reports |
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

Engine adapters ([#11–#17](https://github.com/michal-hradis/ocr_benchmarking/issues/1)) live in `engines/<name>/` and follow these rules; [`engines/tesseract`](../engines/tesseract/README.md) (CPU), [`engines/paddleocr`](../engines/paddleocr/README.md) and [`engines/kraken`](../engines/kraken/README.md) (GPU, Python dependencies locked against a base) are examples, and [`engines/lightonocr`](../engines/lightonocr/README.md), [`engines/fireredocr`](../engines/fireredocr/README.md), [`engines/ovisocr2`](../engines/ovisocr2/README.md), [`engines/glm-ocr`](../engines/glm-ocr/README.md), [`engines/mineru25-pro`](../engines/mineru25-pro/README.md), [`engines/chandra-ocr2`](../engines/chandra-ocr2/README.md), [`engines/olmocr2`](../engines/olmocr2/README.md), [`engines/deepseek-ocr2`](../engines/deepseek-ocr2/README.md), [`engines/hunyuanocr`](../engines/hunyuanocr/README.md), [`engines/dots-mocr`](../engines/dots-mocr/README.md), [`engines/infinityparser2`](../engines/infinityparser2/README.md), [`engines/surya`](../engines/surya/README.md) and [`engines/doctr`](../engines/doctr/README.md) of weights in `/cache`, and [`engines/pero`](../engines/pero/README.md) of user-supplied model bundles in `/cache`:

- **Pin everything.** External `FROM` and `COPY --from` images by digest, shared bases by their versioned `ocrbench/base-*:...-vN` tag; Python packages by version with hashes (uv lock or `uv pip compile --generate-hashes`); engine source by commit or release; model weights by revision and checksum.
- **Install as root, run as `ocr`.** Switch to `USER root` for system packages and back to `USER ocr` at the end.
- **No weights in layers unless small.** Large weights go to the `/cache` volume, downloaded by an explicit, separately invoked step, never during an ordinary test or benchmark run. Record their revision and checksum.
- **Inference runs offline.** Containers must work with `--network none` once weights are cached.
- **Read-only inputs.** Images and other inputs are mounted read-only; the container writes only to its output location and `/cache`. The root filesystem should work with `--read-only` (use `--tmpfs /tmp` if the engine needs scratch space).
- **Resource limits come from the runner** (`--cpus`, `--memory`, `--gpus`, timeouts) and are recorded in the run configuration ([RUNNER.md](RUNNER.md#engine-configuration)); images must not assume unlimited resources.
- **Runner user.** The runner starts containers with the host user's UID/GID and `HOME=/tmp`, not as `ocr`, so files in `/output` and `/cache` belong to the host user. Do not depend on files writable only by `ocr`.
- **Inference servers stay inside the container.** An engine may run a server such as vLLM as a child process on the loopback interface ([ADR 0012](adr/0012-inference-servers-inside-engine-containers.md)); it never starts other containers. Docker mounts the runner's `/tmp` with `noexec`, so kernels compiled at run time (Triton, torch.compile) must be cached under `/cache`.
- **Ground truth never enters an engine image or container** ([ADR 0002](adr/0002-ground-truth-isolation.md)).
