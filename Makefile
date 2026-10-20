# Local developer commands. There is no CI; run the targets relevant to your change.
.PHONY: help setup lint format test check docker-cpu docker-cu130-torch docker-lock \
        docker-smoke gpu-preflight gpu-smoke

UV ?= uv
BASE_VERSION ?= 1
CPU_IMAGE ?= ocrbench/base-cpu:py3.12-v$(BASE_VERSION)
CU130_TORCH_IMAGE ?= ocrbench/base-cu130-torch:torch2.14.1-py3.12-v$(BASE_VERSION)
export CPU_IMAGE CU130_TORCH_IMAGE

help:  ## List targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-20s %s\n", $$1, $$2}'

setup:  ## Create .venv from uv.lock (Python 3.12 is downloaded by uv if missing)
	$(UV) sync --locked

lint:  ## Ruff lint and format check
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .

format:  ## Apply Ruff fixes and formatting
	$(UV) run --locked ruff check --fix .
	$(UV) run --locked ruff format .

test:  ## Offline unit tests (no GPU, Docker, network or model downloads); pass extra args via ARGS=
	$(UV) run --locked pytest $(ARGS)

check: lint test  ## lint + test

docker-cpu:  ## Build the CPU base image
	docker build -t $(CPU_IMAGE) docker/base-cpu

docker-cu130-torch:  ## Build the CUDA 13.0 + torch base image (multi-GB download)
	docker build -t $(CU130_TORCH_IMAGE) docker/base-cu130-torch

docker-lock:  ## Re-lock the hashed torch requirements for the CUDA image
	$(UV) pip compile docker/base-cu130-torch/requirements.in \
		--index-url https://download.pytorch.org/whl/cu130 \
		--python-version 3.12 --python-platform x86_64-manylinux_2_28 \
		--generate-hashes --no-header --custom-compile-command "make docker-lock" \
		-o docker/base-cu130-torch/requirements.txt

docker-smoke:  ## Smoke-test built base images without a GPU (opt-in)
	$(UV) run --locked pytest -m docker --run-docker tests/test_docker_images.py

gpu-preflight:  ## Check host driver, Docker and NVIDIA Container Toolkit
	scripts/gpu_preflight.sh

gpu-smoke:  ## Check torch sees the GPU inside the CUDA base image (opt-in)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_docker_images.py
