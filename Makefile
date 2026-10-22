# Local developer commands. There is no CI; run the targets relevant to your change.
.PHONY: help setup lint format test check docker-cpu docker-cu130-torch docker-lock \
        docker-smoke gpu-preflight gpu-smoke docker-tesseract tesseract-smoke \
        docker-cu130-paddle docker-paddleocr paddleocr-smoke docker-kraken kraken-smoke \
        docker-lightonocr lightonocr-model lightonocr-smoke

UV ?= uv
BASE_VERSION ?= 1
CPU_IMAGE ?= ocrbench/base-cpu:py3.12-v$(BASE_VERSION)
CU130_TORCH_IMAGE ?= ocrbench/base-cu130-torch:torch2.14.1-py3.12-v$(BASE_VERSION)
CU130_PADDLE_IMAGE ?= ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v$(BASE_VERSION)
TESSERACT_IMAGE := ocrbench/engine-tesseract:5.3.0-v1
PADDLEOCR_IMAGE := ocrbench/engine-paddleocr:3.7.0-v1
KRAKEN_IMAGE := ocrbench/engine-kraken:7.1.1-v1
LIGHTONOCR_IMAGE := ocrbench/engine-lightonocr:2-1B-v1
# Host directory mounted as /cache for engines whose weights are not in the image.
OCRBENCH_CACHE ?= $(HOME)/.cache/ocrbench
LOCK := $(UV) pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28 \
	--generate-hashes --no-header --custom-compile-command "make docker-lock"
export CPU_IMAGE CU130_TORCH_IMAGE CU130_PADDLE_IMAGE OCRBENCH_CACHE

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

docker-cpu:  ## Build the CPU base image and print its image ID
	docker build -t $(CPU_IMAGE) docker/base-cpu
	@docker image inspect --format '$(CPU_IMAGE) {{.Id}}' $(CPU_IMAGE)

docker-cu130-torch:  ## Build the CUDA 13.0 + torch base image (multi-GB download)
	docker build -t $(CU130_TORCH_IMAGE) docker/base-cu130-torch
	@docker image inspect --format '$(CU130_TORCH_IMAGE) {{.Id}}' $(CU130_TORCH_IMAGE)

docker-cu130-paddle:  ## Build the CUDA 13.0 + PaddlePaddle base image (multi-GB download)
	docker build -t $(CU130_PADDLE_IMAGE) docker/base-cu130-paddle
	@docker image inspect --format '$(CU130_PADDLE_IMAGE) {{.Id}}' $(CU130_PADDLE_IMAGE)

docker-lock:  ## Re-lock the hashed requirements of the CUDA bases and the GPU engines
	$(LOCK) docker/base-cu130-torch/requirements.in \
		--index-url https://download.pytorch.org/whl/cu130 \
		-o docker/base-cu130-torch/requirements.txt
	$(LOCK) docker/base-cu130-paddle/requirements.in -o docker/base-cu130-paddle/requirements.txt
	$(LOCK) engines/paddleocr/requirements.in -c docker/base-cu130-paddle/requirements.txt \
		-o engines/paddleocr/requirements.txt
	$(LOCK) engines/kraken/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--override engines/kraken/overrides.txt --index-url https://pypi.org/simple \
		--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
		-o engines/kraken/requirements.txt
	$(LOCK) engines/lightonocr/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/lightonocr/requirements.txt

docker-smoke:  ## Smoke-test built bases, a derived CPU image and the fake-engine runner (opt-in)
	$(UV) run --locked pytest -m docker --run-docker tests/test_docker_images.py tests/test_runner.py

docker-tesseract:  ## Build the Tesseract engine image (needs the CPU base) and print its image ID
	docker build -t $(TESSERACT_IMAGE) engines/tesseract
	@docker image inspect --format '$(TESSERACT_IMAGE) {{.Id}}' $(TESSERACT_IMAGE)

tesseract-smoke:  ## Build the Tesseract image and run, evaluate and report a synthetic dataset (opt-in)
	$(UV) run --locked pytest -m docker --run-docker tests/test_tesseract.py

docker-paddleocr:  ## Build the PaddleOCR engine image (needs the Paddle base) and print its image ID
	docker build -t $(PADDLEOCR_IMAGE) engines/paddleocr
	@docker image inspect --format '$(PADDLEOCR_IMAGE) {{.Id}}' $(PADDLEOCR_IMAGE)

paddleocr-smoke:  ## Build the PaddleOCR image and run, evaluate and report a synthetic dataset on the GPU (opt-in)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_paddleocr.py

docker-kraken:  ## Build the Kraken engine image (needs the torch base) and print its image ID
	docker build -t $(KRAKEN_IMAGE) engines/kraken
	@docker image inspect --format '$(KRAKEN_IMAGE) {{.Id}}' $(KRAKEN_IMAGE)

kraken-smoke:  ## Build the Kraken image and run, evaluate and report a synthetic dataset on the GPU (opt-in)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_kraken.py

docker-lightonocr:  ## Build the LightOnOCR-2 engine image (needs the torch base) and print its image ID
	docker build -t $(LIGHTONOCR_IMAGE) engines/lightonocr
	@docker image inspect --format '$(LIGHTONOCR_IMAGE) {{.Id}}' $(LIGHTONOCR_IMAGE)

lightonocr-model:  ## Download the pinned LightOnOCR-2 weights (2 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/lightonocr/fetch_model.py $(OCRBENCH_CACHE)

lightonocr-smoke:  ## Build the LightOnOCR-2 image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs lightonocr-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_lightonocr.py

gpu-preflight:  ## Check host driver, Docker and NVIDIA Container Toolkit
	scripts/gpu_preflight.sh

gpu-smoke:  ## Check torch sees the GPU in the CUDA base and the runner passes it (opt-in)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_docker_images.py tests/test_runner.py
