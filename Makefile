# Local developer commands. There is no CI; run the targets relevant to your change.
.PHONY: help setup lint format test check docker-cpu docker-cu130-torch docker-lock \
        docker-smoke gpu-preflight gpu-smoke docker-tesseract tesseract-best tesseract-smoke \
        docker-cu130-paddle docker-paddleocr paddleocr-smoke docker-kraken kraken-smoke \
        docker-lightonocr lightonocr-model lightonocr-smoke docker-fireredocr fireredocr-model \
        fireredocr-smoke docker-ovisocr2 ovisocr2-model ovisocr2-smoke docker-infinityparser2 \
        infinityparser2-model infinityparser2-smoke docker-pero pero-bundles pero-smoke \
        docker-ppstructurev3 ppstructurev3-smoke docker-paddleocr-vl paddleocr-vl-model \
        paddleocr-vl-smoke docker-ocrd-calamari ocrd-calamari-smoke \
        docker-surya surya-model surya-smoke docker-doctr doctr-model doctr-smoke \
        docker-glm-ocr glm-ocr-model glm-ocr-smoke docker-mineru25-pro mineru25-pro-model \
        mineru25-pro-smoke docker-chandra-ocr2 chandra-ocr2-model chandra-ocr2-smoke \
        docker-olmocr2 olmocr2-model olmocr2-smoke docker-deepseek-ocr2 deepseek-ocr2-model \
        deepseek-ocr2-smoke docker-hunyuanocr hunyuanocr-model hunyuanocr-smoke \
        docker-dots-mocr dots-mocr-model dots-mocr-smoke docker-cu130-vllm docker-surya2 \
        surya2-model surya2-smoke

UV ?= uv
BASE_VERSION ?= 1
CPU_IMAGE ?= ocrbench/base-cpu:py3.12-v$(BASE_VERSION)
CU130_TORCH_IMAGE ?= ocrbench/base-cu130-torch:torch2.14.1-py3.12-v$(BASE_VERSION)
CU130_PADDLE_IMAGE ?= ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v$(BASE_VERSION)
CU130_VLLM_IMAGE ?= ocrbench/base-cu130-vllm:vllm0.20.1-py3.12-v$(BASE_VERSION)
TESSERACT_IMAGE := ocrbench/engine-tesseract:5.3.0-v2
PADDLEOCR_IMAGE := ocrbench/engine-paddleocr:3.7.0-v2
PPSTRUCTUREV3_IMAGE := ocrbench/engine-ppstructurev3:3.7.0-v1
PADDLEOCR_VL_IMAGE := ocrbench/engine-paddleocr-vl:1.6-v1
KRAKEN_IMAGE := ocrbench/engine-kraken:7.1.1-v1
LIGHTONOCR_IMAGE := ocrbench/engine-lightonocr:2-1B-v1
FIREREDOCR_IMAGE := ocrbench/engine-fireredocr:2B-v1
OVISOCR2_IMAGE := ocrbench/engine-ovisocr2:0.8B-v1
INFINITYPARSER2_IMAGE := ocrbench/engine-infinityparser2:flash-v1
PERO_IMAGE := ocrbench/engine-pero:0.7.0-81b496e-v1
OCRD_CALAMARI_IMAGE := ocrbench/engine-ocrd-calamari:2.1.0-v1
SURYA_IMAGE := ocrbench/engine-surya:0.17.1-v1
DOCTR_IMAGE := ocrbench/engine-doctr:1.1.0-v1
GLM_OCR_IMAGE := ocrbench/engine-glm-ocr:0.9B-v1
MINERU25_PRO_IMAGE := ocrbench/engine-mineru25-pro:2605-1.2B-v1
CHANDRA_OCR2_IMAGE := ocrbench/engine-chandra-ocr2:2-v1
OLMOCR2_IMAGE := ocrbench/engine-olmocr2:7B-1025-v1
DEEPSEEK_OCR2_IMAGE := ocrbench/engine-deepseek-ocr2:2-v1
HUNYUANOCR_IMAGE := ocrbench/engine-hunyuanocr:1.5-v1
DOTS_MOCR_IMAGE := ocrbench/engine-dots-mocr:3B-v1
SURYA2_IMAGE := ocrbench/engine-surya2:0.22.1-v1
# Host directory with the user-supplied PERO OCR model bundles (engines/pero/README.md).
PERO_BUNDLES ?= dev_data/pero_ocr_models
# Host directory mounted as /cache for engines whose weights are not in the image.
OCRBENCH_CACHE ?= $(HOME)/.cache/ocrbench
LOCK_FLAGS := --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header \
	--custom-compile-command "make docker-lock"
LOCK := $(UV) pip compile --python-version 3.12 $(LOCK_FLAGS)
# The standalone OCR-D/Calamari image runs Python 3.11 (engines/ocrd-calamari/README.md).
LOCK_PY311 := $(UV) pip compile --python-version 3.11 $(LOCK_FLAGS)
export CPU_IMAGE CU130_TORCH_IMAGE CU130_PADDLE_IMAGE CU130_VLLM_IMAGE OCRBENCH_CACHE

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

docker-cu130-vllm:  ## Build the CUDA 13.0 + vLLM 0.20.1 base image (multi-GB download)
	docker build -t $(CU130_VLLM_IMAGE) docker/base-cu130-vllm
	@docker image inspect --format '$(CU130_VLLM_IMAGE) {{.Id}}' $(CU130_VLLM_IMAGE)

docker-lock:  ## Re-lock the hashed requirements of the CUDA bases and the GPU engines
	$(LOCK) docker/base-cu130-torch/requirements.in \
		--index-url https://download.pytorch.org/whl/cu130 \
		-o docker/base-cu130-torch/requirements.txt
	$(LOCK) docker/base-cu130-paddle/requirements.in -o docker/base-cu130-paddle/requirements.txt
	$(LOCK) docker/base-cu130-vllm/requirements.in --index-url https://pypi.org/simple \
		-o docker/base-cu130-vllm/requirements.txt
	$(LOCK) engines/surya2/requirements.in -c docker/base-cu130-vllm/requirements.txt \
		--override engines/surya2/overrides.txt --index-url https://pypi.org/simple \
		-o engines/surya2/requirements.txt
	$(LOCK) engines/paddleocr/requirements.in -c docker/base-cu130-paddle/requirements.txt \
		-o engines/paddleocr/requirements.txt
	$(LOCK) engines/ppstructurev3/requirements.in -c docker/base-cu130-paddle/requirements.txt \
		-o engines/ppstructurev3/requirements.txt
	$(LOCK) engines/paddleocr-vl/requirements.in -c docker/base-cu130-paddle/requirements.txt \
		-o engines/paddleocr-vl/requirements.txt
	$(LOCK) engines/kraken/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--override engines/kraken/overrides.txt --index-url https://pypi.org/simple \
		--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
		-o engines/kraken/requirements.txt
	$(LOCK) engines/lightonocr/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/lightonocr/requirements.txt
	$(LOCK) engines/fireredocr/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/fireredocr/requirements.txt
	$(LOCK) engines/ovisocr2/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/ovisocr2/requirements.txt
	$(LOCK) engines/glm-ocr/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/glm-ocr/requirements.txt
	$(LOCK) engines/mineru25-pro/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/mineru25-pro/requirements.txt
	$(LOCK) engines/chandra-ocr2/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/chandra-ocr2/requirements.txt
	$(LOCK) engines/olmocr2/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/olmocr2/requirements.txt
	$(LOCK) engines/deepseek-ocr2/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/deepseek-ocr2/requirements.txt
	$(LOCK) engines/hunyuanocr/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/hunyuanocr/requirements.txt
	$(LOCK) engines/dots-mocr/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/dots-mocr/requirements.txt
	$(LOCK) engines/infinityparser2/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/infinityparser2/requirements.txt
	$(LOCK) engines/pero/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple -o engines/pero/requirements.txt
	$(LOCK) engines/surya/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu130 \
		--index-strategy unsafe-best-match -o engines/surya/requirements.txt
	$(LOCK) engines/doctr/requirements.in -c docker/base-cu130-torch/requirements.txt \
		--override engines/doctr/overrides.txt --index-url https://pypi.org/simple \
		--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match \
		-o engines/doctr/requirements.txt
	$(LOCK_PY311) engines/ocrd-calamari/requirements.in -o engines/ocrd-calamari/requirements.txt

docker-smoke:  ## Smoke-test built bases, a derived CPU image and the fake-engine runner (opt-in)
	$(UV) run --locked pytest -m docker --run-docker tests/test_docker_images.py tests/test_runner.py

docker-tesseract:  ## Build the Tesseract engine image (needs the CPU base) and print its image ID
	docker build -t $(TESSERACT_IMAGE) engines/tesseract
	@docker image inspect --format '$(TESSERACT_IMAGE) {{.Id}}' $(TESSERACT_IMAGE)

tesseract-best:  ## Download the pinned tessdata_best models (48 MB) for the best presets into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/tesseract/fetch_tessdata.py $(OCRBENCH_CACHE)

tesseract-smoke:  ## Build the Tesseract image and run, evaluate and report a synthetic dataset (opt-in; best presets need tesseract-best)
	$(UV) run --locked pytest -m docker --run-docker tests/test_tesseract.py

docker-paddleocr:  ## Build the PaddleOCR engine image (needs the Paddle base) and print its image ID
	docker build -t $(PADDLEOCR_IMAGE) engines/paddleocr
	@docker image inspect --format '$(PADDLEOCR_IMAGE) {{.Id}}' $(PADDLEOCR_IMAGE)

paddleocr-smoke:  ## Build the PaddleOCR image and run, evaluate and report a synthetic dataset on the GPU (opt-in)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_paddleocr.py

docker-ppstructurev3:  ## Build the PP-StructureV3 engine image (needs the Paddle base) and print its image ID
	docker build -t $(PPSTRUCTUREV3_IMAGE) engines/ppstructurev3
	@docker image inspect --format '$(PPSTRUCTUREV3_IMAGE) {{.Id}}' $(PPSTRUCTUREV3_IMAGE)

ppstructurev3-smoke:  ## Build the PP-StructureV3 image and run, evaluate and report a synthetic dataset on the GPU (opt-in)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_ppstructurev3.py

docker-paddleocr-vl:  ## Build the PaddleOCR-VL-1.6 engine image (needs the Paddle base) and print its image ID
	docker build -t $(PADDLEOCR_VL_IMAGE) engines/paddleocr-vl
	@docker image inspect --format '$(PADDLEOCR_VL_IMAGE) {{.Id}}' $(PADDLEOCR_VL_IMAGE)

paddleocr-vl-model:  ## Download the pinned PaddleOCR-VL-1.6 and PP-DocLayoutV3 models (2 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/paddleocr-vl/fetch_model.py $(OCRBENCH_CACHE)

paddleocr-vl-smoke:  ## Build the PaddleOCR-VL-1.6 image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs paddleocr-vl-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_paddleocr_vl.py

docker-ocrd-calamari:  ## Build the standalone OCR-D/Calamari CPU engine image, with its models, and print its image ID
	docker build -t $(OCRD_CALAMARI_IMAGE) engines/ocrd-calamari
	@docker image inspect --format '$(OCRD_CALAMARI_IMAGE) {{.Id}}' $(OCRD_CALAMARI_IMAGE)

ocrd-calamari-smoke:  ## Build the OCR-D/Calamari image and run, evaluate and report a synthetic dataset on the CPU (opt-in)
	$(UV) run --locked pytest -m docker --run-docker tests/test_ocrd_calamari.py

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

docker-fireredocr:  ## Build the FireRed-OCR engine image (needs the torch base) and print its image ID
	docker build -t $(FIREREDOCR_IMAGE) engines/fireredocr
	@docker image inspect --format '$(FIREREDOCR_IMAGE) {{.Id}}' $(FIREREDOCR_IMAGE)

fireredocr-model:  ## Download the pinned FireRed-OCR weights (4.3 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/fireredocr/fetch_model.py $(OCRBENCH_CACHE)

fireredocr-smoke:  ## Build the FireRed-OCR image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs fireredocr-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_fireredocr.py

docker-ovisocr2:  ## Build the OvisOCR2 engine image (needs the torch base) and print its image ID
	docker build -t $(OVISOCR2_IMAGE) engines/ovisocr2
	@docker image inspect --format '$(OVISOCR2_IMAGE) {{.Id}}' $(OVISOCR2_IMAGE)

ovisocr2-model:  ## Download the pinned OvisOCR2 weights (1.7 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/ovisocr2/fetch_model.py $(OCRBENCH_CACHE)

ovisocr2-smoke:  ## Build the OvisOCR2 image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs ovisocr2-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_ovisocr2.py

docker-glm-ocr:  ## Build the GLM-OCR engine image (needs the torch base) and print its image ID
	docker build -t $(GLM_OCR_IMAGE) engines/glm-ocr
	@docker image inspect --format '$(GLM_OCR_IMAGE) {{.Id}}' $(GLM_OCR_IMAGE)

glm-ocr-model:  ## Download the pinned GLM-OCR weights (2.7 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/glm-ocr/fetch_model.py $(OCRBENCH_CACHE)

glm-ocr-smoke:  ## Build the GLM-OCR image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs glm-ocr-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_glm_ocr.py

docker-mineru25-pro:  ## Build the MinerU2.5-Pro engine image (needs the torch base) and print its image ID
	docker build -t $(MINERU25_PRO_IMAGE) engines/mineru25-pro
	@docker image inspect --format '$(MINERU25_PRO_IMAGE) {{.Id}}' $(MINERU25_PRO_IMAGE)

mineru25-pro-model:  ## Download the pinned MinerU2.5-Pro weights (2.3 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/mineru25-pro/fetch_model.py $(OCRBENCH_CACHE)

mineru25-pro-smoke:  ## Build the MinerU2.5-Pro image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs mineru25-pro-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_mineru25_pro.py

docker-chandra-ocr2:  ## Build the Chandra OCR 2 engine image (needs the torch base) and print its image ID
	docker build -t $(CHANDRA_OCR2_IMAGE) engines/chandra-ocr2
	@docker image inspect --format '$(CHANDRA_OCR2_IMAGE) {{.Id}}' $(CHANDRA_OCR2_IMAGE)

chandra-ocr2-model:  ## Download the pinned Chandra OCR 2 weights (10.6 GB, modified OpenRAIL-M license) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/chandra-ocr2/fetch_model.py $(OCRBENCH_CACHE)

chandra-ocr2-smoke:  ## Build the Chandra OCR 2 image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs chandra-ocr2-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_chandra_ocr2.py

docker-olmocr2:  ## Build the olmOCR-2 engine image (needs the torch base) and print its image ID
	docker build -t $(OLMOCR2_IMAGE) engines/olmocr2
	@docker image inspect --format '$(OLMOCR2_IMAGE) {{.Id}}' $(OLMOCR2_IMAGE)

olmocr2-model:  ## Download the pinned olmOCR-2 weights (16.6 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/olmocr2/fetch_model.py $(OCRBENCH_CACHE)

olmocr2-smoke:  ## Build the olmOCR-2 image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs olmocr2-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_olmocr2.py

docker-deepseek-ocr2:  ## Build the DeepSeek-OCR-2 engine image (needs the torch base) and print its image ID
	docker build -t $(DEEPSEEK_OCR2_IMAGE) engines/deepseek-ocr2
	@docker image inspect --format '$(DEEPSEEK_OCR2_IMAGE) {{.Id}}' $(DEEPSEEK_OCR2_IMAGE)

deepseek-ocr2-model:  ## Download the pinned DeepSeek-OCR-2 weights and code (6.8 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/deepseek-ocr2/fetch_model.py $(OCRBENCH_CACHE)

deepseek-ocr2-smoke:  ## Build the DeepSeek-OCR-2 image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs deepseek-ocr2-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_deepseek_ocr2.py

docker-hunyuanocr:  ## Build the HunyuanOCR-1.5 engine image (needs the torch base) and print its image ID
	docker build -t $(HUNYUANOCR_IMAGE) engines/hunyuanocr
	@docker image inspect --format '$(HUNYUANOCR_IMAGE) {{.Id}}' $(HUNYUANOCR_IMAGE)

hunyuanocr-model:  ## Download the pinned HunyuanOCR-1.5 weights (2.2 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/hunyuanocr/fetch_model.py $(OCRBENCH_CACHE)

hunyuanocr-smoke:  ## Build the HunyuanOCR-1.5 image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs hunyuanocr-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_hunyuanocr.py

docker-dots-mocr:  ## Build the dots.mocr engine image (needs the torch base) and print its image ID
	docker build -t $(DOTS_MOCR_IMAGE) engines/dots-mocr
	@docker image inspect --format '$(DOTS_MOCR_IMAGE) {{.Id}}' $(DOTS_MOCR_IMAGE)

dots-mocr-model:  ## Download the pinned dots.mocr weights and code (6.1 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/dots-mocr/fetch_model.py $(OCRBENCH_CACHE)

dots-mocr-smoke:  ## Build the dots.mocr image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs dots-mocr-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_dots_mocr.py

docker-surya2:  ## Build the Surya 2 engine image (needs the vLLM base) and print its image ID
	docker build -t $(SURYA2_IMAGE) engines/surya2
	@docker image inspect --format '$(SURYA2_IMAGE) {{.Id}}' $(SURYA2_IMAGE)

surya2-model:  ## Download the pinned surya-ocr-2 weights (1.4 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/surya2/fetch_model.py $(OCRBENCH_CACHE)

surya2-smoke:  ## Build the Surya 2 image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs surya2-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_surya2.py

docker-infinityparser2:  ## Build the Infinity-Parser2-Flash engine image (needs the torch base) and print its image ID
	docker build -t $(INFINITYPARSER2_IMAGE) engines/infinityparser2
	@docker image inspect --format '$(INFINITYPARSER2_IMAGE) {{.Id}}' $(INFINITYPARSER2_IMAGE)

infinityparser2-model:  ## Download the pinned Infinity-Parser2-Flash weights (4.4 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/infinityparser2/fetch_model.py $(OCRBENCH_CACHE)

infinityparser2-smoke:  ## Build the Infinity-Parser2-Flash image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs infinityparser2-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_infinityparser2.py

docker-pero:  ## Build the PERO OCR engine image (needs the torch base) and print its image ID
	docker build -t $(PERO_IMAGE) engines/pero
	@docker image inspect --format '$(PERO_IMAGE) {{.Id}}' $(PERO_IMAGE)

pero-bundles:  ## Copy the user-supplied PERO OCR bundles from $(PERO_BUNDLES) into $(OCRBENCH_CACHE), checked against bundles.sha256 (opt-in)
	python3 engines/pero/install_bundles.py $(PERO_BUNDLES) $(OCRBENCH_CACHE)

pero-smoke:  ## Build the PERO OCR image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs pero-bundles)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_pero.py

docker-surya:  ## Build the Surya OCR engine image (needs the torch base) and print its image ID
	docker build -t $(SURYA_IMAGE) engines/surya
	@docker image inspect --format '$(SURYA_IMAGE) {{.Id}}' $(SURYA_IMAGE)

surya-model:  ## Download the pinned Surya checkpoints (3 GB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/surya/fetch_model.py $(OCRBENCH_CACHE)

surya-smoke:  ## Build the Surya image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs surya-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_surya.py

docker-doctr:  ## Build the docTR engine image (needs the torch base) and print its image ID
	docker build -t $(DOCTR_IMAGE) engines/doctr
	@docker image inspect --format '$(DOCTR_IMAGE) {{.Id}}' $(DOCTR_IMAGE)

doctr-model:  ## Download the pinned docTR checkpoints (129 MB) into $(OCRBENCH_CACHE) (opt-in)
	python3 engines/doctr/fetch_model.py $(OCRBENCH_CACHE)

doctr-smoke:  ## Build the docTR image and run, evaluate and report a synthetic dataset on the GPU (opt-in; needs doctr-model)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_doctr.py

gpu-preflight:  ## Check host driver, Docker and NVIDIA Container Toolkit
	scripts/gpu_preflight.sh

gpu-smoke:  ## Check torch sees the GPU in the CUDA base and the runner passes it (opt-in)
	$(UV) run --locked pytest -m gpu --run-docker --run-gpu tests/test_docker_images.py tests/test_runner.py
