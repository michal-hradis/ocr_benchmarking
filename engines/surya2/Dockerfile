# syntax=docker/dockerfile:1
# Surya 2 (surya-ocr-2 served by vLLM) full-page OCR engine adapter (engines/surya2/README.md) on
# the shared vLLM base.
FROM ocrbench/base-cu130-vllm:vllm0.20.1-py3.12-v1

USER root
# Surya 0.22.1 and its dependencies, hash-locked against the base (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt

# No weights in the image: the 1.4 GB checkpoint is read from the /cache volume and checked
# against these checksums (make surya2-model downloads them).
COPY models.sha256 /opt/ocrbench/models.sha256
COPY surya2_engine.py /opt/ocrbench/surya2_engine.py

# Offline. vLLM, Triton, torch.compile and FlashInfer compile kernels at start-up and load them as
# shared objects; the runner's /tmp is mounted noexec, so these caches are kept in the /cache
# volume (README.md), where later runs reuse them.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME=/cache/runtime/surya2 \
    VLLM_CACHE_ROOT=/cache/runtime/surya2/vllm \
    TRITON_CACHE_DIR=/cache/runtime/surya2/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/runtime/surya2/torchinductor \
    FLASHINFER_WORKSPACE_BASE=/cache/runtime/surya2 \
    SURYA_INFERENCE_BACKEND=vllm \
    SURYA_INFERENCE_AUTOSTART=false

LABEL org.opencontainers.image.title="ocrbench-engine-surya2" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="surya2"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/surya2_engine.py"]
