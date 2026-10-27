# syntax=docker/dockerfile:1
# Unlimited-OCR (served by vLLM's built-in UnlimitedOCRForCausalLM) page parsing engine adapter
# (engines/unlimited-ocr/README.md) on the vLLM 0.26 base.
FROM ocrbench/base-cu130-vllm026:vllm0.26.0-py3.12-v1

USER root
# No packages beyond the base. No weights in the image: the 6.7 GB checkpoint is read from the
# /cache volume and checked against models.sha256 (make unlimited-ocr-model downloads it).
COPY models.sha256 fetch_model.py unlimited_ocr_engine.py /opt/ocrbench/
RUN python -c "import vllm.model_executor.models.unlimited_ocr"

# Offline. vLLM, Triton and torch.compile compile kernels at start-up and load them as shared
# objects; the runner's /tmp is mounted noexec, so these caches are kept in the /cache volume
# (README.md), where later runs reuse them.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME=/cache/runtime/unlimited-ocr \
    VLLM_CACHE_ROOT=/cache/runtime/unlimited-ocr/vllm \
    TRITON_CACHE_DIR=/cache/runtime/unlimited-ocr/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/runtime/unlimited-ocr/torchinductor \
    FLASHINFER_WORKSPACE_BASE=/cache/runtime/unlimited-ocr

LABEL org.opencontainers.image.title="ocrbench-engine-unlimited-ocr" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="unlimited-ocr"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/unlimited_ocr_engine.py"]
