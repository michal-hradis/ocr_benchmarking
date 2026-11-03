# syntax=docker/dockerfile:1
# Nanonets-OCR2-3B (Qwen2.5-VL, served by vLLM's built-in model) page-image to Markdown engine
# adapter (engines/nanonets-ocr2/README.md) on the vLLM 0.30 base.
FROM ocrbench/base-cu130-vllm030:vllm0.30.0-py3.12-v1

USER root
# No packages beyond the base. No weights in the image: the 7.5 GB checkpoint is read from the
# /cache volume and checked against models.sha256 (make nanonets-ocr2-model downloads it).
COPY models.sha256 prompt.txt fetch_model.py nanonets_ocr2_engine.py /opt/ocrbench/
RUN python -c "import vllm.model_executor.models.qwen2_5_vl"

# Offline. vLLM, Triton and torch.compile compile kernels at start-up and load them as shared
# objects; the runner's /tmp is mounted noexec, so these caches are kept in the /cache volume
# (README.md), where later runs reuse them.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME=/cache/runtime/nanonets-ocr2 \
    VLLM_CACHE_ROOT=/cache/runtime/nanonets-ocr2/vllm \
    TRITON_CACHE_DIR=/cache/runtime/nanonets-ocr2/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/runtime/nanonets-ocr2/torchinductor \
    FLASHINFER_WORKSPACE_BASE=/cache/runtime/nanonets-ocr2

LABEL org.opencontainers.image.title="ocrbench-engine-nanonets-ocr2" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="nanonets-ocr2"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/nanonets_ocr2_engine.py"]
