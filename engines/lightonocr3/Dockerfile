# syntax=docker/dockerfile:1
# LightOnOCR-3 (0.8B, 1B and 4B, plain and grounding, served by vLLM's built-in models) engine
# adapter (engines/lightonocr3/README.md) on the vLLM 0.30 base.
FROM ocrbench/base-cu130-vllm030:vllm0.30.0-py3.12-v1

USER root
# No packages beyond the base. No weights in the image: the checkpoints are read from the /cache
# volume and checked against models-<checkpoint>.sha256 (make lightonocr3-model downloads them).
COPY models-0.8B.sha256 models-1B.sha256 models-4B.sha256 fetch_model.py lightonocr3_engine.py \
     /opt/ocrbench/
RUN python -c "import vllm.model_executor.models.qwen3_5, vllm.model_executor.models.lightonocr"

# Offline. vLLM, Triton and torch.compile compile kernels at start-up and load them as shared
# objects; the runner's /tmp is mounted noexec, so these caches are kept in the /cache volume
# (README.md), where later runs reuse them.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME=/cache/runtime/lightonocr3 \
    VLLM_CACHE_ROOT=/cache/runtime/lightonocr3/vllm \
    TRITON_CACHE_DIR=/cache/runtime/lightonocr3/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/runtime/lightonocr3/torchinductor \
    FLASHINFER_WORKSPACE_BASE=/cache/runtime/lightonocr3

LABEL org.opencontainers.image.title="ocrbench-engine-lightonocr3" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="lightonocr3"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/lightonocr3_engine.py"]
