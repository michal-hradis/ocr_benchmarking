# syntax=docker/dockerfile:1
# MonkeyOCRv2 parsing (upstream's two-stage pipeline, served by vLLM with upstream's model plugin)
# engine adapter (engines/monkeyocrv2/README.md) on the vLLM 0.26 base.
FROM ocrbench/base-cu130-vllm026:vllm0.26.0-py3.12-v1

USER root
# No packages beyond the base: upstream's serving and pipeline code needs vLLM, torch, OpenCV,
# requests and Pillow, which the base has. Upstream's source files at the pinned commit (the vLLM
# model plugin, serve.py and core_runner.py) are downloaded and checked against source.sha256.
# No weights in the image: the checkpoints are read from the /cache volume and checked against
# models-<checkpoint>.sha256 (make monkeyocrv2-model downloads them).
COPY models-B-Parsing.sha256 models-B-Parsing-DFlash.sha256 source.sha256 fetch_model.py \
     monkeyocrv2_engine.py /opt/ocrbench/
RUN cd /opt/ocrbench && python fetch_model.py --source /opt/monkeyocrv2 \
    && cd /opt/monkeyocrv2/parsing \
    && python -c "import core_runner, modeling.modeling_monkeyocrv2_vllm, vllm.v1.spec_decode.dflash"

# Offline. vLLM, Triton and torch.compile compile kernels at start-up and load them as shared
# objects; the runner's /tmp is mounted noexec, so these caches are kept in the /cache volume
# (README.md), where later runs reuse them.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME=/cache/runtime/monkeyocrv2 \
    VLLM_CACHE_ROOT=/cache/runtime/monkeyocrv2/vllm \
    TRITON_CACHE_DIR=/cache/runtime/monkeyocrv2/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/runtime/monkeyocrv2/torchinductor \
    FLASHINFER_WORKSPACE_BASE=/cache/runtime/monkeyocrv2 \
    HF_MODULES_CACHE=/cache/runtime/monkeyocrv2/hf_modules

LABEL org.opencontainers.image.title="ocrbench-engine-monkeyocrv2" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="monkeyocrv2"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/monkeyocrv2_engine.py"]
