# syntax=docker/dockerfile:1
# Nemotron 3.5 Parse (served by vLLM with NVIDIA's nemotron-parse plugin) page parsing engine
# adapter (engines/nemotron-parse/README.md) on the vLLM 0.26 base.
FROM ocrbench/base-cu130-vllm026:vllm0.26.0-py3.12-v1

USER root
# The nemotron-parse package's dependencies, hash-locked against the base (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt

# The nemotron-parse 3.5.0.post7 package (vLLM plugin, DocumentParser and postprocessing) from the
# source files of the pinned model revision, checked against source.sha256 and installed without
# dependencies (README.md). No weights in the image: the 2.5 GB checkpoint is read from the
# /cache volume and checked against models.sha256 (make nemotron-parse-model downloads it).
COPY models.sha256 source.sha256 fetch_model.py nemotron_parse_engine.py /opt/ocrbench/
RUN python /opt/ocrbench/fetch_model.py --source /opt/nemotron-parse-src \
    && uv pip install --no-deps --no-build-isolation /opt/nemotron-parse-src \
    && python -c "import nemotron_parse.inference, vllm_tied_patch.plugin"

# Offline. vLLM, Triton, torch.compile and FlashInfer compile kernels at start-up and load them as
# shared objects; the runner's /tmp is mounted noexec, so these caches are kept in the /cache
# volume (README.md), where later runs reuse them.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME=/cache/runtime/nemotron-parse \
    VLLM_CACHE_ROOT=/cache/runtime/nemotron-parse/vllm \
    TRITON_CACHE_DIR=/cache/runtime/nemotron-parse/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/runtime/nemotron-parse/torchinductor \
    FLASHINFER_WORKSPACE_BASE=/cache/runtime/nemotron-parse

LABEL org.opencontainers.image.title="ocrbench-engine-nemotron-parse" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="nemotron-parse"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/nemotron_parse_engine.py"]
