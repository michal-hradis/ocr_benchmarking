# syntax=docker/dockerfile:1
# Surya OCR text line detection, recognition and layout/reading order engine adapter
# (engines/surya/README.md) on the shared torch base.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# Surya 0.17.1 with transformers 4.57.6 and their dependencies, hash-locked against the base
# (make docker-lock). torchvision comes from the PyTorch CUDA 13.0 index, everything else from
# PyPI.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/engine-requirements.txt

# No weights in the image: the 3 GB of checkpoints are read from the /cache volume and checked
# against these checksums (make surya-model downloads them).
COPY models.sha256 /opt/ocrbench/models.sha256
COPY surya_engine.py /opt/ocrbench/surya_engine.py

# Offline: the checkpoints are local directories; Surya's download cache is never used.
# torch 2.14 sends some ops (here the rotary embedding's batched outer product) to Triton kernels
# that are compiled at first use, which needs a C compiler the image does not have;
# TORCH_DISABLE_NATIVE_JIT keeps them on the standard ATen kernels.
ENV HF_HUB_OFFLINE=1 \
    TORCH_DISABLE_NATIVE_JIT=1 \
    TRANSFORMERS_OFFLINE=1 \
    MODEL_CACHE_DIR=/tmp/surya-models

LABEL org.opencontainers.image.title="ocrbench-engine-surya" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="surya"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/surya_engine.py"]
