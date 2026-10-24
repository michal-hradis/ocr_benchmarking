# syntax=docker/dockerfile:1
# DeepSeek-OCR-2 page-image to Markdown engine adapter (engines/deepseek-ocr2/README.md) on the
# shared torch base.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# transformers 4.46.3, tokenizers 0.20.3 and the imports of the model's remote code, hash-locked
# against the base (make docker-lock). torchvision comes from the PyTorch CUDA 13.0 index,
# everything else from PyPI.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/engine-requirements.txt

# No weights in the image: the 6.8 GB checkpoint and its code are read from the /cache volume and
# checked against these checksums (make deepseek-ocr2-model downloads them).
COPY models.sha256 /opt/ocrbench/models.sha256
COPY deepseek_ocr2_engine.py /opt/ocrbench/deepseek_ocr2_engine.py

# Offline: model files are read from local directories only. transformers copies the remote code
# into HF_MODULES_CACHE to import it; that copy stays in the container's /tmp, not in /cache.
# torch 2.14 sends some ops (here the rotary embedding's batched outer product) to Triton kernels
# that are compiled at first use, which needs a C compiler the image does not have;
# TORCH_DISABLE_NATIVE_JIT keeps them on the standard ATen kernels.
ENV HF_HUB_OFFLINE=1 \
    TORCH_DISABLE_NATIVE_JIT=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_MODULES_CACHE=/tmp/hf_modules

LABEL org.opencontainers.image.title="ocrbench-engine-deepseek-ocr2" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="deepseek-ocr2"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/deepseek_ocr2_engine.py"]
