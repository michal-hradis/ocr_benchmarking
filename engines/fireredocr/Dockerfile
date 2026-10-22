# syntax=docker/dockerfile:1
# FireRed-OCR end-to-end image-to-Markdown engine adapter (engines/fireredocr/README.md) on the
# shared torch base.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# transformers 5.19.0 and its dependencies, hash-locked against the base (make docker-lock).
# torchvision (for the fast Qwen2-VL image processor) comes from the PyTorch CUDA 13.0 index,
# everything else from PyPI.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/engine-requirements.txt

# No weights in the image: the 4.3 GB checkpoint is read from the /cache volume and checked
# against these checksums (make fireredocr-model downloads it).
COPY models.sha256 /opt/ocrbench/models.sha256
COPY prompt.txt /opt/ocrbench/prompt.txt
COPY fireredocr_engine.py /opt/ocrbench/fireredocr_engine.py

# Offline: model files are read from local directories only.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

LABEL org.opencontainers.image.title="ocrbench-engine-fireredocr" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="fireredocr"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/fireredocr_engine.py"]
