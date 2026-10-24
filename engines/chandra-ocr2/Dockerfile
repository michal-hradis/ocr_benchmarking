# syntax=docker/dockerfile:1
# Chandra OCR 2 page-image OCR adapter (engines/chandra-ocr2/README.md) on the shared torch base.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# transformers 5.19.0 and its dependencies, hash-locked against the base (make docker-lock).
# torchvision (for the fast Qwen2-VL image processor) comes from the PyTorch CUDA 13.0 index,
# everything else from PyPI.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/engine-requirements.txt

# No weights in the image: the 10.6 GB checkpoint is read from the /cache volume and checked
# against these checksums (make chandra-ocr2-model downloads it).
COPY models.sha256 /opt/ocrbench/models.sha256
COPY prompt.txt /opt/ocrbench/prompt.txt
COPY chandra_ocr2_engine.py /opt/ocrbench/chandra_ocr2_engine.py

# Offline: model files are read from local directories only.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

LABEL org.opencontainers.image.title="ocrbench-engine-chandra-ocr2" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="chandra-ocr2"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/chandra_ocr2_engine.py"]
