# syntax=docker/dockerfile:1
# docTR text detection and recognition engine adapter (engines/doctr/README.md) on the shared
# torch base.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# docTR 1.1.0 and its dependencies, hash-locked against the base (make docker-lock), with the
# headless OpenCV instead of opencv-python (overrides.txt). torchvision comes from the PyTorch
# CUDA 13.0 index, everything else from PyPI.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/engine-requirements.txt

# No weights in the image: the two checkpoints (129 MB) are read from the /cache volume and checked
# against these checksums (make doctr-model downloads them).
COPY models.sha256 /opt/ocrbench/models.sha256
COPY doctr_engine.py /opt/ocrbench/doctr_engine.py

# Offline: docTR loads the checkpoints from DOCTR_CACHE_DIR (set by the adapter) and downloads
# nothing once they are there.
ENV HF_HUB_OFFLINE=1

LABEL org.opencontainers.image.title="ocrbench-engine-doctr" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="doctr"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/doctr_engine.py"]
