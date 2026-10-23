# syntax=docker/dockerfile:1
# PaddleOCR-VL-1.6 document parsing pipeline adapter (engines/paddleocr-vl/README.md) on the shared
# Paddle base.
FROM ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1

USER root
# OpenCV (opencv-contrib-python, required by PaddleX) loads libGL and GLib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1=1.7.0-1build1 libglib2.0-0t64=2.80.0-6ubuntu3.9 \
    && rm -rf /var/lib/apt/lists/*

# PaddleOCR 3.7.0 with the PaddleX "ocr" extra, hash-locked against the base (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt

# No models in the image: the 2 GB of checkpoints are read from the /cache volume and checked
# against these checksums (make paddleocr-vl-model downloads them).
COPY models.sha256 /opt/ocrbench/models.sha256
COPY paddleocr_vl_engine.py /opt/ocrbench/paddleocr_vl_engine.py

# Offline: no model-host connectivity check, no silent fallback from GPU to CPU, and the
# PaddleX cache in the writable /tmp (the runner mounts the root filesystem read-only).
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_DISABLE_DEVICE_FALLBACK=True \
    PADDLE_PDX_CACHE_HOME=/tmp/paddlex \
    HF_HUB_OFFLINE=1

LABEL org.opencontainers.image.title="ocrbench-engine-paddleocr-vl" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="paddleocr-vl"

# The runner uses the host UID, which cannot enter /home/ocr; PaddleX stats relative paths.
WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/paddleocr_vl_engine.py"]
