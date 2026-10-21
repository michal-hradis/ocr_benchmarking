# syntax=docker/dockerfile:1
# PaddleOCR PP-OCRv6 engine adapter (engines/paddleocr/README.md) on the shared Paddle base.
FROM ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1

USER root
# OpenCV (opencv-contrib-python, required by PaddleX) loads libGL and GLib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1=1.7.0-1build1 libglib2.0-0t64=2.80.0-6ubuntu3.9 \
    && rm -rf /var/lib/apt/lists/*

# PaddleOCR 3.7.0 and its dependencies, hash-locked against the base (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt

# PP-OCRv6_medium detection and recognition models from Hugging Face at pinned commits;
# the build fails unless every file matches models.sha256.
ARG DET_REVISION=8e0f56fb2ef86b461d99cfc7ac5c137738985f61
ARG REC_REVISION=e5a92bcbc5cc1b494628e458d267778f0704fd7c
COPY models.sha256 /opt/ocrbench/models.sha256
RUN cd /opt/ocrbench \
    && python -c 'import sys, urllib.request as u; from pathlib import Path; \
[(Path(f"models/{m}").mkdir(parents=True, exist_ok=True), \
  u.urlretrieve(f"https://huggingface.co/PaddlePaddle/{m}/resolve/{r}/{f}", f"models/{m}/{f}")) \
 for m, r in (("PP-OCRv6_medium_det", sys.argv[1]), ("PP-OCRv6_medium_rec", sys.argv[2])) \
 for f in ("inference.json", "inference.pdiparams", "inference.yml")]' \
        "$DET_REVISION" "$REC_REVISION" \
    && cd models && sha256sum --check --strict /opt/ocrbench/models.sha256
COPY paddleocr_engine.py /opt/ocrbench/paddleocr_engine.py

# Offline: no model-host connectivity check, no silent fallback from GPU to CPU, and the
# PaddleX cache in the writable /tmp (the runner mounts the root filesystem read-only).
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_DISABLE_DEVICE_FALLBACK=True \
    PADDLE_PDX_CACHE_HOME=/tmp/paddlex \
    OCRBENCH_PADDLEOCR_MODELS=/opt/ocrbench/models

LABEL org.opencontainers.image.title="ocrbench-engine-paddleocr" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="paddleocr"

# The runner uses the host UID, which cannot enter /home/ocr; PaddleX stats relative paths.
WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/paddleocr_engine.py"]
