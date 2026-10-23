# syntax=docker/dockerfile:1
# PaddleOCR PP-StructureV3 layout pipeline adapter (engines/ppstructurev3/README.md) on the shared Paddle base.
FROM ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1

USER root
# OpenCV (opencv-contrib-python, required by PaddleX) loads libGL and GLib.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1=1.7.0-1build1 libglib2.0-0t64=2.80.0-6ubuntu3.9 \
    && rm -rf /var/lib/apt/lists/*

# PaddleOCR 3.7.0 with the PaddleX "ocr" extra that PP-StructureV3 requires, hash-locked against
# the base (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt

# The layout, region and text line orientation models of the PP-StructureV3 defaults and the text
# detection and recognition models of both OCR model sets (README.md) from Hugging Face at pinned
# commits; the build fails unless every file matches models.sha256. Table, formula, seal and chart
# models are not included.
COPY models.sha256 /opt/ocrbench/models.sha256
RUN cd /opt/ocrbench \
    && python -c 'import sys, urllib.request as u; from pathlib import Path; \
[(Path(f"models/{m}").mkdir(parents=True, exist_ok=True), \
  u.urlretrieve(f"https://huggingface.co/PaddlePaddle/{m}/resolve/{r}/{f}", f"models/{m}/{f}")) \
 for m, r in (a.split("@") for a in sys.argv[1:]) \
 for f in ("inference.json", "inference.pdiparams", "inference.yml")]' \
        PP-DocLayout_plus-L@aa52b8528c84f9b1a34ac3a88fe0e576edb9d11d \
        PP-DocBlockLayout@f270657da59d956ab69ffc2c4722fe1f557bf17b \
        PP-OCRv5_server_det@ca867c897ecbca8873081573a802ad70d499cb94 \
        PP-LCNet_x1_0_textline_ori@cd237a44b0e359d4fe38310a416203cf7403faa5 \
        latin_PP-OCRv5_mobile_rec@ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57 \
        PP-OCRv6_medium_det@8e0f56fb2ef86b461d99cfc7ac5c137738985f61 \
        PP-OCRv6_medium_rec@e5a92bcbc5cc1b494628e458d267778f0704fd7c \
    && cd models && sha256sum --check --strict /opt/ocrbench/models.sha256
COPY ppstructurev3_engine.py /opt/ocrbench/ppstructurev3_engine.py

# Offline: no model-host connectivity check, no silent fallback from GPU to CPU, and the
# PaddleX cache in the writable /tmp (the runner mounts the root filesystem read-only).
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_PDX_DISABLE_DEVICE_FALLBACK=True \
    PADDLE_PDX_CACHE_HOME=/tmp/paddlex \
    OCRBENCH_PPSTRUCTUREV3_MODELS=/opt/ocrbench/models

LABEL org.opencontainers.image.title="ocrbench-engine-ppstructurev3" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="ppstructurev3"

# The runner uses the host UID, which cannot enter /home/ocr; PaddleX stats relative paths.
WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/ppstructurev3_engine.py"]
