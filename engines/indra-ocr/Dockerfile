# syntax=docker/dockerfile:1
# Indra OCR CPU engine adapter (engines/indra-ocr/README.md) on the shared CPU base: ONNX Runtime
# line detection, recognition, layout and reading order, no PyTorch and no GPU.
FROM ocrbench/base-cpu:py3.12-v1

USER root
# Indra OCR 1.1.2 with ONNX Runtime and OpenCV, hash-locked for Python 3.12 (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt

# The default models of Indra OCR 1.1.2 (Hugging Face datalab-to/indra-ocr, tag v1.1.0 at the
# commit below, which indra_engine.py's MODEL_COMMIT repeats; 73 MB, modified OpenRAIL-M license, copied with them), without the optional math
# models. The build fails unless every file matches models.sha256; the adapter checks them again
# before it loads them.
ARG INDRA_MODEL_COMMIT=a1283c4b42a273c09480ad50e71461daff50fcc7
COPY models.sha256 /opt/ocrbench/models.sha256
RUN mkdir -p /opt/ocrbench/models && cd /opt/ocrbench/models \
    && python -c "import sys, urllib.request as u; \
[u.urlretrieve(f'https://huggingface.co/datalab-to/indra-ocr/resolve/${INDRA_MODEL_COMMIT}/{name}', name) \
 for name in (line.split()[1] for line in open('/opt/ocrbench/models.sha256'))]" \
    && sha256sum --check --strict /opt/ocrbench/models.sha256 \
    && chmod -R a+rX /opt/ocrbench/models
COPY indra_engine.py /opt/ocrbench/indra_engine.py

# Offline: the models are read from the image only (INDRA_OCR_MODEL_DIR), never downloaded, and
# ONNX Runtime's telemetry is off.
ENV OCRBENCH_INDRA_MODELS=/opt/ocrbench/models \
    INDRA_OCR_MODEL_DIR=/opt/ocrbench/models \
    HF_HUB_OFFLINE=1 \
    ORT_DISABLE_TELEMETRY=1

LABEL org.opencontainers.image.title="ocrbench-engine-indra-ocr" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="indra-ocr"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/indra_engine.py"]
