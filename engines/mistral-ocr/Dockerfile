# syntax=docker/dockerfile:1
# Mistral OCR 4.1 hosted OCR engine adapter (engines/mistral-ocr/README.md) on the shared CPU base.
# The adapter calls Mistral's API; it runs only with the runner's --allow-network (ADR 0013).
FROM ocrbench/base-cpu:py3.12-v1

USER root
# Pillow reads the image size for the block boxes, hash-locked (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt
COPY mistral_ocr_engine.py /opt/ocrbench/mistral_ocr_engine.py

LABEL org.opencontainers.image.title="ocrbench-engine-mistral-ocr" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="mistral-ocr"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/mistral_ocr_engine.py"]
