# syntax=docker/dockerfile:1
# OpenRouter hosted vision-model engine adapter (engines/openrouter/README.md) on the shared CPU base.
# The adapter calls OpenRouter's API; it runs only with the runner's --allow-network (ADR 0013).
FROM ocrbench/base-cpu:py3.12-v1

USER root
# Pillow reads the image format and size, hash-locked (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt
COPY openrouter_engine.py /opt/ocrbench/openrouter_engine.py

LABEL org.opencontainers.image.title="ocrbench-engine-openrouter" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="openrouter"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/openrouter_engine.py"]
