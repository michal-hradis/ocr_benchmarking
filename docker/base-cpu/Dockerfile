# syntax=docker/dockerfile:1
# Minimal CPU base for engine images. See docs/DOCKER.md.
FROM python:3.12.15-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

RUN uv venv --python /usr/local/bin/python3.12 /opt/venv \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin ocr \
    && install -d -o ocr -g ocr /cache

LABEL org.opencontainers.image.title="ocrbench-base-cpu" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.base.family="cpu-py3.12"

USER ocr
WORKDIR /home/ocr
