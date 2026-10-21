# syntax=docker/dockerfile:1
# CUDA 13.0 + PaddlePaddle 3.4.0 base for Paddle GPU engine images. See docs/DOCKER.md.
# A separate family from base-cu130-torch: Paddle pins other cuDNN and nvJitLink versions.
# The Paddle wheel uses the CUDA user-space libraries from the nvidia-* wheels; the
# nvidia/cuda "base" image only provides Ubuntu and the NVIDIA container runtime settings.
FROM nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1

COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface

# Paddle needs the GNU OpenMP runtime; the version matches the gcc-14-base of the pinned image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1=14.2.0-4ubuntu2~24.04.1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/ocrbench/base-requirements.txt
RUN uv python install 3.12.15 \
    && uv venv --python 3.12.15 /opt/venv \
    && uv pip sync --require-hashes /opt/ocrbench/base-requirements.txt \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin ocr \
    && install -d -o ocr -g ocr /cache

LABEL org.opencontainers.image.title="ocrbench-base-cu130-paddle" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.base.family="cu130-paddle3.4.0-py3.12"

USER ocr
WORKDIR /home/ocr
