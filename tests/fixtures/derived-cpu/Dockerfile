# syntax=docker/dockerfile:1
# Minimal engine-style image inheriting the locally built CPU base; used by
# tests/test_docker_images.py to exercise the adapter inheritance path.
FROM ocrbench/base-cpu:py3.12-v1

USER root
RUN echo "derived-cpu fixture" > /opt/ocrbench-derived
USER ocr
