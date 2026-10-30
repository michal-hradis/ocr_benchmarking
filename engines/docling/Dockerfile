# syntax=docker/dockerfile:1
# Docling standard image pipeline adapter (engines/docling/README.md) on the shared torch base.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# Docling 2.145.0 (docling-slim with the layout, TableFormer, RapidOCR/onnxruntime and image
# conversion extras) and its dependencies, hash-locked against the base (make docker-lock).
# torchvision comes from the PyTorch CUDA 13.0 index, everything else from PyPI. OpenCV is the
# headless build that RapidOCR declares; the base has no libGL.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/engine-requirements.txt

# No weights in the image: the layout, TableFormer and RapidOCR models are read from the /cache
# volume and checked against these checksums (make docling-model downloads them).
COPY models.sha256 /opt/ocrbench/models.sha256
COPY docling_engine.py /opt/ocrbench/docling_engine.py

# Offline: Docling reads every model from the artifacts directory the adapter passes and must not
# reach Hugging Face or RapidOCR's model hosts.
ENV HF_HUB_OFFLINE=1

LABEL org.opencontainers.image.title="ocrbench-engine-docling" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="docling"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/docling_engine.py"]
