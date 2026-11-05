# syntax=docker/dockerfile:1
# Docling standard image pipeline adapter (engines/docling/README.md) on the shared torch base.

# tesserocr 2.12.0 compiled from its source release against Ubuntu 24.04's libtesseract 5.3.4,
# the library that the tesseract CLI of the engine image uses, so both Docling Tesseract backends
# run the same Tesseract (README.md#versions). The compiler and headers stay in this stage.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1 AS tesserocr
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ pkg-config \
        libtesseract-dev=5.3.4-1build5 libleptonica-dev=1.82.0-3build4 \
    && rm -rf /var/lib/apt/lists/*
COPY tesserocr-build.txt /tmp/tesserocr-build.txt
ADD --checksum=sha256:362b5f7c2fd5bf3452a719829999f263ebca687bfe5da8c2eb1d77792264a4aa \
    https://files.pythonhosted.org/packages/source/t/tesserocr/tesserocr-2.12.0.tar.gz /tmp/
RUN uv pip install --require-hashes --no-deps -r /tmp/tesserocr-build.txt \
    && uv build --no-build-isolation --wheel --out-dir /wheels /tmp/tesserocr-2.12.0.tar.gz

FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# Tesseract 5.3.4 from Ubuntu 24.04 for Docling's Tesseract CLI backend and, through tesserocr,
# its bindings backend. The packages' own tessdata_fast eng model is not used: the adapter points
# Tesseract at a directory with the pinned tessdata_best models from /cache, the osd model and the
# output configs below, checked against tessdata.sha256.
COPY tessdata.sha256 /opt/ocrbench/tessdata.sha256
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr=5.3.4-1build5 libtesseract5=5.3.4-1build5 \
        tesseract-ocr-osd=1:4.1.0-2 tesseract-ocr-eng=1:4.1.0-2 \
    && rm -rf /var/lib/apt/lists/* \
    && cd /usr/share/tesseract-ocr/5/tessdata \
    && sha256sum --check --strict /opt/ocrbench/tessdata.sha256

# Docling 2.145.0 (docling-slim with the layout, TableFormer, RapidOCR/onnxruntime and image
# conversion extras) and its dependencies, hash-locked against the base (make docker-lock).
# torchvision comes from the PyTorch CUDA 13.0 index, everything else from PyPI. OpenCV is the
# headless build that RapidOCR declares; the base has no libGL. pandas, which Docling's Tesseract
# backends also need, is already among them.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/engine-requirements.txt
COPY --from=tesserocr /wheels /tmp/wheels
RUN uv pip install --no-deps /tmp/wheels/tesserocr-2.12.0-*.whl && rm -rf /tmp/wheels

# No weights in the image: the layout, TableFormer and RapidOCR models and the tessdata_best
# models are read from the /cache volume and checked against these checksums (make docling-model,
# make tesseract-best download them).
COPY models.sha256 tessdata_best.sha256 /opt/ocrbench/
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
