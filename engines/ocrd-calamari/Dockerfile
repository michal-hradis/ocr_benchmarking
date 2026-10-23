# syntax=docker/dockerfile:1
# OCR-D pipeline with Calamari recognition (engines/ocrd-calamari/README.md), CPU only. A standalone
# image (ADR 0004): ocrd_calamari 2.1.0 needs TensorFlow < 2.16 and so Python 3.11, which no base
# family provides. It follows the base conventions (docs/DOCKER.md): /opt/venv, user ocr, /cache.
FROM python:3.11.17-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf

COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    XDG_CACHE_HOME=/cache

RUN uv venv --python /usr/local/bin/python3.11 /opt/venv \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin ocr \
    && install -d -o ocr -g ocr /cache

# libmagic for OCR-D core (python-magic) and Tesseract's English model (tessdata_fast 4.1.0), which
# ocrd-tesserocr-segment loads for its layout analysis; Debian bookworm packages pinned by version.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libmagic1=1:5.44-3 tesseract-ocr-eng=1:4.1.0-2 \
    && rm -rf /var/lib/apt/lists/*

# OCR-D core, ocrd_cis, ocrd_tesserocr (with the tesserocr wheel's bundled Tesseract), Calamari and
# TensorFlow, hash-locked for Python 3.11 (make docker-lock).
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt

# ocrd_calamari 2.1.0 (not on PyPI) from its source archive at the release commit; the build fails
# unless the archive matches the checksum. Its dependencies come from requirements.txt.
ARG OCRD_CALAMARI_COMMIT=4d66fa1b809abe124a8e632a0eda6a2c133d9f2d
ARG OCRD_CALAMARI_SHA256=e74eba9a1ba522b710dfc6d9d898f3ed6940b189f90e5eeb13ed8a22c50bf730
RUN python -c "import urllib.request as u; u.urlretrieve('https://github.com/OCR-D/ocrd_calamari/archive/${OCRD_CALAMARI_COMMIT}.tar.gz', '/tmp/ocrd_calamari.tar.gz')" \
    && echo "${OCRD_CALAMARI_SHA256}  /tmp/ocrd_calamari.tar.gz" | sha256sum --check --strict \
    && uv pip install --no-deps --no-build-isolation /tmp/ocrd_calamari.tar.gz \
    && rm /tmp/ocrd_calamari.tar.gz

# The three Calamari model ensembles of the presets (calamari_models release 2.2, MIT, about 30 MB
# each); the build fails unless every archive matches models.sha256.
COPY models.sha256 /opt/ocrbench/models.sha256
RUN mkdir -p /opt/ocrbench/models && cd /opt/ocrbench/models \
    && python -c 'import urllib.request as u; \
[u.urlretrieve(f"https://github.com/Calamari-OCR/calamari_models/releases/download/2.2/{m}.tar.gz", f"{m}.tar.gz") \
 for m in ("gt4histocr", "fraktur_19th_century", "antiqua_historical")]' \
    && sha256sum --check --strict /opt/ocrbench/models.sha256 \
    && for archive in *.tar.gz; do tar -xzf "$archive" --no-same-owner && rm "$archive"; done \
    && chmod -R a+rX /opt/ocrbench/models
COPY ocrd_calamari_engine.py /opt/ocrbench/ocrd_calamari_engine.py

# Offline on the CPU, with TensorFlow's logs quiet and the Tesseract models from the Debian package.
# A processor error on a page is re-raised (OCRD_MISSING_OUTPUT=ABORT) instead of skipping the page.
# GitPython, imported by OCR-D's unused resource manager, must not require a git executable. The
# runner mounts the root filesystem read-only and sets HOME=/tmp; OCR-D's data and config
# locations point into /tmp.
ENV OCRBENCH_OCRD_CALAMARI_MODELS=/opt/ocrbench/models \
    OCRBENCH_OCRD_CALAMARI_COMMIT=${OCRD_CALAMARI_COMMIT} \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    CUDA_VISIBLE_DEVICES="" \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OCRD_MISSING_OUTPUT=ABORT \
    GIT_PYTHON_REFRESH=quiet \
    XDG_DATA_HOME=/tmp/xdg-data \
    XDG_CONFIG_HOME=/tmp/xdg-config

LABEL org.opencontainers.image.title="ocrbench-engine-ocrd-calamari" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="ocrd-calamari"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/ocrd_calamari_engine.py"]
