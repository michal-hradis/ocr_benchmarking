# syntax=docker/dockerfile:1
# PERO OCR offline page pipeline adapter (engines/pero/README.md) on the shared torch base.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# The runtime dependencies of pero-ocr's inference path, hash-locked against the base
# (make docker-lock), all from PyPI.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps -r /opt/ocrbench/engine-requirements.txt

# pero-ocr from its source archive at a fixed commit of the develop branch, the first line with
# TorchScript-exported transformer OCR models (README.md); the build fails unless the archive
# matches the checksum. Its declared dependencies are replaced by requirements.txt.
ARG PERO_COMMIT=81b496e3084e294d2945e139fe60cf291010a83f
ARG PERO_SHA256=e684442d023f552d42cd767dd18254ba1384e78c4e39808801be3b1e1288baca
RUN python -c "import urllib.request as u; u.urlretrieve('https://github.com/DCGM/pero-ocr/archive/${PERO_COMMIT}.tar.gz', '/tmp/pero-ocr.tar.gz')" \
    && echo "${PERO_SHA256}  /tmp/pero-ocr.tar.gz" | sha256sum --check --strict \
    && uv pip install --no-deps --no-build-isolation /tmp/pero-ocr.tar.gz \
    && rm /tmp/pero-ocr.tar.gz

# No weights in the image: the user-supplied model bundles are read from the /cache volume and
# checked against these checksums (make pero-bundles copies them there).
COPY bundles.sha256 /opt/ocrbench/bundles.sha256
COPY pero_engine.py /opt/ocrbench/pero_engine.py

# Offline, with Numba's cache in the writable /tmp (the runner mounts the root filesystem
# read-only).
ENV OCRBENCH_PERO_COMMIT=${PERO_COMMIT} \
    NUMBA_CACHE_DIR=/tmp/numba

LABEL org.opencontainers.image.title="ocrbench-engine-pero" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="pero"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/pero_engine.py"]
