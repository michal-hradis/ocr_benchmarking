# syntax=docker/dockerfile:1
# PyLaia HTR engine (engines/pylaia/README.md): Kraken page segmentation in the torch base's venv,
# PyLaia line recognition in a second venv. Build context: engines/ (make docker-pylaia), so the
# image reuses the Kraken engine's lock and adapter unchanged.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# PyLaia's `import laia` runs `git describe` and fails when no git executable exists; Ubuntu 24.04
# package pinned by version.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git=1:2.43.0-1ubuntu7.4 \
    && rm -rf /var/lib/apt/lists/*

# Segmentation: Kraken 7.1.1 and its default blla.mlmodel from the Kraken engine's hash-locked
# requirements, so pages are segmented exactly as by engines/kraken.
COPY kraken/requirements.txt /opt/ocrbench/kraken-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/kraken-requirements.txt

# Recognition: PyLaia 1.1.2 pins Python < 3.11 and torch 1.13, so it gets its own venv with a
# uv-managed Python 3.10 and torch 1.13.1 (CUDA 11.7 libraries from PyPI), hash-locked with the
# packages available on PyLaia's release date (make docker-lock).
COPY pylaia/pylaia-requirements.txt /opt/ocrbench/pylaia-requirements.txt
RUN uv python install 3.10.22 \
    && uv venv --python 3.10.22 /opt/pylaia \
    && VIRTUAL_ENV=/opt/pylaia uv pip install --require-hashes --no-deps \
        -r /opt/ocrbench/pylaia-requirements.txt

# The PyLaia NewsEye Austrian model (MIT) from Hugging Face at a fixed revision; the build fails
# unless every file matches models.sha256.
ARG NEWSEYE_REVISION=5361aa77db34d23cb23b27194c7c095f01f5ea54
COPY pylaia/models.sha256 /opt/ocrbench/models.sha256
RUN mkdir -p /opt/ocrbench/models/newseye-austrian && cd /opt/ocrbench/models \
    && python -c "import urllib.request as u; \
[u.urlretrieve(f'https://huggingface.co/Teklia/pylaia-newseye-austrian/resolve/${NEWSEYE_REVISION}/{f}', \
    f'newseye-austrian/{f}') for f in ('model', 'weights.ckpt', 'syms.txt')]" \
    && sha256sum --check --strict /opt/ocrbench/models.sha256 \
    && chmod -R a+rX /opt/ocrbench/models
COPY kraken/kraken_engine.py pylaia/pylaia_engine.py pylaia/pylaia_worker.py /opt/ocrbench/

# Offline, with Lightning's and Matplotlib's caches in the writable /tmp (the runner mounts the
# root filesystem read-only).
ENV OCRBENCH_PYLAIA_MODELS=/opt/ocrbench/models \
    MPLCONFIGDIR=/tmp/matplotlib

LABEL org.opencontainers.image.title="ocrbench-engine-pylaia" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="pylaia"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/pylaia_engine.py"]
