# syntax=docker/dockerfile:1
# Kraken historical OCR/HTR engine adapter (engines/kraken/README.md) on the shared torch base.
FROM ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1

USER root
# Kraken 7.1.1 and its dependencies, hash-locked against the base (make docker-lock); the lock
# overrides Kraken's torch<=2.14 bound to keep the base's torch 2.14.1 (overrides.txt).
# torchvision comes from the PyTorch CUDA 13.0 index, everything else from PyPI.
COPY requirements.txt /opt/ocrbench/engine-requirements.txt
RUN uv pip install --require-hashes --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        -r /opt/ocrbench/engine-requirements.txt

# Recognition models from Zenodo (CC BY 4.0, CC0) at fixed record versions; the build fails unless
# every file matches models.sha256. The default segmentation model (blla.mlmodel, Apache-2.0)
# ships inside the hash-locked kraken wheel.
COPY models.sha256 /opt/ocrbench/models.sha256
RUN mkdir -p /opt/ocrbench/models && cd /opt/ocrbench/models \
    && python -c 'import urllib.request as u; \
[u.urlretrieve(f"https://zenodo.org/records/{r}/files/{f}?download=1", f) for r, f in ( \
    ("10592716", "catmus-print-fondue-large.mlmodel"), \
    ("13788177", "McCATMuS_nfd_nofix_V1.mlmodel"), \
    ("10519596", "german_print.mlmodel"))]' \
    && sha256sum --check --strict /opt/ocrbench/models.sha256
COPY kraken_engine.py /opt/ocrbench/kraken_engine.py

# Offline, with Lightning's and Matplotlib's caches in the writable /tmp (the runner mounts the
# root filesystem read-only).
ENV OCRBENCH_KRAKEN_MODELS=/opt/ocrbench/models \
    MPLCONFIGDIR=/tmp/matplotlib

LABEL org.opencontainers.image.title="ocrbench-engine-kraken" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="kraken"

WORKDIR /tmp
USER ocr
CMD ["python", "/opt/ocrbench/kraken_engine.py"]
