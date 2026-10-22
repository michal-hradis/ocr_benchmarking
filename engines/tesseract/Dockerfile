# syntax=docker/dockerfile:1
# Tesseract CPU engine adapter (engines/tesseract/README.md) on the shared CPU base.
FROM ocrbench/base-cpu:py3.12-v1

USER root
# Debian bookworm packages pinned by version: Tesseract 5.3.0 with the tessdata_fast 4.1.0
# models; the build fails unless the models match the recorded checksums.
COPY tessdata.sha256 /opt/ocrbench/tessdata.sha256
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr=5.3.0-2 libtesseract5=5.3.0-2 \
        tesseract-ocr-osd=1:4.1.0-2 tesseract-ocr-eng=1:4.1.0-2 \
        tesseract-ocr-ces=1:4.1.0-2 tesseract-ocr-deu=1:4.1.0-2 \
    && rm -rf /var/lib/apt/lists/* \
    && cd /usr/share/tesseract-ocr/5/tessdata \
    && sha256sum --check --strict /opt/ocrbench/tessdata.sha256
# Checksums of the opt-in tessdata_best models in /cache, checked by the adapter (README.md).
COPY tessdata_best.sha256 /opt/ocrbench/tessdata_best.sha256
COPY tesseract_engine.py /opt/ocrbench/tesseract_engine.py

LABEL org.opencontainers.image.title="ocrbench-engine-tesseract" \
      org.opencontainers.image.source="https://github.com/michal-hradis/ocr_benchmarking" \
      ocrbench.engine="tesseract"

USER ocr
CMD ["python", "/opt/ocrbench/tesseract_engine.py"]
