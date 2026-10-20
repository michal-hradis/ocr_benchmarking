# syntax=docker/dockerfile:1
# Fake OCR engine image for the opt-in runner Docker test (tests/test_runner.py).
FROM ocrbench/base-cpu:py3.12-v1

USER root
COPY fake_engine.py /opt/fake_engine.py
USER ocr
CMD ["python", "/opt/fake_engine.py"]
