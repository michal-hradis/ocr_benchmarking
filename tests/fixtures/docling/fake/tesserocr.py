"""Fake tesserocr 2.12.0 on Tesseract 5.3.4 (FAKE_TESSERACT_VERSION overrides)."""

import os

__version__ = "2.12.0"


def tesseract_version():
    version = os.environ.get("FAKE_TESSERACT_VERSION", "5.3.4")
    return f"tesseract {version}\n leptonica-1.82.0\n"
