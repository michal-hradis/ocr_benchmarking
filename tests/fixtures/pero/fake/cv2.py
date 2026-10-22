"""Fake OpenCV: imread returns a stand-in for the 8-bit BGR array with only ``shape``."""

from types import SimpleNamespace

from PIL import Image

IMREAD_COLOR = 1
IMREAD_IGNORE_ORIENTATION = 128


def imread(path, flags):
    assert flags == IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        return None  # like OpenCV, an undecodable image is None
    return SimpleNamespace(shape=(height, width, 3))
