"""Fake OpenCV: imread returns an object with the image shape, or None if unreadable."""

from types import SimpleNamespace

from PIL import Image

IMREAD_COLOR, IMREAD_IGNORE_ORIENTATION = 1, 128


def imread(path, flags):
    assert flags == IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION
    try:
        with Image.open(path) as image:
            return SimpleNamespace(shape=(image.height, image.width, 3))
    except OSError:
        return None
