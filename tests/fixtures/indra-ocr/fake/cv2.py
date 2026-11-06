"""Fake OpenCV: decodes an image with Pillow and returns an object with the array's shape."""

from PIL import Image

IMREAD_GRAYSCALE = 0
IMREAD_IGNORE_ORIENTATION = 128
THREADS = []


class Gray:
    def __init__(self, size):
        self.shape = (size[1], size[0])


def setNumThreads(n):  # noqa: N802 (OpenCV's name)
    THREADS.append(n)


def imread(path, flags):
    assert flags == IMREAD_GRAYSCALE | IMREAD_IGNORE_ORIENTATION, flags
    try:
        with Image.open(path) as image:
            return Gray(image.size)
    except Exception:
        return None  # as OpenCV does for an undecodable file
