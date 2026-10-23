"""Fake NumPy: asarray passes the PIL image through, so the fake predictor sees its size."""


def asarray(image):
    return image
