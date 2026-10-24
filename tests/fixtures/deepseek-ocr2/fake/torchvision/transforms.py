"""Fake torchvision transforms: ToTensor records the size and the central gray level of a view."""

from torch import Tensor


class Compose:
    def __init__(self, steps):
        self.steps = steps

    def __call__(self, image):
        for step in self.steps:
            image = step(image)
        return image


class ToTensor:
    def __call__(self, image):
        assert image.mode == "RGB"
        gray = image.getpixel((image.width // 2, image.height // 2))[0]
        return Tensor([0.0], meta={"size": image.size, "gray": gray})


class Normalize:
    def __init__(self, mean, std):
        assert mean == std == (0.5, 0.5, 0.5)

    def __call__(self, tensor):
        return tensor
