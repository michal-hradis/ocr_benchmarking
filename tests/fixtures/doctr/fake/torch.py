"""Fake PyTorch: versions, devices and one visible 24 GB CUDA device (FAKE_TORCH_GPUS overrides)."""

import os
from types import SimpleNamespace

__version__ = "2.14.1-fake"
version = SimpleNamespace(cuda="13.0")
_count = int(os.environ.get("FAKE_TORCH_GPUS", "1"))
cuda = SimpleNamespace(
    device_count=lambda: _count,
    get_device_name=lambda i: "Fake GPU",
    get_device_properties=lambda i: SimpleNamespace(total_memory=24 << 30),
)


def device(name):
    return name
