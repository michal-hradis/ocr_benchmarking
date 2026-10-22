"""Fake PyTorch: versions and one visible CUDA device (FAKE_TORCH_GPUS overrides)."""

import os
from types import SimpleNamespace

__version__ = "2.14.1-fake"
version = SimpleNamespace(cuda="13.0")
_count = int(os.environ.get("FAKE_TORCH_GPUS", "1"))
cuda = SimpleNamespace(
    device_count=lambda: _count,
    get_device_name=lambda i: "Fake GPU",
    reset_peak_memory_stats=lambda device: None,
    max_memory_allocated=lambda device: 3 << 30,
)


def device(name):
    return SimpleNamespace(type=name.split(":")[0], name=name)
