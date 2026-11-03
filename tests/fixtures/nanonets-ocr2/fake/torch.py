"""Fake PyTorch: versions and one visible 24 GB CUDA device (FAKE_TORCH_GPUS overrides), with only
what the adapter's environment report uses."""

import os
from types import SimpleNamespace

__version__ = "2.13.0-fake"
version = SimpleNamespace(cuda="13.0")
_count = int(os.environ.get("FAKE_TORCH_GPUS", "1"))
cuda = SimpleNamespace(
    device_count=lambda: _count,
    get_device_name=lambda i: "Fake GPU",
    get_device_properties=lambda i: SimpleNamespace(total_memory=24 << 30),
    get_device_capability=lambda i: (8, 9),
)
