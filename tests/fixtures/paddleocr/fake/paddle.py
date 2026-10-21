"""Fake PaddlePaddle: versions and one visible CUDA device (FAKE_PADDLE_GPUS overrides)."""

import os
from types import SimpleNamespace

__version__ = "3.4.0-fake"
version = SimpleNamespace(cuda=lambda: "13.0", cudnn=lambda: "9.13.0")
_count = int(os.environ.get("FAKE_PADDLE_GPUS", "1"))
device = SimpleNamespace(
    is_compiled_with_cuda=lambda: True,
    cuda=SimpleNamespace(device_count=lambda: _count, get_device_name=lambda i: "Fake GPU"),
)
