"""Fake PyTorch: versions, one visible 24 GB CUDA device (FAKE_TORCH_GPUS overrides), a seed, and
list-backed tensors with only the operations the adapter uses."""

import contextlib
import os
from types import SimpleNamespace

__version__ = "2.14.1-fake"
version = SimpleNamespace(cuda="13.0")
bfloat16, float32 = "bfloat16", "float32"
_count = int(os.environ.get("FAKE_TORCH_GPUS", "1"))
cuda = SimpleNamespace(
    device_count=lambda: _count,
    get_device_name=lambda i: "Fake GPU",
    get_device_properties=lambda i: SimpleNamespace(total_memory=24 << 30),
    reset_peak_memory_stats=lambda device: None,
    max_memory_allocated=lambda device: 3 << 30,
)
seeds = []


def manual_seed(seed):
    seeds.append(seed)


def inference_mode():
    return contextlib.nullcontext()


class Tensor:
    """A list of rows; ``meta`` carries what the fake model needs to know about the image."""

    def __init__(self, data, floating=False, meta=None):
        self.data, self.floating, self.meta = data, floating, meta or {}

    def to(self, device=None, dtype=None):
        return self

    def is_floating_point(self):
        return self.floating

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, columns = key
            return Tensor(self.data[row][columns])
        return Tensor(self.data[key])

    def tolist(self):
        return list(self.data)
