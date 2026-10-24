"""Fake PyTorch: versions, one visible 24 GB CUDA device (FAKE_TORCH_GPUS overrides), autocast and
no_grad, and list-backed tensors with only the operations the adapter uses."""

import contextlib
import os
from types import SimpleNamespace

__version__ = "2.14.1-fake"
version = SimpleNamespace(cuda="13.0")
bfloat16, long, bool = "bfloat16", "long", "bool"  # noqa: A001
_count = int(os.environ.get("FAKE_TORCH_GPUS", "1"))
devices = []
cuda = SimpleNamespace(
    device_count=lambda: _count,
    set_device=devices.append,
    get_device_name=lambda i: "Fake GPU",
    get_device_properties=lambda i: SimpleNamespace(total_memory=24 << 30),
    reset_peak_memory_stats=lambda device: None,
    max_memory_allocated=lambda device: 7 << 30,
)


def autocast(device_type, dtype):
    assert (device_type, dtype) == ("cuda", bfloat16)
    return contextlib.nullcontext()


def no_grad():
    return contextlib.nullcontext()


class Tensor:
    """Nested lists; ``meta`` carries what the fake model needs to know about the images."""

    def __init__(self, data, dtype=None, meta=None):
        self.data, self.dtype, self.meta = data, dtype, meta or {}

    def to(self, device_or_dtype):
        return Tensor(self.data, device_or_dtype if device_or_dtype == bfloat16 else self.dtype,
                      self.meta)  # fmt: skip

    def unsqueeze(self, dim):
        assert dim == 0
        return Tensor([self.data], self.dtype, self.meta)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        row, columns = key
        return Tensor(self.data[row][columns], self.dtype)

    def tolist(self):
        return list(self.data)


def LongTensor(data):  # noqa: N802
    return Tensor(list(data), long)


def tensor(data, dtype):
    return Tensor(list(data), dtype)


def stack(tensors):
    assert all(t.dtype == bfloat16 for t in tensors)
    return Tensor([t.data for t in tensors], bfloat16, {"views": [t.meta for t in tensors]})


def zeros(shape):
    return Tensor([0] * shape[0], meta={"views": [], "shape": shape})
