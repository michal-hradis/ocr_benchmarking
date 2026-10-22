"""Fake LightOnOCR processor and model (README.md one level up)."""

import math
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

PAGE = (Path(__file__).resolve().parents[2] / "page.md").read_text(encoding="utf-8")
IMAGE, END, PAD = 151655, 151645, 151643
PROMPT = (
    "<|im_start|>system<|im_end|>\n<|im_start|>user\n<|image_pad|><|im_end|>\n"
    "<|im_start|>assistant\n"
)
PATCH, MERGED = 14, 28  # one image token per 2 x 2 patches


def newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def _check(path):
    if not (Path(path) / "config.json").is_file():
        raise OSError(f"{path} does not appear to have a file named config.json")


class LightOnOcrProcessor:
    image_token_id = IMAGE

    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert local_files_only
        _check(path)
        return cls()

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        assert conversation == [{"role": "user", "content": [{"type": "image"}]}]
        assert add_generation_prompt
        assert not tokenize
        return PROMPT

    def __call__(self, text, images, size, return_tensors):
        assert text == [PROMPT]
        assert return_tensors == "pt"
        [image] = images
        width, height = image.size
        ratio = max(width / size["longest_edge"], height / size["longest_edge"], 1)
        height = math.ceil(math.floor(height / ratio) / PATCH) * PATCH
        width = math.ceil(math.floor(width / ratio) / PATCH) * PATCH
        tokens = [1, 2, 3] + [IMAGE] * ((height // MERGED) * (width // MERGED)) + [4, 5]
        return {
            "input_ids": Tensor([tokens]),
            "attention_mask": Tensor([[1] * len(tokens)]),
            "pixel_values": Tensor([[0.0]], floating=True, meta={"source": image.size}),
            "image_sizes": Tensor([[height, width]]),
        }

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return "".join(chr(i) for i in ids if i < PAD)


class LightOnOcrForConditionalGeneration:
    def __init__(self, dtype):
        self.dtype = dtype
        self.generation_config = SimpleNamespace(eos_token_id=[END, PAD])

    @classmethod
    def from_pretrained(cls, path, dtype, local_files_only):
        assert local_files_only
        _check(path)
        return cls(dtype)

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, pixel_values, image_sizes, max_new_tokens, **kw):
        assert kw.get("do_sample") is (kw.get("temperature", 0) > 0)
        width = pixel_values.meta["source"][0]
        if width == 666:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if width == 2000:
            text = "\n\n".join(newspaper_line(n) for n in range(1, max_new_tokens))
            tokens = [ord(c) for c in text]
        else:
            text = {900: PAGE, 300: ""}.get(width, "x")
            tokens = [ord(c) for c in text] + [END, PAD, PAD]
        return Tensor([input_ids.data[0] + tokens[:max_new_tokens]])
