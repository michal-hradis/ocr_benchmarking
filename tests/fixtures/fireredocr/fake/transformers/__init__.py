"""Fake FireRed-OCR (Qwen3-VL) processor and model (README.md one level up)."""

import math
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "page.md").read_text(encoding="utf-8")
INSTRUCTION = (FIXTURES.parents[2] / "engines" / "fireredocr" / "prompt.txt").read_text("utf-8")
IMAGE, END, PAD = 151655, 151645, 151643
PROMPT = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>(prompt)<|im_end|>\n"
FACTOR = 32  # 16-pixel patches, one image token per 2 x 2 patches


def newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def smart_resize(height, width, min_pixels, max_pixels):
    """As transformers' Qwen2-VL image processor: whole 32-pixel units within the pixel range."""
    h_bar, w_bar = round(height / FACTOR) * FACTOR, round(width / FACTOR) * FACTOR
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def _check(path):
    if not (Path(path) / "config.json").is_file():
        raise OSError(f"{path} does not appear to have a file named config.json")


class AutoProcessor:
    image_token_id = IMAGE

    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert local_files_only
        _check(path)
        return cls()

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        assert conversation == [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}
        ]
        assert add_generation_prompt
        assert not tokenize
        return PROMPT + "<|im_start|>assistant\n"

    def __call__(self, text, images, size, return_tensors):
        assert text == [PROMPT + "<|im_start|>assistant\n"]
        assert return_tensors == "pt"
        [image] = images
        width, height = image.size
        height, width = smart_resize(height, width, size["shortest_edge"], size["longest_edge"])
        tokens = [1, 2, 3] + [IMAGE] * ((height // FACTOR) * (width // FACTOR)) + [4, 5]
        return {
            "input_ids": Tensor([tokens]),
            "attention_mask": Tensor([[1] * len(tokens)]),
            "pixel_values": Tensor([[0.0]], floating=True, meta={"source": image.size}),
            "image_grid_thw": Tensor([[1, height // 16, width // 16]]),
            "mm_token_type_ids": Tensor([[0] * len(tokens)]),
        }

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(i) for i in ids if i < PAD)


class Qwen3VLForConditionalGeneration:
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

    def generate(self, input_ids, pixel_values, image_grid_thw, max_new_tokens, **kw):
        sampling = kw.get("temperature", 0) > 0
        assert kw.get("do_sample") is sampling
        assert ("top_k" in kw) is sampling
        width = pixel_values.meta["source"][0]
        if width == 666:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if width == 2000:
            tokens, number = [], 0
            while len(tokens) < max_new_tokens:
                number += 1
                tokens += [ord(c) for c in newspaper_line(number) + "\n\n"]
        else:
            text = {900: PAGE, 300: ""}.get(width, "x")
            tokens = [ord(c) for c in text] + [END, PAD, PAD]
        return Tensor([input_ids.data[0] + tokens[:max_new_tokens]])
