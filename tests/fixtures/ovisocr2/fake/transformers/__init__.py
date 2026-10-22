"""Fake OvisOCR2 (Qwen3.5) processor and model (README.md one level up)."""

import math
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "page.md").read_text(encoding="utf-8")
INSTRUCTION = (FIXTURES.parents[2] / "engines" / "ovisocr2" / "prompt.txt").read_text("utf-8")
# <|image_pad|>, <|im_end|> (the tokenizer's end token) and <|endoftext|> (config.json's).
IMAGE, END, TEXT_END = 248056, 248046, 248044
PROMPT = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>(prompt)<|im_end|>\n"
ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
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
    tokenizer = SimpleNamespace(eos_token_id=END, pad_token_id=TEXT_END)

    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert local_files_only
        _check(path)
        return cls()

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize, enable_thinking):
        assert conversation == [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}
        ]
        assert add_generation_prompt
        assert not tokenize
        assert enable_thinking is False
        return PROMPT + ASSISTANT

    def __call__(self, text, images, size, return_tensors):
        assert text == [PROMPT + ASSISTANT]
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
        return "".join(chr(i) for i in ids if i < TEXT_END)


class Qwen3_5ForConditionalGeneration:  # noqa: N801 (the transformers name)
    def __init__(self, dtype):
        self.dtype = dtype
        # The checkpoint has no generation_config.json; config.json names only <|endoftext|>.
        self.config = SimpleNamespace(text_config=SimpleNamespace(eos_token_id=TEXT_END))

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
        assert kw["do_sample"] is False
        assert kw["eos_token_id"] == [END, TEXT_END]
        assert kw["pad_token_id"] == TEXT_END
        width = pixel_values.meta["source"][0]
        if width == 666:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if width == 2000:
            tokens, number = [], 0
            while len(tokens) < max_new_tokens:
                number += 1
                tokens += [ord(c) for c in newspaper_line(number) + "\n\n"]
        elif width == 901:
            tokens = [ord("x"), TEXT_END, TEXT_END]
        else:
            text = {900: PAGE, 300: ""}.get(width, "x")
            tokens = [ord(c) for c in text] + [END, TEXT_END, TEXT_END]
        return Tensor([input_ids.data[0] + tokens[:max_new_tokens]])
