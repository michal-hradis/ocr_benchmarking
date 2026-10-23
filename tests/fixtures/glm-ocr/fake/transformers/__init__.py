"""Fake GLM-OCR (Glm46V) processor and model (README.md one level up)."""

import math
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "page.md").read_text(encoding="utf-8")
# <|image|>, <|endoftext|> and <|user|> (both end tokens in generation_config.json).
IMAGE, TEXT_END, USER = 59280, 59246, 59253
PROMPT = "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>Text Recognition:"
ASSISTANT = "<|assistant|>\n"
FACTOR, FRAMES = 28, 2  # 14-pixel patches, one image token per 2 x 2; two temporal frames
# What the real model wrote for a blank page: an empty fenced block.
BLANK = "```markdown\n\n```"


def newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def smart_resize(height, width, min_pixels, max_pixels):
    """As transformers' Glm46V image processor: whole 28-pixel units, both frames counted
    against the pixel range."""
    h_bar, w_bar = round(height / FACTOR) * FACTOR, round(width / FACTOR) * FACTOR
    if FRAMES * h_bar * w_bar > max_pixels:
        beta = math.sqrt(FRAMES * height * width / max_pixels)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif FRAMES * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (FRAMES * height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def _check(path):
    if not (Path(path) / "config.json").is_file():
        raise OSError(f"{path} does not appear to have a file named config.json")


class AutoProcessor:
    image_token_id = IMAGE
    tokenizer = SimpleNamespace(pad_token_id=TEXT_END)

    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert local_files_only
        _check(path)
        return cls()

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        # No other template arguments (such as enable_thinking): the template's defaults.
        assert conversation == [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "Text Recognition:"}],
            }
        ]
        assert add_generation_prompt
        assert not tokenize
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
            "image_grid_thw": Tensor([[1, height // 14, width // 14]]),
            "mm_token_type_ids": Tensor([[0] * len(tokens)]),
        }

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(i) for i in ids if i < TEXT_END)


class GlmOcrForConditionalGeneration:
    def __init__(self, dtype):
        self.dtype = dtype
        self.generation_config = SimpleNamespace(eos_token_id=[TEXT_END, USER])

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
        assert kw["eos_token_id"] == [TEXT_END, USER]
        assert kw["pad_token_id"] == TEXT_END
        assert isinstance(kw["repetition_penalty"], float)
        width = pixel_values.meta["source"][0]
        if width == 666:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if width == 2000:
            tokens, number = [], 0
            while len(tokens) < max_new_tokens:
                number += 1
                tokens += [ord(c) for c in newspaper_line(number) + "\n\n"]
        elif width == 901:
            tokens = [ord("x"), USER, TEXT_END]
        else:
            text = {900: PAGE, 300: BLANK}.get(width, "x")
            tokens = [ord(c) for c in text] + [TEXT_END, TEXT_END]
        return Tensor([input_ids.data[0] + tokens[:max_new_tokens]])
