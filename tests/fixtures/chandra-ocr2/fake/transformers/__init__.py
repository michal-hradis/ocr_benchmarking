"""Fake Chandra OCR 2 (Qwen3.5) processor and model (README.md one level up)."""

import math
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "page.html").read_text(encoding="utf-8")
ENGINE = FIXTURES.parents[2] / "engines" / "chandra-ocr2"
INSTRUCTION = (ENGINE / "prompt.txt").read_text(encoding="utf-8")
# <|image_pad|>, <|im_end|> (ends the chat turn) and <|endoftext|> (generation_config's).
IMAGE, END, TEXT_END = 248056, 248046, 248044
FACTOR = 32  # 16-pixel patches, one image token per 2 x 2 patches
# preprocessor_config.json "size": the processor's own pixel limits.
MIN_PIXELS, MAX_PIXELS = 65536, 16777216
BLANK_PAGE = '<div data-bbox="0 0 1000 1000" data-label="Blank-Page"></div>'


def newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def newspaper_block(number):
    box = f"10 {number % 1000} 990 {number % 1000 + 1}"
    return f'<div data-bbox="{box}" data-label="Text"><p>{newspaper_line(number)}</p></div>\n'


def _check(path):
    if not (Path(path) / "config.json").is_file():
        raise OSError(f"{path} does not appear to have a file named config.json")


def smart_resize(height, width):
    """The size the Qwen2-VL image processor resizes to (factor 32, MIN_PIXELS, MAX_PIXELS)."""
    h_bar, w_bar = round(height / FACTOR) * FACTOR, round(width / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


class _Tokenizer:
    pad_token_id = TEXT_END
    padding_side = "right"

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": END}.get(token)


class AutoProcessor:
    image_token_id = IMAGE

    def __init__(self):
        self.tokenizer = _Tokenizer()

    @classmethod
    def from_pretrained(cls, path, local_files_only):
        assert local_files_only
        _check(path)
        return cls()

    def apply_chat_template(self, conversations, **kw):
        assert kw == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "padding": True,
        }
        assert self.tokenizer.padding_side == "left"
        [[message]] = conversations
        assert message["role"] == "user"
        image_part, text_part = message["content"]
        assert text_part == {"type": "text", "text": INSTRUCTION}
        assert image_part["type"] == "image"
        image = image_part["image"]
        assert image.mode == "RGB"
        width, height = image.size
        assert width % 28 == 0  # the adapter has scaled it to fit, as chandra-ocr
        assert height % 28 == 0
        model_height, model_width = smart_resize(height, width)
        units = (model_height // FACTOR) * (model_width // FACTOR)
        tokens = [1, 2, 3] + [IMAGE] * units + [4, 5]
        return {
            "input_ids": Tensor([tokens]),
            "attention_mask": Tensor([[1] * len(tokens)]),
            "pixel_values": Tensor([[0.0]], floating=True, meta={"size": image.size}),
            "image_grid_thw": Tensor([[1, model_height // 16, model_width // 16]]),
            "mm_token_type_ids": Tensor([[0] * len(tokens)]),
        }

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(i) for i in ids if i < TEXT_END)


class AutoModelForImageTextToText:
    def __init__(self, dtype):
        self.dtype = dtype
        # generation_config.json names <|endoftext|> only.
        self.generation_config = SimpleNamespace(eos_token_id=TEXT_END)

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
        assert kw["eos_token_id"] == [TEXT_END, END]
        assert kw["pad_token_id"] == TEXT_END
        width = pixel_values.meta["size"][0]  # of the image given to the processor
        if width == 2548:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if width == 1988:
            text, number = "", 0
            while len(text) < max_new_tokens:
                number += 1
                text += newspaper_block(number)
            tokens = [ord(c) for c in text]
        elif width == 4480:
            tokens = [ord(c) for c in BLANK_PAGE] + [TEXT_END, TEXT_END]
        else:
            text = {3304: PAGE, 4312: "Sorry, this is plain text."}
            tokens = [ord(c) for c in text.get(width, "")] + [END, TEXT_END, TEXT_END]
        return Tensor([input_ids.data[0] + tokens[:max_new_tokens]])
