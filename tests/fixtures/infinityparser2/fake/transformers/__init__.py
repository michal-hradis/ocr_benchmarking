"""Fake Infinity-Parser2-Flash (Qwen3.5) processor and model (README.md one level up)."""

import json
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "page.json").read_text(encoding="utf-8")
ENGINE = FIXTURES.parents[2] / "engines" / "infinityparser2"
INSTRUCTION = (ENGINE / "prompt.txt").read_text(encoding="utf-8")
# <|image_pad|>, <|im_end|> (the tokenizer's end token) and <|endoftext|> (text_config's).
IMAGE, END, TEXT_END = 248056, 248046, 248044
PROMPT = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>(prompt)<|im_end|>\n"
ASSISTANT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
FACTOR = 32  # 16-pixel patches, one image token per 2 x 2 patches
# A complete array followed by a cut-off element, although the model ended (width 1280).
MALFORMED = '[{"bbox": [0, 0, 1000, 500], "category": "text", "text": "kept"}, {"bbox": [0, 5'


def newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def newspaper_element(number):
    box = [10, number % 1000, 990, number % 1000 + 1]
    return {"bbox": box, "category": "text", "text": newspaper_line(number)}


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

    def __call__(self, text, images, do_resize, return_tensors):
        assert text == [PROMPT + ASSISTANT]
        assert return_tensors == "pt"
        assert do_resize is False  # the adapter resizes, as the official pipeline
        [image] = images
        width, height = image.size
        assert image.mode == "RGB"
        assert width % FACTOR == 0
        assert height % FACTOR == 0
        tokens = [1, 2, 3] + [IMAGE] * ((height // FACTOR) * (width // FACTOR)) + [4, 5]
        return {
            "input_ids": Tensor([tokens]),
            "attention_mask": Tensor([[1] * len(tokens)]),
            "pixel_values": Tensor([[0.0]], floating=True, meta={"size": image.size}),
            "image_grid_thw": Tensor([[1, height // 16, width // 16]]),
            "mm_token_type_ids": Tensor([[0] * len(tokens)]),
            "token_type_ids": Tensor([[0] * len(tokens)]),
        }

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(i) for i in ids if i < TEXT_END)


class Qwen3_5ForConditionalGeneration:  # noqa: N801 (the transformers name)
    def __init__(self, dtype):
        self.dtype = dtype
        # The checkpoint has no generation_config.json; text_config names <|endoftext|>.
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
        assert "token_type_ids" not in kw
        width = pixel_values.meta["size"][0]  # of the resized image
        if width == 672:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if width == 1984:
            text, number = "[", 0
            while len(text) < max_new_tokens:
                number += 1
                text += json.dumps(newspaper_element(number)) + ", "
            tokens = [ord(c) for c in text]
        elif width == 960:
            tokens = [ord(c) for c in "[]"] + [TEXT_END, TEXT_END]
        else:
            text = {896: PAGE, 288: "", 1216: "Sorry, I cannot read this.", 1280: MALFORMED}
            tokens = [ord(c) for c in text.get(width, "[]")] + [END, TEXT_END, TEXT_END]
        return Tensor([input_ids.data[0] + tokens[:max_new_tokens]])
