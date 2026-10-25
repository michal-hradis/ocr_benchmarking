"""Fake HunyuanOCR-1.5 (HunYuanVL) processor and model (README.md one level up)."""

import math
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "output.md").read_text(encoding="utf-8")
IMAGE, END, ASSISTANT, PAD = 120120, 120020, 120007, 120002
BOS, SYSTEM = "<｜hy_begin▁of▁sentence｜>", "<｜hy_place▁holder▁no▁3｜>"
IMAGE_SPAN = "<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>"
FACTOR = 32  # 16-pixel patches, one image token per 2 x 2 patches
MIN_PIXELS, MAX_PIXELS = 262144, 16777216  # the checkpoint's preprocessor_config.json
REPEATED = "The same line again and again.\n"


def newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def smart_resize(height, width):
    """As transformers' HunYuanVL image processor: whole 32-pixel units within the pixel range."""
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


def _check(path):
    if not (Path(path) / "config.json").is_file():
        raise OSError(f"{path} does not appear to have a file named config.json")


def _decode(ids, skip_special_tokens, clean_up_tokenization_spaces=False):
    assert skip_special_tokens
    assert not clean_up_tokenization_spaces
    return "".join(chr(i) for i in ids if i < PAD)


class StoppingCriteria:
    pass


class AutoProcessor:
    image_token_id = IMAGE
    tokenizer = SimpleNamespace(decode=_decode)

    def __init__(self):
        self.prompt = None

    @classmethod
    def from_pretrained(cls, path, backend, local_files_only):
        assert backend == "pil"  # the official use_fast=False
        assert local_files_only
        _check(path)
        return cls()

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        system, user = conversation
        assert system == {"role": "system", "content": ""}
        image, text = user["content"]  # the image before the instruction
        assert user["role"] == "user"
        assert image == {"type": "image"}
        assert text["type"] == "text"
        assert add_generation_prompt
        assert not tokenize
        return f"{BOS}{SYSTEM}{IMAGE_SPAN}{text['text']}<｜hy_User｜>"

    def __call__(self, text, images, return_tensors):
        [prompt] = text
        assert prompt.startswith(BOS + SYSTEM + IMAGE_SPAN)
        assert return_tensors == "pt"
        [image] = images
        assert image.mode == "RGB"
        width, height = image.size
        height, width = smart_resize(height, width)
        rows, columns = height // FACTOR, width // FACTOR
        instruction = prompt[len(BOS + SYSTEM + IMAGE_SPAN) : -len("<｜hy_User｜>")]
        tokens = [1, 2] + [IMAGE] * (rows * (columns + 1) + 2) + [3] * len(instruction) + [4]
        return {
            "input_ids": Tensor([tokens]),
            "attention_mask": Tensor([[1] * len(tokens)]),
            "pixel_values": Tensor([[0.0]], floating=True, meta={"gray": image.getpixel((0, 0))}),
            "image_grid_thw": Tensor([[1, height // 16, width // 16]]),
            "mm_token_type_ids": Tensor([[0] * len(tokens)]),
        }

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        return _decode(ids, skip_special_tokens, clean_up_tokenization_spaces)


class HunYuanVLForConditionalGeneration:
    def __init__(self, dtype):
        self.dtype = dtype
        self.generation_config = SimpleNamespace(eos_token_id=[ASSISTANT, END])

    @classmethod
    def from_pretrained(cls, path, dtype, device_map, attn_implementation, local_files_only):
        assert local_files_only
        assert device_map in ("cpu", "cuda:0")
        assert attn_implementation == "sdpa"
        _check(path)
        return cls(dtype)

    def eval(self):
        return self

    def generate(self, input_ids, pixel_values, image_grid_thw, max_new_tokens, **kw):
        assert kw.pop("do_sample") is False  # greedy decoding
        assert kw.pop("repetition_penalty") > 0
        criteria = kw.pop("stopping_criteria")
        assert kw.keys() == {"attention_mask", "mm_token_type_ids"}
        gray = pixel_values.meta["gray"][0]
        if gray == 200:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if gray in (100, 150):  # a newspaper page that never ends, or that ends repeating a line
            text = "".join(
                newspaper_line(n) + "\n\n" for n in range(1, 101 if gray == 150 else 3000)
            )
            text += REPEATED * (max_new_tokens // len(REPEATED) + 1)
            tokens = [ord(c) for c in text]
        else:
            text = {255: PAGE, 250: ""}.get(gray, "x")
            tokens = [ord(c) for c in text] + [END, PAD, PAD]
        prompt = input_ids.data[0]
        generated = []
        for token in tokens[:max_new_tokens]:  # one token at a time, as generate() checks
            generated.append(token)
            if token == END or any(c(Tensor([prompt + generated]), None) for c in criteria):
                break
        return Tensor([prompt + generated])
