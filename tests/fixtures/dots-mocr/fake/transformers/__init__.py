"""Fake dots.mocr processor and model, as loaded with the checkpoint's own code (README.md one
level up)."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "output.json").read_text(encoding="utf-8").strip()
END, ASSISTANT, END_ASSISTANT = 151643, 151672, 151673  # generation_config.json eos_token_id
IMAGE, PAD = 151665, 151680  # <|imgpad|>, [PAD]
FACTOR = 28  # 14-pixel patches, one image token per 2 x 2 patches
MIN_PIXELS, MAX_PIXELS = 3136, 11289600  # the checkpoint's preprocessor_config.json
TEMPLATE = "<|user|><|img|><|imgpad|><|endofimg|>{}<|endofuser|><|assistant|>"


def newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def element(number, text):
    box = [20, 10 + 20 * number, 900, 28 + 20 * number]
    return json.dumps({"bbox": box, "category": "Text", "text": text}) + ", "


def smart_resize(height, width):
    """As Qwen2-VL's image processor: whole 28-pixel units within the pixel range."""
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


def _check(path, trust_remote_code, local_files_only):
    assert trust_remote_code  # the checkpoint's own code
    assert local_files_only
    if not (Path(path) / "config.json").is_file():
        raise OSError(f"{path} does not appear to have a file named config.json")


class AutoProcessor:
    image_token_id = IMAGE
    tokenizer = SimpleNamespace(pad_token_id=PAD)

    @classmethod
    def from_pretrained(cls, path, trust_remote_code, use_fast, local_files_only):
        assert use_fast  # as the official parser
        _check(path, trust_remote_code, local_files_only)
        return cls()

    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        [user] = conversation
        image, text = user["content"]  # the image before the instruction
        assert user["role"] == "user"
        assert image == {"type": "image"}
        assert text["type"] == "text"
        assert add_generation_prompt
        assert not tokenize
        return TEMPLATE.format(text["text"])

    def __call__(self, text, images, padding, return_tensors):
        [prompt] = text
        head, tail = TEMPLATE.split("{}")
        assert prompt.startswith(head)
        assert prompt.endswith(tail)
        assert return_tensors == "pt"
        [image] = images
        assert image.mode == "RGB"
        height, width = smart_resize(image.height, image.width)
        rows, columns = height // 14, width // 14
        instruction = prompt[len(head) : -len(tail)]
        image_tokens = rows * columns // 4
        tokens = [1, 2] + [IMAGE] * image_tokens + [3] + [4] * len(instruction) + [5, ASSISTANT]
        return {
            "input_ids": Tensor([tokens]),
            "attention_mask": Tensor([[1] * len(tokens)]),
            "pixel_values": Tensor([[0.0]], floating=True, meta={"gray": image.getpixel((0, 0))}),
            "image_grid_thw": Tensor([[1, rows, columns]]),
        }

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(i) for i in ids if i < END)


class AutoModelForCausalLM:
    def __init__(self, dtype):
        self.dtype = dtype
        self.generation_config = SimpleNamespace(eos_token_id=[END, ASSISTANT, END_ASSISTANT])

    @classmethod
    def from_pretrained(cls, path, dtype, attn_implementation, trust_remote_code, local_files_only):
        assert dtype == "bfloat16"
        assert attn_implementation == "sdpa"
        _check(path, trust_remote_code, local_files_only)
        # The model's vision code imports this at the top; flash-attn is not installed.
        from flash_attn import flash_attn_varlen_func

        assert callable(flash_attn_varlen_func)
        return cls(dtype)

    def to(self, device):
        assert device == "cuda:0"
        return self

    def eval(self):
        return self

    def generate(self, input_ids, pixel_values, image_grid_thw, max_new_tokens, **kw):
        assert kw.pop("do_sample") is False  # greedy decoding
        assert kw.pop("eos_token_id") == [END, ASSISTANT, END_ASSISTANT]
        assert kw.pop("pad_token_id") == PAD
        assert kw.keys() == {"attention_mask"}
        gray = pixel_values.meta["gray"][0]
        if gray == 200:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if gray == 100:  # a newspaper page whose list never ends: it repeats one element
            text = "[" + "".join(element(n, newspaper_line(n)) for n in range(1, 101))
            repeated = element(101, "The same line again and again.")
            text += repeated * (max_new_tokens // len(repeated) + 1)
            tokens = [ord(c) for c in text]
        else:
            text = {
                255: PAGE,
                250: "[]",  # a blank page
                150: "This page has no layout.",
                50: '[{"bbox": [1, 1, 50, 20], "category": "Text", "text": "Complete element."}, '
                '{"bbox": [1, 30',  # ends without closing the list
            }.get(gray, "[]")
            tokens = [ord(c) for c in text] + [END_ASSISTANT, PAD, PAD]
        prompt = input_ids.data[0]
        return Tensor([prompt + tokens[:max_new_tokens]])
