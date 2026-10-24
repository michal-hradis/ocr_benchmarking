"""Fake olmOCR-2 (Qwen2.5-VL) processor and model (README.md one level up)."""

import math
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "output.md").read_text(encoding="utf-8")
INSTRUCTION = (FIXTURES.parents[2] / "engines" / "olmocr2" / "prompt.txt").read_text("utf-8")
IMAGE, END, PAD = 151655, 151645, 151643
PROMPT = "<|im_start|>user\n(prompt)<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
FACTOR = 28  # 14-pixel patches, one image token per 2 x 2 patches
MIN_PIXELS, MAX_PIXELS = 3136, 12845056  # the checkpoint's preprocessor_config.json
FRONT_MATTER = (
    "---\nprimary_language: {language}\nis_rotation_valid: True\nrotation_correction: 0\n"
    "is_table: False\nis_diagram: False\n---"
)


def newspaper_line(number):
    return f"Line {number} of a very long newspaper page."


def smart_resize(height, width, min_pixels, max_pixels):
    """As transformers' Qwen2-VL image processor: whole 28-pixel units within the pixel range."""
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
            {"role": "user", "content": [{"type": "text", "text": INSTRUCTION}, {"type": "image"}]}
        ]
        assert add_generation_prompt
        assert not tokenize
        return PROMPT + "<|im_start|>assistant\n"

    def __call__(self, text, images, return_tensors):
        assert text == [PROMPT + "<|im_start|>assistant\n"]
        assert return_tensors == "pt"
        [image] = images
        assert image.mode == "RGB"
        width, height = image.size
        height, width = smart_resize(height, width, MIN_PIXELS, MAX_PIXELS)
        tokens = [1, 2, 3] + [IMAGE] * ((height // FACTOR) * (width // FACTOR)) + [4, 5]
        return {
            "input_ids": Tensor([tokens]),
            "attention_mask": Tensor([[1] * len(tokens)]),
            "pixel_values": Tensor([[0.0]], floating=True, meta={"gray": image.getpixel((0, 0))}),
            "image_grid_thw": Tensor([[1, height // 14, width // 14]]),
            "mm_token_type_ids": Tensor([[0] * len(tokens)]),
        }

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(i) for i in ids if i < PAD)


class Qwen2_5_VLForConditionalGeneration:
    def __init__(self, dtype):
        self.dtype = dtype
        self.generation_config = SimpleNamespace(eos_token_id=[END, PAD])

    @classmethod
    def from_pretrained(cls, path, dtype, device_map, local_files_only):
        assert local_files_only
        assert device_map in ("cpu", "cuda:0")
        _check(path)
        return cls(dtype)

    def eval(self):
        return self

    def generate(self, input_ids, pixel_values, image_grid_thw, max_new_tokens, **kw):
        assert kw.pop("repetition_penalty") > 0
        sampling = kw.get("temperature", 0) > 0
        assert kw.pop("do_sample") is sampling
        if sampling:  # vLLM's defaults: no top-p or top-k truncation
            assert (kw.pop("top_p"), kw.pop("top_k")) == (1.0, 0)
            kw.pop("temperature")
        assert kw.keys() == {"attention_mask", "mm_token_type_ids"}
        gray = pixel_values.meta["gray"][0]
        if gray == 200:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if gray == 100:
            tokens, number = [ord(c) for c in FRONT_MATTER.format(language="en") + "\n"], 0
            while len(tokens) < max_new_tokens:
                number += 1
                tokens += [ord(c) for c in newspaper_line(number) + "\n\n"]
        else:
            text = {
                255: PAGE,
                250: FRONT_MATTER.format(language="null"),
                150: "Plain text without front matter.\n",
            }.get(gray, "x")
            tokens = [ord(c) for c in text] + [END, PAD, PAD]
        return Tensor([input_ids.data[0] + tokens[:max_new_tokens]])
