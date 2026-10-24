"""Fake DeepSeek-OCR-2 tokenizer and model (README.md one level up)."""

from pathlib import Path

from torch import Tensor, bfloat16, long

FIXTURES = Path(__file__).resolve().parents[2]
PAGE = (FIXTURES / "output.md").read_text(encoding="utf-8")
IMAGE, BOS, EOS = 128815, 0, 1
TEXT = "\n<|grounding|>Convert the document to markdown."


def newspaper_line(number):
    return (
        f"<|ref|>text<|/ref|><|det|>[[10, 10, 990, 20]]<|/det|>\nLine {number} of a long page.\n\n"
    )


def _check(path, trust_remote_code, local_files_only):
    assert trust_remote_code
    assert local_files_only
    if not (Path(path) / "config.json").is_file():
        raise OSError(f"{path} does not appear to have a file named config.json")


class AutoTokenizer:
    eos_token_id = EOS

    @classmethod
    def from_pretrained(cls, path, trust_remote_code, local_files_only):
        _check(path, trust_remote_code, local_files_only)
        return cls()

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return [1000 + ord(c) for c in text]  # one token per character, never IMAGE, BOS or EOS

    def decode(self, ids, skip_special_tokens):
        assert not skip_special_tokens  # the layout tags are special tokens
        return "".join(chr(i - 1000) for i in ids)


class AutoModel:
    @classmethod
    def from_pretrained(cls, path, **kw):
        _check(path, kw.pop("trust_remote_code"), kw.pop("local_files_only"))
        assert kw == {"use_safetensors": True, "_attn_implementation": "eager",
                      "torch_dtype": bfloat16}  # fmt: skip
        return cls()

    def eval(self):
        return self

    def to(self, device_or_dtype):
        assert device_or_dtype in ("cuda:0", "cuda:1", bfloat16)
        return self

    def generate(self, input_ids, images, images_seq_mask, images_spatial_crop, **kw):
        max_new_tokens = kw.pop("max_new_tokens")
        assert kw == {"do_sample": False, "eos_token_id": EOS, "use_cache": True,
                      "no_repeat_ngram_size": kw["no_repeat_ngram_size"]}  # fmt: skip
        [ids], [mask], [(crops, original)] = input_ids.data, images_seq_mask.data, images
        assert (input_ids.dtype, images_seq_mask.dtype, images_spatial_crop.dtype) == (
            long,
            "bool",
            long,
        )
        [[columns, rows]] = images_spatial_crop.data
        # BOS, the image tokens where the mask is set, then the text after <image>.
        assert ids[0] == BOS
        assert not mask[0]
        assert [i == IMAGE for i in ids] == mask
        assert "".join(chr(i - 1000) for i in ids[1 + sum(mask) :]) == TEXT
        [view] = original.meta["views"]
        assert view["size"] == (1024, 1024)
        tiles = crops.meta["views"]
        if tiles:
            assert sum(mask) == 16 * 16 + 1 + 12 * columns * 12 * rows
            assert len(tiles) == columns * rows
            assert {t["size"] for t in tiles} == {(768, 768)}
        else:  # infer()'s zero placeholder
            assert (columns, rows) == (1, 1)
            assert crops.meta["shape"] == (1, 3, 1024, 1024)
            assert sum(mask) == 16 * 16 + 1
        gray = view["gray"]
        if gray == 200:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if gray == 100:
            tokens, number = [], 0
            while len(tokens) < max_new_tokens:
                number += 1
                tokens += [1000 + ord(c) for c in newspaper_line(number)]
        else:
            text = {
                255: PAGE,
                250: "",
                150: "Plain text without layout tags.\n",
            }.get(gray, "x")
            tokens = [1000 + ord(c) for c in text] + [EOS]
        return Tensor([ids + tokens[:max_new_tokens]], long)
