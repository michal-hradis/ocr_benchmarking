"""Fake model and processor loaders: they load only from a directory with config.json."""

from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

# <|im_end|> and <|endoftext|> (generation_config.json), <|image_pad|> (config.json).
IM_END, END_OF_TEXT, IMAGE = 151645, 151643, 151655
WORD = 7  # one generated token per word


class Model:
    """generate returns the prompt and the generated tokens, as transformers does. The client
    sets ``planned`` (None: never ends) before every call."""

    def __init__(self, device):
        self.device, self.dtype = device, "bfloat16"
        self.generation_config = SimpleNamespace(eos_token_id=[IM_END, END_OF_TEXT])
        self.config = SimpleNamespace(image_token_id=IMAGE)
        self.planned = 0

    def generate(self, input_ids, use_cache, do_sample, no_repeat_ngram_size, **limit):
        # MinerU's sampling: greedy, no repeated 100-grams; max_new_tokens, or else the context.
        assert use_cache is True
        assert do_sample is False
        assert no_repeat_ngram_size == 100
        [prompt] = input_ids.tolist()
        if "max_new_tokens" in limit:
            budget = limit.pop("max_new_tokens")
        else:
            budget = limit.pop("max_length") - len(prompt)
        assert not limit
        if self.planned is None or self.planned >= budget:
            generated = [WORD] * budget
        else:
            generated = [WORD] * self.planned + [IM_END]
        return Tensor([prompt + generated])


def load_transformers_model(model_path, *, device_map="auto"):
    if not (Path(model_path) / "config.json").is_file():
        raise OSError(f"{model_path} does not appear to have a file named config.json")
    return Model(device_map)


def load_transformers_processor(model_path):
    if not (Path(model_path) / "config.json").is_file():
        raise OSError(f"{model_path} does not appear to have a file named config.json")
    return SimpleNamespace(path=Path(model_path))
