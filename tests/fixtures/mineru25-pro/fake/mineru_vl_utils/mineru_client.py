"""Fake MinerUClient: two_step_extract returns committed real results, chosen by the image width,
and calls the model's generate once for the layout and once per block it reads, as the real
transformers backend does (README.md one level up)."""

import copy
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from torch import Tensor

from .transformers_loading import IMAGE

FIXTURES = Path(__file__).resolve().parents[2]
RESULTS = {900: "page.json", 520: "paragraph.json", 1000: "columns.json", 1500: "newspaper.json"}
CONTEXT = 8192  # text_config.max_position_embeddings: max_length without max_new_tokens
LAYOUT_IMAGE_TOKENS = 37 * 37  # the page scaled to 1036 x 1036 pixels
LAYOUT_TOKENS = 2  # generated tokens per detected block
NOT_READ = {"image", "chart", "list", "equation_block", "image_block"}
TASKS = ("table", "equation", "image", "chart", "[default]", "[layout]")
DEFAULT_SAMPLING_PARAMS = {
    task: SimpleNamespace(no_repeat_ngram_size=100, max_new_tokens=None) for task in TASKS
}


def newspaper_words():
    """A page whose only block never ends."""
    for number in itertools.count(1):
        yield from f"Line {number} of a very long newspaper page.".split()


class MinerUClient:
    def __init__(self, backend, model, processor, sampling_params, use_tqdm):
        assert backend == "transformers"
        assert use_tqdm is False
        assert set(sampling_params) == set(TASKS)
        self.model, self.sampling_params = model, sampling_params

    def _generate(self, task, image_tokens, tokens):
        """Generated token count and whether it ended; ``tokens`` None never ends."""
        self.model.planned = tokens
        prompt = [1, 2, 3] + [IMAGE] * image_tokens + [4, 5]
        params = self.sampling_params[task]
        limit = {"max_length": CONTEXT}
        if params.max_new_tokens is not None:
            limit = {"max_new_tokens": params.max_new_tokens}
        output = self.model.generate(
            input_ids=Tensor([prompt]),
            use_cache=True,
            do_sample=False,
            no_repeat_ngram_size=params.no_repeat_ngram_size,
            **limit,
        )
        generated = output.tolist()[0][len(prompt) :]
        ended = generated[-1] != 7
        return len(generated) - ended, ended

    def two_step_extract(self, image):
        width, height = image.size
        if width == 666:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (fake)")
        if width == 2000:
            blocks = [{"type": "text", "bbox": [0.05, 0.05, 0.95, 0.95], "angle": 0}]
            blocks[0] |= {"content": None, "merge_prev": False}
        else:
            name = RESULTS.get(width)
            blocks = json.loads((FIXTURES / name).read_text("utf-8")) if name else []
        count, ended = self._generate("[layout]", LAYOUT_IMAGE_TOKENS, LAYOUT_TOKENS * len(blocks))
        if not ended:
            blocks = blocks[: count // LAYOUT_TOKENS]
        for block in blocks:
            if block["type"] in NOT_READ:
                continue
            x1, y1, x2, y2 = block["bbox"]
            crop_tokens = max(4, round((x2 - x1) * width * (y2 - y1) * height / 784))
            words = (block["content"] or "").split() if width != 2000 else newspaper_words()
            task = block["type"] if block["type"] in self.sampling_params else "[default]"
            planned = len(words) if isinstance(words, list) else None
            count, ended = self._generate(task, crop_tokens, planned)
            if not ended:  # cut off: the words generated so far
                block["content"] = " ".join(itertools.islice(words, count))
        return [copy.deepcopy(block) for block in blocks]
