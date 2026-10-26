"""PyLaia line recognizer process of the PyLaia engine; runs in the image's Python 3.10 venv.

The adapter (pylaia_engine.py, README.md) starts this process once with a JSON argument
(``model_dir``, ``device``, ``batch_size``, ``fixed_height``), and it loads the model, then:

1. writes one JSON line to stdout: ``{"ready": true, ...versions}`` or ``{"error": "..."}``;
2. for every request, reads one JSON line ``{"lines": [[width, height], ...]}`` from stdin
   followed by the lines' 8-bit grayscale pixels (row-major, concatenated), and writes one JSON
   line ``{"lines": [{"symbols": [...], "text": "...", "confidences": [...]}, ...]}`` in request
   order, or ``{"error": "..."}`` when the request fails.

Everything else (PyLaia's logging, warnings) goes to stderr. Recognition follows PyLaia's own
``pylaia-htr-decode-ctc`` path: ``ToImageTensor`` (grayscale, inverted, resized to the model's
fixed height, padded to the minimum valid width), ``PaddingCollater`` batches sorted by
descending width, the model, and greedy CTC decoding (``CTCGreedyDecoder``, temperature 1).
"""

from __future__ import annotations

import json
import os
import sys
from importlib import metadata
from typing import Any, BinaryIO

SPACE = "<space>"


def symbols_to_text(symbols: list[str]) -> str:
    """Line text as ``pylaia-htr-decode-ctc --decode.use_symbols true --decode.convert_spaces
    true --decode.join_string ""`` prints it: ``<space>`` becomes a space, other symbols
    (including ``<unk>``) are kept as they are, and the result is stripped."""
    return "".join(" " if symbol == SPACE else symbol for symbol in symbols).strip()


class Recognizer:
    def __init__(self, model_dir: str, device: str, batch_size: int, fixed_height: int) -> None:
        import torch
        from laia.common.loader import ModelLoader
        from laia.data.padding_collater import PaddingCollater, by_descending_width
        from laia.data.transforms.vision import ToImageTensor
        from laia.decoders import CTCGreedyDecoder
        from laia.engine import ImageFeeder
        from laia.utils import SymbolsTable

        if device.startswith("cuda:"):
            index = int(device.split(":", 1)[1])
            count = torch.cuda.device_count()
            if index >= count:
                raise RuntimeError(f"{device} requested but {count} CUDA devices are visible")
        self.torch = torch
        self.device = torch.device(device)
        loader = ModelLoader(model_dir, filename="model", device="cpu")
        model = loader.load_by(os.path.join(model_dir, "weights.ckpt"))
        if model is None:
            raise RuntimeError(f"no PyLaia model in {model_dir}")
        self.model = model.to(self.device).eval()
        self.syms = SymbolsTable(os.path.join(model_dir, "syms.txt"))
        if len(self.syms) != self.model.linear.out_features:
            raise RuntimeError(
                f"syms.txt has {len(self.syms)} symbols, the model "
                f"{self.model.linear.out_features} outputs"
            )
        self.batch_size = batch_size
        self.min_width = self.model.get_min_valid_image_size(4096)
        self.transform = ToImageTensor(
            invert=True, mode="L", fixed_height=fixed_height, min_width=self.min_width
        )
        self.collate = PaddingCollater({"img": (1, None, None)}, sort_key=by_descending_width)
        self.feed = ImageFeeder()
        self.decoder = CTCGreedyDecoder()
        self.info: dict[str, Any] = {
            "pylaia": metadata.version("pylaia"),
            "pylaia_torch": torch.__version__,
            "pylaia_cuda": torch.version.cuda,
            "symbols": len(self.syms),
            "min_width": self.min_width,
        }
        if self.device.type == "cuda":
            self.info["pylaia_gpu"] = torch.cuda.get_device_name(self.device)

    def recognize(self, images: list[Any]) -> list[dict[str, Any]]:
        items = [{"img": self.transform(image), "id": n} for n, image in enumerate(images)]
        results: list[dict[str, Any]] = [{}] * len(items)
        with self.torch.inference_mode():
            for start in range(0, len(items), self.batch_size):
                batch = self.collate(items[start : start + self.batch_size])
                images = self.feed(batch["img"])
                images = type(images)(*(tensor.to(self.device) for tensor in images))
                output = self.model(images)
                decoded = self.decoder(output)
                outputs = zip(batch["id"], decoded["hyp"], decoded["prob-htr-char"], strict=True)
                for n, hyp, probs in outputs:
                    symbols = [self.syms[value] for value in hyp]
                    results[n] = {
                        "symbols": symbols,
                        "text": symbols_to_text(symbols),
                        "confidences": [round(p, 6) for p in probs],
                    }
        return results


def read_request(stream: BinaryIO) -> list[Any] | None:
    from PIL import Image

    header = stream.readline()
    if not header:
        return None
    sizes = json.loads(header)["lines"]
    images = []
    for width, height in sizes:
        data = stream.read(width * height)
        if len(data) != width * height:
            raise EOFError("truncated request")
        images.append(Image.frombytes("L", (width, height), data))
    return images


def main() -> None:
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # PyLaia and Lightning print to stdout; keep it for the protocol

    def reply(message: dict[str, Any]) -> None:
        out.write(json.dumps(message, ensure_ascii=False) + "\n")
        out.flush()

    try:
        recognizer = Recognizer(**json.loads(sys.argv[1]))
    except Exception as error:
        reply({"error": f"cannot load PyLaia: {type(error).__name__}: {error}"})
        return
    reply({"ready": True, **recognizer.info})
    while True:
        try:
            images = read_request(sys.stdin.buffer)
        except Exception as error:  # the stream is out of sync: stop
            reply({"error": f"{type(error).__name__}: {error}"})
            return
        if images is None:
            return
        try:
            reply({"lines": recognizer.recognize(images)})
        except Exception as error:  # e.g. CUDA out of memory: fail this request only
            reply({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
