"""Chandra OCR 2 page-image OCR adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the model
from the /cache volume once (checked against models.sha256), read every sample image in job order
with the official ocr_layout prompt (prompt.txt) and write $OCRBENCH_OUTPUT/<id>/ with the model's
HTML output unchanged (output.html), its top-level layout blocks with boxes in original pixels
(layout.json), the output without its Image and Figure blocks (page.html, the primary output,
converted to plain text on the host) and, last, result.json with the token counts, the stop reason
and the timing. The prediction has no positions: the block boxes stay in layout.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from html.parser import HTMLParser
from importlib import metadata
from pathlib import Path
from typing import Any

REPOSITORY = "datalab-to/chandra-ocr-2"
REVISION = "af93b47dba1b47b6640c86ccf487ed2260ab9a09"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_CHANDRA_OCR2_CACHE", "/cache"))
PINNED_DIR = f"models/chandra-ocr2/chandra-ocr-2/{REVISION}"  # relative to the cache
CHECKSUMS = Path(os.environ.get("OCRBENCH_CHANDRA_OCR2_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# The model's whole context (text_config.max_position_embeddings): prompt plus generated tokens.
CONTEXT_TOKENS = 262144
# The official ocr_layout instruction (OCR_LAYOUT_PROMPT of chandra-ocr 0.2.0), byte for byte; it
# follows the image in one user turn. The chat template always closes an empty thinking block.
PROMPT = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
CONVERSATION = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
# chandra.model.util.scale_to_fit: at most 3072 x 2048 pixels, at least 1792 x 28, in whole units
# of 28 pixels. The processor then rounds to its own 32-pixel units (16-pixel patches, 2 x 2).
MAX_SIZE, MIN_SIZE, GRID = (3072, 2048), (1792, 28), 28
PATCH = 16
# Block boxes (data-bbox) are "x0 y0 x1 y1" scaled to [0, 1000] of the image width and height.
BOX_SCALE = 1000
# Blocks left out of the scored text: the model describes images in them (as the prompt asks),
# which is not text on the page. chandra.output.parse_html(include_images=False) drops the same.
IMAGE_LABELS = {"Image", "Figure"}
# Engine parameters (job.json "params"), README.md. max_new_tokens (MAX_OUTPUT_TOKENS) and
# min_image_dim (MIN_IMAGE_DIM, the CLI's upscaling of image files) follow chandra-ocr 0.2.0.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "model_dir": PINNED_DIR,
    "max_new_tokens": 12384,
    "min_image_dim": 1536,
}
INT_PARAMS = {"max_new_tokens": 1, "min_image_dim": 0}  # minimum


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {
        **DEFAULT_PARAMS,
        **{key: value for key, value in params.items() if value is not None},
    }
    device = merged["device"]
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+|cpu", device):
        raise AdapterError(f"device must be 'cuda:<n>' or 'cpu': {device!r}")
    for key, low in INT_PARAMS.items():
        value = merged[key]
        if type(value) is not int or value < low:
            raise AdapterError(f"{key} must be an integer of at least {low}: {value!r}")
    if merged["max_new_tokens"] >= CONTEXT_TOKENS:
        raise AdapterError(f"max_new_tokens must be less than the context of {CONTEXT_TOKENS}")
    model_dir = merged["model_dir"]
    if not (isinstance(model_dir, str) and _relative(model_dir)):
        raise AdapterError(f"model_dir must be a relative path in the cache: {model_dir!r}")
    return merged


def _relative(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(model_dir: str) -> tuple[dict[str, str], bool]:
    """SHA-256 of the model files (subdirectories included), and whether they are the official
    pinned checkpoint: exactly the files of models.sha256, all matching. Extra files count, since
    transformers may load them. The pinned directory must match."""
    directory = CACHE / model_dir
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make chandra-ocr2-model and run with --cache (README.md)"
        )
    files = {
        p.relative_to(directory).as_posix(): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    pinned = dict(
        reversed(line.split()) for line in CHECKSUMS.read_text().splitlines() if line.strip()
    )
    bad = sorted(name for name, digest in pinned.items() if files.get(name) != digest)
    extra = sorted(set(files) - set(pinned))
    if Path(model_dir) == Path(PINNED_DIR):
        if bad:
            raise AdapterError(f"model files missing or not matching models.sha256: {bad}")
        if extra:
            raise AdapterError(f"model files not in models.sha256: {extra}")
    return files, not bad and not extra


def environment(device: str) -> dict[str, Any]:
    """Versions and GPU diagnostics; raises AdapterError when the requested GPU is unusable."""
    try:
        import torch
        import transformers  # noqa: F401
    except Exception as error:
        raise AdapterError(f"cannot import transformers/PyTorch: {error}") from error
    info: dict[str, Any] = {
        "transformers": metadata.version("transformers"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
    }
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        count = torch.cuda.device_count()
        if index >= count:
            raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
        info["gpu"] = torch.cuda.get_device_name(index)
        info["gpu_memory_mb"] = torch.cuda.get_device_properties(index).total_memory >> 20
    return info


def load_model(params: dict[str, Any]) -> tuple[Any, Any]:
    """The processor and the bfloat16 model (float32 on the CPU), from local files only."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = params["device"]
    dtype = torch.bfloat16 if device.startswith("cuda:") else torch.float32
    path = CACHE / params["model_dir"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.padding_side = "left"  # as chandra.model.hf.load_model
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, local_files_only=True)
    return processor, model.to(device).eval()


def read_image(path: Path) -> Any:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode == "RGB" else image.convert("RGB")


def upscale(image: Any, min_dim: int) -> Any:
    """chandra.input.load_image: an image with a side below ``min_dim`` is scaled up (Lanczos)
    so that its shorter side is ``min_dim``; 0 turns this off."""
    from PIL import Image

    if image.width < min_dim or image.height < min_dim:
        scale = min_dim / min(image.width, image.height)
        size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def fit_size(width: int, height: int) -> tuple[int, int]:
    """The size chandra.model.util.scale_to_fit of chandra-ocr 0.2.0 resizes to: whole units of
    GRID pixels, scaled into the pixel counts of MIN_SIZE and MAX_SIZE, keeping the aspect ratio
    as well as possible."""
    original_ar = width / height
    pixels = width * height
    max_pixels = MAX_SIZE[0] * MAX_SIZE[1]
    min_pixels = MIN_SIZE[0] * MIN_SIZE[1]
    scale = 1.0
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
    elif pixels < min_pixels:
        scale = (min_pixels / pixels) ** 0.5
    w_blocks = max(1, round((width * scale) / GRID))
    h_blocks = max(1, round((height * scale) / GRID))
    while w_blocks * h_blocks * GRID * GRID > max_pixels:
        if w_blocks == 1 and h_blocks == 1:
            break
        if w_blocks == 1:
            h_blocks -= 1
            continue
        if h_blocks == 1:
            w_blocks -= 1
            continue
        ar_w_loss = abs(((w_blocks - 1) / h_blocks) - original_ar)
        ar_h_loss = abs((w_blocks / (h_blocks - 1)) - original_ar)
        if ar_w_loss < ar_h_loss:
            w_blocks -= 1
        else:
            h_blocks -= 1
    return w_blocks * GRID, h_blocks * GRID


def prepare_image(image: Any, min_dim: int) -> Any:
    """The image as the official CLI gives it to the processor: upscaled, then scaled to fit
    (Lanczos), unchanged if already of that size."""
    from PIL import Image

    image = upscale(image, min_dim)
    size = fit_size(image.width, image.height)
    if size == image.size:
        return image
    return image.resize(size, resample=Image.Resampling.LANCZOS)


def stop_tokens(processor: Any, model: Any) -> list[int]:
    """generation_config's end token (<|endoftext|>) and <|im_end|>, which ends the chat turn, as
    chandra.model.hf.generate_hf."""
    eos = model.generation_config.eos_token_id
    stops = list(eos) if isinstance(eos, list) else [eos]
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end not in stops:
        stops.append(im_end)
    return stops


def generate(processor: Any, model: Any, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """Generate one page of layout HTML greedily; returns the decoded text and the token and
    timing details.

    Generation stops at an end token, at ``max_new_tokens`` or at the end of the model's context,
    whichever comes first; without an end token the output is truncated.
    """
    import torch

    device = params["device"]
    start = time.monotonic()
    scaled = prepare_image(image, params["min_image_dim"])
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "image": scaled}, {"type": "text", "text": PROMPT}],
        }
    ]
    # The processor resizes the image again, to whole 32-pixel units (Qwen2-VL smart_resize).
    inputs = processor.apply_chat_template(
        [conversation],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    if device.startswith("cuda:"):  # the peak includes moving the inputs to the GPU
        torch.cuda.reset_peak_memory_stats(device)
    dtype = model.dtype
    inputs = {
        key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        for key, value in dict(inputs).items()
    }
    prompt_ids = inputs["input_ids"][0].tolist()
    _, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
    budget = min(params["max_new_tokens"], CONTEXT_TOKENS - len(prompt_ids))
    if budget < 1:
        raise AdapterError(f"the image needs {len(prompt_ids)} of {CONTEXT_TOKENS} context tokens")
    stops = stop_tokens(processor, model)
    prepared = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    finished = time.monotonic()
    ids = output[0, len(prompt_ids) :].tolist()
    stopped = next((i for i, token in enumerate(ids) if token in stops), None)
    generated = ids if stopped is None else ids[:stopped]
    if stopped is not None:
        reason = "end-token"
    elif budget < params["max_new_tokens"]:
        reason = "context"
    else:
        reason = "max_new_tokens"
    details: dict[str, Any] = {
        "stop_reason": reason,
        "prompt_tokens": len(prompt_ids),
        "image_tokens": prompt_ids.count(processor.image_token_id),
        "generated_tokens": len(generated),
        "max_new_tokens": budget,
        "scaled_image": {"width": scaled.width, "height": scaled.height},
        "model_image": {"width": grid_w * PATCH, "height": grid_h * PATCH},
        "preprocess_seconds": round(prepared - start, 3),
        "generate_seconds": round(finished - prepared, 3),
        "tokens_per_second": round(len(generated) / max(finished - prepared, 1e-9), 1),
    }
    if device.startswith("cuda:"):
        details["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) >> 20
    text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"text": text, **details}


class _Blocks(HTMLParser):
    """The top-level <div> blocks of the output: their attributes, inner HTML and the character
    offsets of the whole block, tags included (``span``). Only <div> tags are counted, so other
    unclosed tags cannot hide a block; a block still open at the end of the output (cut off) is
    kept as incomplete and runs to the end."""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.lines = [0]  # the offset of every line as HTMLParser counts them (by "\n" only)
        for line in text.split("\n")[:-1]:
            self.lines.append(self.lines[-1] + len(line) + 1)
        self.text, self.depth, self.blocks = text, 0, []
        self.feed(text)
        self.close()
        if self.depth:
            self.blocks[-1]["content"] = text[self.blocks[-1].pop("start") :]
            self.blocks[-1]["span"][1] = len(text)
            self.blocks[-1]["complete"] = False

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "div":
            return
        if not self.depth:
            values = dict(attrs)
            block = {"label": values.get("data-label"), "bbox": values.get("data-bbox")}
            offset = self._offset()
            start = offset + len(self.get_starttag_text())
            self.blocks.append(block | {"start": start, "span": [offset, None]})
        self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag != "div" or not self.depth:
            return
        self.depth -= 1
        if not self.depth:
            block, offset = self.blocks[-1], self._offset()
            block["content"] = self.text[block.pop("start") : offset]
            block["span"][1] = self.text.index(">", offset) + 1
            block["complete"] = True


def layout_blocks(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """The top-level blocks with their label, the model's box (``bbox``, a list of four integers
    as chandra.output.parse_layout reads it, or the attribute as written when it is not one), the
    box in original image pixels (``bbox_px``, float pixel-edge coordinates, not clipped), the
    inner HTML, its character offsets in the output and whether the block was closed. A block
    without a valid box has no ``bbox_px``: no box is invented."""
    result = []
    for block in _Blocks(text).blocks:
        item = dict(block)
        try:
            box = [int(v) for v in (block["bbox"] or "").split(" ")]
        except ValueError:
            box = []
        if len(box) == 4:
            x0, y0, x1, y1 = box
            item["bbox"] = box
            item["bbox_px"] = [
                x0 * width / BOX_SCALE,
                y0 * height / BOX_SCALE,
                x1 * width / BOX_SCALE,
                y1 * height / BOX_SCALE,
            ]
        result.append(item)
    return result


def without_images(text: str, blocks: list[dict[str, Any]]) -> str:
    """The output without its top-level IMAGE_LABELS blocks, otherwise unchanged."""
    kept, position = [], 0
    for block in blocks:
        if block["label"] in IMAGE_LABELS:
            start, end = block["span"]
            kept.append(text[position:start])
            position = end
    return "".join(kept) + text[position:]


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Read one sample, write its outputs and return ``result.json`` content."""
    processor, model = models
    image = read_image(source / sample["image"])
    generation = generate(processor, model, params, image)
    text = generation.pop("text")
    (out / "output.html").write_text(text, encoding="utf-8")
    blocks = layout_blocks(text, image.width, image.height)
    layout = {"image": {"width": image.width, "height": image.height}, "blocks": blocks}
    (out / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), "utf-8")
    (out / "page.html").write_text(without_images(text, blocks), encoding="utf-8")
    generation["blocks"] = len(blocks)
    generation["image_blocks"] = sum(b["label"] in IMAGE_LABELS for b in blocks)
    result: dict[str, Any] = {"status": "ok", "primary": "page.html"}
    if generation["stop_reason"] != "end-token":
        limit = (
            "the model's context" if generation["stop_reason"] == "context" else "max_new_tokens"
        )
        result["status"] = "truncated"
        result["message"] = f"generation stopped at {limit} ({generation['max_new_tokens']} tokens)"
    return result | {"extra": generation}


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    models, params, info, files, official, init_error = None, None, {}, None, False, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        files, official = model_files(params["model_dir"])
        models = load_model(params)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load Chandra OCR 2: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    # The official repository and revision are claimed only for verified files; a custom
    # checkpoint is identified by its directory, its file hashes and the configured model name.
    model_info: dict[str, Any] = {"verified_official_checkpoint": official}
    if official:
        model_info |= {"repository": REPOSITORY, "revision": REVISION}
    model_info |= {
        "model_dir": params and params["model_dir"],
        "files": files,
        "job_model": job.get("model"),
        "prompt": CONVERSATION,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
    }
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(models, params, sample, source, out)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # e.g. CUDA out of memory: fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result.setdefault("extra", {}).update(environment=info, models=model_info, params=params)
        write_result(out, result)
    print(f"chandra-ocr2 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
