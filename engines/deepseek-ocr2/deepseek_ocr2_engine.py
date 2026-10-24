"""DeepSeek-OCR-2 page-image to Markdown engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the model
and its remote code from the /cache volume once (checked against models.sha256), cut every sample
image into the official global view and local tiles, generate its Markdown with grounding in job
order with the official document prompt and write $OCRBENCH_OUTPUT/<id>/ with the model's output
unchanged (output.md, layout tags included), its grounded blocks with boxes in original pixels
(layout.json), the Markdown without the layout tags (page.md, the primary output, converted to
plain text on the host) and, last, result.json with the token counts, the stop reason and the
timing. The prediction has no positions: the block boxes stay in layout.json.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

REPOSITORY = "deepseek-ai/DeepSeek-OCR-2"
REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_DEEPSEEK_OCR2_CACHE", "/cache"))
PINNED_DIR = f"models/deepseek-ocr2/DeepSeek-OCR-2/{REVISION}"  # relative to the cache
CHECKSUMS = Path(os.environ.get("OCRBENCH_DEEPSEEK_OCR2_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# The language model's context (language_config.max_position_embeddings): prompt plus generated
# tokens, the official vLLM max_model_len.
CONTEXT_TOKENS = 8192
# The official document prompt. The model card's "...markdown. " loses its trailing space in
# format_messages (sft_format "plain"), which gives exactly PROMPT of the official vLLM config.
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
# Image views of infer(base_size=1024, image_size=768, crop_mode=True) in the checkpoint's
# modeling_deepseekocr2.py: a 1024-pixel global view and 2-6 local 768-pixel tiles.
BASE_SIZE, TILE_SIZE, MIN_TILES, MAX_TILES = 1024, 768, 2, 6
PATCH, DOWNSAMPLE = 16, 4  # one image token per 64 x 64 pixels of a view
MEAN = (0.5, 0.5, 0.5)  # image_mean and image_std of processor_config.json
IMAGE_TOKEN, BOS = 128815, 0
# Grounding: <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>, boxes scaled to [0, 999].
GROUNDING = re.compile(r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL)
BOX_SCALE = 999
IMAGE_LABEL = "image"
# Engine parameters (job.json "params"), README.md. max_new_tokens and no_repeat_ngram_size are
# those of infer() (not eval_mode), the decoding of the model card's example.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "model_dir": PINNED_DIR,
    "max_new_tokens": 8192,
    "no_repeat_ngram_size": 20,
}
INT_PARAMS = {"max_new_tokens": 1, "no_repeat_ngram_size": 0}  # minimum


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
    # The remote code moves tensors with .cuda(): it runs on a CUDA device only.
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
        raise AdapterError(f"device must be 'cuda:<n>': {device!r}")
    for key, low in INT_PARAMS.items():
        value = merged[key]
        if type(value) is not int or value < low:
            raise AdapterError(f"{key} must be an integer of at least {low}: {value!r}")
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
    the remote code is executed and could import them. The pinned directory must match."""
    directory = CACHE / model_dir
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make deepseek-ocr2-model and run with --cache (README.md)"
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
    index = int(device.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
    torch.cuda.set_device(index)  # the remote code's .cuda() means this device
    return {
        "transformers": metadata.version("transformers"),
        "tokenizers": metadata.version("tokenizers"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(index),
        "gpu_memory_mb": torch.cuda.get_device_properties(index).total_memory >> 20,
    }


def load_model(params: dict[str, Any]) -> tuple[Any, Any]:
    """The tokenizer and the bfloat16 model with the checkpoint's own (verified) code, from local
    files only. The language model uses eager attention: flash-attention 2 is not installed."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = CACHE / params["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        path,
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=True,
        _attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    # As the model card: every parameter and buffer in bfloat16 on the GPU.
    return tokenizer, model.eval().to(params["device"]).to(torch.bfloat16)


def read_image(path: Path) -> Any:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode == "RGB" else image.convert("RGB")


def tile_grid(width: int, height: int) -> tuple[int, int]:
    """Columns and rows of the local tiles: dynamic_preprocess and find_closest_aspect_ratio of
    modeling_deepseekocr2.py, the grid of MIN_TILES to MAX_TILES tiles closest to the aspect
    ratio; of equally close grids, a larger one wins when the image has enough pixels."""
    ratios = sorted(
        {
            (i, j)
            for n in range(MIN_TILES, MAX_TILES + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if MIN_TILES <= i * j <= MAX_TILES
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    aspect, best, best_diff = width / height, (1, 1), float("inf")
    for ratio in ratios:
        diff = abs(aspect - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff, best = diff, ratio
        elif (
            diff == best_diff and width * height > 0.5 * TILE_SIZE * TILE_SIZE * ratio[0] * ratio[1]
        ):
            best = ratio
    return best


def views(image: Any) -> tuple[Any, list[Any], tuple[int, int]]:
    """The global view, the local tiles and their grid, as infer() with crop_mode makes them.

    The global view is the whole image scaled into BASE_SIZE x BASE_SIZE, keeping the aspect
    ratio, and padded with gray (ImageOps.pad). An image larger than TILE_SIZE on either side is
    also stretched to the tile grid and cut into TILE_SIZE tiles; a smaller one has none.
    Both resizes are Pillow's bicubic, the default of the official calls.
    """
    from PIL import Image, ImageOps

    fill = tuple(int(x * 255) for x in MEAN)
    global_view = ImageOps.pad(image, (BASE_SIZE, BASE_SIZE), color=fill)
    if image.width <= TILE_SIZE and image.height <= TILE_SIZE:
        return global_view, [], (1, 1)
    columns, rows = tile_grid(image.width, image.height)
    resized = image.resize((TILE_SIZE * columns, TILE_SIZE * rows), Image.Resampling.BICUBIC)
    tiles = [
        resized.crop((c * TILE_SIZE, r * TILE_SIZE, (c + 1) * TILE_SIZE, (r + 1) * TILE_SIZE))
        for r in range(rows)
        for c in range(columns)
    ]
    return global_view, tiles, (columns, rows)


def prompt_ids(tokenizer: Any, grid: tuple[int, int]) -> tuple[list[int], list[bool]]:
    """The prompt's token ids and image mask, as infer(): BOS, the image tokens (the global
    view's 16 x 16 and a view separator, then 12 x 12 per tile), then the text after <image>."""
    before, after = PROMPT.split("<image>")
    base = math.ceil((BASE_SIZE // PATCH) / DOWNSAMPLE)
    local = math.ceil((TILE_SIZE // PATCH) / DOWNSAMPLE)
    image = base * base + 1
    if grid != (1, 1):
        image += local * grid[0] * local * grid[1]
    head = tokenizer.encode(before, add_special_tokens=False)
    tail = tokenizer.encode(after, add_special_tokens=False)
    ids = [BOS, *head, *[IMAGE_TOKEN] * image, *tail]
    mask = [False] * (1 + len(head)) + [True] * image + [False] * len(tail)
    return ids, mask


def generate(tokenizer: Any, model: Any, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """Generate one page greedily; returns the decoded text and the token and timing details.

    Generation stops at the end token, at ``max_new_tokens`` or at the end of the model's
    context, whichever comes first; without an end token the output is truncated.
    """
    import torch
    from torchvision import transforms

    device = params["device"]
    start = time.monotonic()
    global_view, tiles, grid = views(image)
    ids, mask = prompt_ids(tokenizer, grid)
    budget = min(params["max_new_tokens"], CONTEXT_TOKENS - len(ids))
    if budget < 1:
        raise AdapterError(f"the image needs {len(ids)} of {CONTEXT_TOKENS} context tokens")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, MEAN)])
    torch.cuda.reset_peak_memory_stats(device)  # the peak includes moving the inputs to the GPU
    original = torch.stack([transform(global_view).to(torch.bfloat16)]).to(device)
    if tiles:
        crops = torch.stack([transform(tile).to(torch.bfloat16) for tile in tiles]).to(device)
    else:  # infer()'s placeholder: all zeros means no local views
        crops = torch.zeros((1, 3, BASE_SIZE, BASE_SIZE)).to(device)
    stop = tokenizer.eos_token_id
    prepared = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        output = model.generate(
            torch.LongTensor(ids).unsqueeze(0).to(device),
            images=[(crops, original)],
            images_seq_mask=torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(device),
            images_spatial_crop=torch.tensor([list(grid)], dtype=torch.long),
            do_sample=False,
            eos_token_id=stop,
            max_new_tokens=budget,
            no_repeat_ngram_size=params["no_repeat_ngram_size"],  # 0: off
            use_cache=True,
        )
    finished = time.monotonic()
    generated = output[0, len(ids) :].tolist()
    stopped = next((i for i, token in enumerate(generated) if token == stop), None)
    if stopped is not None:
        generated, reason = generated[:stopped], "end-token"
    elif budget < params["max_new_tokens"]:
        reason = "context"
    else:
        reason = "max_new_tokens"
    details: dict[str, Any] = {
        "stop_reason": reason,
        "prompt_tokens": len(ids),
        "image_tokens": sum(mask),
        "generated_tokens": len(generated),
        "max_new_tokens": budget,
        "tiles": {"columns": grid[0], "rows": grid[1]} if tiles else None,
        "preprocess_seconds": round(prepared - start, 3),
        "generate_seconds": round(finished - prepared, 3),
        "tokens_per_second": round(len(generated) / max(finished - prepared, 1e-9), 1),
        "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) >> 20,
    }
    # Special tokens are kept: the layout tags are special tokens.
    return {"text": tokenizer.decode(generated, skip_special_tokens=False), **details}


def layout_blocks(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """The grounded blocks of the stripped output in order: the label, the boxes as written
    (``det``), their parsed values (``boxes``, lists of four numbers scaled to [0, 999], or
    ``null`` when ``det`` is not such a list), the same boxes in original image pixels
    (``boxes_px``, float pixel-edge coordinates, not clipped), the Markdown up to the next block
    (``text``) and the character offsets of the tag in the stripped output (``span``). The boxes
    are read as Python literals, never evaluated: no box is invented or computed."""
    matches = list(GROUNDING.finditer(text))
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block: dict[str, Any] = {"label": match.group(2), "det": match.group(3), "boxes": None}
        boxes = _boxes(match.group(3))
        if boxes is not None:
            block["boxes"] = boxes
            block["boxes_px"] = [
                [x1 * width / BOX_SCALE, y1 * height / BOX_SCALE, x2 * width / BOX_SCALE,
                 y2 * height / BOX_SCALE]
                for x1, y1, x2, y2 in boxes
            ]  # fmt: skip
        blocks.append(block | {"text": text[match.end() : end].strip(), "span": list(match.span())})
    return blocks


def _boxes(det: str) -> list[list[float]] | None:
    try:
        value = ast.literal_eval(det.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None
    if not isinstance(value, list) or not value:
        return None
    for box in value:
        if not (isinstance(box, list) and len(box) == 4 and all(_number(v) for v in box)):
            return None
    return value


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def markdown(text: str) -> str:
    """The official result.mmd of the stripped output (infer(save_results=True)): every image
    block's tag becomes a Markdown image reference, every other tag is removed and, if there was
    one, ``\\coloneqq`` and ``\\eqqcolon`` are written out."""
    matches = GROUNDING.findall(text)
    images = [m[0] for m in matches if f"<|ref|>{IMAGE_LABEL}<|/ref|>" in m[0]]
    others = [m[0] for m in matches if f"<|ref|>{IMAGE_LABEL}<|/ref|>" not in m[0]]
    for index, tag in enumerate(images):
        text = text.replace(tag, f"![](images/{index}.jpg)\n")
    for tag in others:
        text = text.replace(tag, "").replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    return text


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Read one sample, write its outputs and return ``result.json`` content."""
    tokenizer, model = models
    image = read_image(source / sample["image"])
    generation = generate(tokenizer, model, params, image)
    output = generation.pop("text")
    (out / "output.md").write_text(output, encoding="utf-8")
    stripped = output.strip()
    blocks = layout_blocks(stripped, image.width, image.height)
    layout = {"image": {"width": image.width, "height": image.height}, "blocks": blocks}
    (out / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), "utf-8")
    (out / "page.md").write_text(markdown(stripped), encoding="utf-8")
    generation["blocks"] = len(blocks)
    generation["image_blocks"] = sum(b["label"] == IMAGE_LABEL for b in blocks)
    generation["invalid_boxes"] = sum(b["boxes"] is None for b in blocks)
    result: dict[str, Any] = {"status": "ok", "primary": "page.md"}
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
        init_error = f"cannot load DeepSeek-OCR-2: {type(error).__name__}: {error}"
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
        "prompt": PROMPT,
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
    print(f"deepseek-ocr2 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
