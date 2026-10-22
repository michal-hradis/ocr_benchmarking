"""Infinity-Parser2-Flash page-image layout parser adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the model
from the /cache volume once (checked against models.sha256), parse every sample image in job order
with the official doc2json prompt (prompt.txt) and write $OCRBENCH_OUTPUT/<id>/ with the model's
output unchanged (output.txt), its layout elements with boxes in original pixels (layout.json),
the elements' text joined in model order (page.md, the primary output, converted to plain text on
the host) and, last, result.json with the token counts, the stop reason and the timing. The
prediction has no positions: the element boxes stay in layout.json.
"""

from __future__ import annotations

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

REPOSITORY = "infly/Infinity-Parser2-Flash"
REVISION = "cfddc4106b0abc4706f05575a164f7dd35d09c46"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_INFINITYPARSER2_CACHE", "/cache"))
PINNED_DIR = f"models/infinityparser2/Infinity-Parser2-Flash/{REVISION}"  # relative to the cache
CHECKSUMS = Path(
    os.environ.get("OCRBENCH_INFINITYPARSER2_CHECKSUMS", "/opt/ocrbench/models.sha256")
)
# The model's whole context (text_config.max_position_embeddings): prompt plus generated tokens.
CONTEXT_TOKENS = 262144
# The official doc2json instruction (PROMPT_DOC2JSON of infinity_parser2 0.4.0, the same as in the
# model card at REVISION), byte for byte; it follows the image in one user turn, without thinking.
PROMPT = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
CONVERSATION = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
# One image token per 32 x 32 pixels (16-pixel patches, merged 2 x 2). The official pipeline
# resizes the image itself (qwen_vl_utils.smart_resize) to between MIN_PIXELS and max_pixels, and
# rejects aspect ratios above MAX_RATIO.
PATCH, FACTOR, MIN_PIXELS, MAX_RATIO = 16, 32, 2048, 200
# Element boxes are [x1, y1, x2, y2] scaled to [0, 1000] of the image width and height.
BOX_SCALE = 1000
# Engine parameters (job.json "params"), README.md. Greedy decoding, max_new_tokens and
# max_pixels (4096 x 4096) follow infinity_parser2 0.4.0.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "model_dir": PINNED_DIR,
    "max_new_tokens": 32768,
    "max_pixels": 16777216,
}
INT_PARAMS = {"max_new_tokens": 1, "max_pixels": MIN_PIXELS}  # minimum


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
    transformers may load them (a generation_config.json changes the decoding). The pinned
    directory must match."""
    directory = CACHE / model_dir
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make infinityparser2-model and run with --cache (README.md)"
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
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = params["device"]
    dtype = torch.bfloat16 if device.startswith("cuda:") else torch.float32
    path = CACHE / params["model_dir"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=dtype, local_files_only=True
    )
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


def smart_resize(height: int, width: int, max_pixels: int) -> tuple[int, int]:
    """The model input size as in qwen_vl_utils 0.0.14 (smart_resize with factor 32), which the
    official pipeline uses: whole 32-pixel units, scaled into [MIN_PIXELS, max_pixels]."""
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise AdapterError(f"aspect ratio of {width} x {height} pixels is above {MAX_RATIO}")
    h_bar = max(FACTOR, round(height / FACTOR) * FACTOR)
    w_bar = max(FACTOR, round(width / FACTOR) * FACTOR)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h_bar = math.floor(height / beta / FACTOR) * FACTOR
        w_bar = math.floor(width / beta / FACTOR) * FACTOR
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def stop_tokens(processor: Any, model: Any) -> list[int]:
    """The tokenizer's end token (<|im_end|>, which ends the chat turn) and the language model's
    (<|endoftext|> in config.json); the checkpoint has no generation_config.json."""
    eos = model.config.text_config.eos_token_id
    return [processor.tokenizer.eos_token_id, *(eos if isinstance(eos, list) else [eos])]


def generate(processor: Any, model: Any, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """Generate one layout JSON greedily; returns the decoded text and the token and timing
    details.

    Generation stops at an end token, at ``max_new_tokens`` or at the end of the model's context,
    whichever comes first; without an end token the output is truncated.
    """
    import torch
    from PIL import Image

    device = params["device"]
    start = time.monotonic()
    height, width = smart_resize(image.height, image.width, params["max_pixels"])
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    prompt = processor.apply_chat_template(
        CONVERSATION, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    inputs = processor(text=[prompt], images=[resized], do_resize=False, return_tensors="pt")
    inputs.pop("token_type_ids", None)  # as the official transformers backend
    if device.startswith("cuda:"):  # the peak includes moving the inputs to the GPU
        torch.cuda.reset_peak_memory_stats(device)
    dtype = model.dtype
    inputs = {
        key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        for key, value in inputs.items()
    }
    prompt_ids = inputs["input_ids"][0].tolist()
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
        "model_image": {"width": width, "height": height},
        "preprocess_seconds": round(prepared - start, 3),
        "generate_seconds": round(finished - prepared, 3),
        "tokens_per_second": round(len(generated) / max(finished - prepared, 1e-9), 1),
    }
    if device.startswith("cuda:"):
        details["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) >> 20
    text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"text": text, **details}


def parse_layout(text: str) -> tuple[list[Any], str]:
    """The complete elements of the model's JSON array, and the text after the last of them that
    is not the end of the array ('' when the array is complete).

    As in infinity_parser2 (extract_json_content), a ```json fence is unwrapped, also an unclosed
    one; an element cut off by truncation (or anything malformed) ends the array, keeping the
    elements before it (truncate_last_incomplete_element), and is returned as the rest. Output
    that is not a JSON array at all raises AdapterError; an empty output has no elements.
    """
    fenced = re.search(r"```json\n(.*?)\n```", text, re.DOTALL) or re.search(
        r"```json\n(.*)", text, re.DOTALL
    )
    body = (fenced.group(1) if fenced else text).strip()
    if not body:
        return [], ""
    if not body.startswith("["):
        raise AdapterError(f"the output is not a layout JSON array: {body[:200]!r}")
    decoder = json.JSONDecoder()
    elements: list[Any] = []
    position = 1
    while True:
        position = _skip(body, position, ",") if elements else _skip(body, position, "")
        if body.startswith("]", position):
            return elements, body[position + 1 :].strip()
        try:
            element, position = decoder.raw_decode(body, position)
        except json.JSONDecodeError:
            return elements, body[position:]
        elements.append(element)


def _skip(text: str, position: int, separator: str) -> int:
    """The position after whitespace and at most one ``separator``."""
    while position < len(text) and text[position].isspace():
        position += 1
    if separator and text.startswith(separator, position):
        position += len(separator)
        while position < len(text) and text[position].isspace():
            position += 1
    return position


def layout_elements(elements: list[Any], width: int, height: int) -> list[dict[str, Any]]:
    """The elements with their box also in original image pixels (``bbox_px``, float pixel-edge
    coordinates, not clipped); raises AdapterError for an element that is not an object with a
    string (or null) text."""
    result = []
    for number, element in enumerate(elements):
        text = element.get("text") if isinstance(element, dict) else None
        if not isinstance(element, dict) or not isinstance(text, str | None):
            shown = json.dumps(element, ensure_ascii=False)[:200]
            raise AdapterError(f"layout element {number} is not an object with a text: {shown}")
        item = dict(element)
        box = element.get("bbox")
        if (
            isinstance(box, list)
            and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in box)
        ):
            x1, y1, x2, y2 = box
            item["bbox_px"] = [
                x1 * width / BOX_SCALE,
                y1 * height / BOX_SCALE,
                x2 * width / BOX_SCALE,
                y2 * height / BOX_SCALE,
            ]
        result.append(item)
    return result


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Parse one sample, write its outputs and return ``result.json`` content."""
    processor, model = models
    image = read_image(source / sample["image"])
    generation = generate(processor, model, params, image)
    text = generation.pop("text")
    (out / "output.txt").write_text(text, encoding="utf-8")
    parsed, rest = parse_layout(text)
    elements = layout_elements(parsed, image.width, image.height)
    layout = {"image": {"width": image.width, "height": image.height}, "elements": elements}
    (out / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), "utf-8")
    page = "\n\n".join(e["text"] for e in elements if e.get("text"))
    (out / "page.md").write_text(page, encoding="utf-8")
    generation |= {"elements": len(elements), "dropped_characters": len(rest)}
    result: dict[str, Any] = {"status": "ok", "primary": "page.md"}
    if generation["stop_reason"] != "end-token":
        limit = (
            "the model's context" if generation["stop_reason"] == "context" else "max_new_tokens"
        )
        result["status"] = "truncated"
        result["message"] = f"generation stopped at {limit} ({generation['max_new_tokens']} tokens)"
    elif rest:
        result["status"] = "partial"
        result["message"] = (
            f"malformed layout JSON after element {len(elements)}: {len(rest)} characters dropped"
        )
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
        init_error = f"cannot load Infinity-Parser2: {type(error).__name__}: {error}"
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
    print(f"infinityparser2 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
