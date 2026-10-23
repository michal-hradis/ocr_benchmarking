"""GLM-OCR compact page OCR engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the model
from the /cache volume once (checked against models.sha256), transcribe every whole sample image
in job order with the official text recognition prompt and write $OCRBENCH_OUTPUT/<id>/ with the
model's Markdown output unchanged (page.md, the primary output, converted to plain text on the
host) and, last, result.json with the token counts, the stop reason and the timing. The
prediction has no positions: the model generates text only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

REPOSITORY = "zai-org/GLM-OCR"
REVISION = "2e85a62840ccac27daa451df36c736c4636b8628"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_GLM_OCR_CACHE", "/cache"))
PINNED_DIR = f"models/glm-ocr/GLM-OCR/{REVISION}"  # relative to the cache
CHECKSUMS = Path(os.environ.get("OCRBENCH_GLM_OCR_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# The model's whole context (text_config.max_position_embeddings): prompt plus generated tokens.
CONTEXT_TOKENS = 131072
# The model card's text recognition prompt ("Prompt Limited": the model supports only its fixed
# task prompts), as in the transformers example and the official SDK. It follows the image in
# one user turn; the chat template is used with its defaults.
PROMPT = "Text Recognition:"
CONVERSATION = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
# One image token per 28 x 28 pixels (14-pixel patches, merged 2 x 2). The processor counts the
# image twice (two temporal frames) against its pixel range, preprocessor_config.json's
# shortest_edge (MIN_PIXELS) and longest_edge (max_pixels).
PATCH, MIN_PIXELS = 14, 12544
# Engine parameters (job.json "params"), README.md. Greedy decoding with the official SDK's
# max_tokens and repetition_penalty; max_pixels is the checkpoint's own (6,144 image tokens).
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "model_dir": PINNED_DIR,
    "max_new_tokens": 8192,
    "max_pixels": 9633792,
    "repetition_penalty": 1.1,
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
    penalty = merged["repetition_penalty"]
    if type(penalty) not in (int, float) or not 1.0 <= penalty <= 2.0:
        raise AdapterError(f"repetition_penalty must be a number from 1.0 to 2.0: {penalty!r}")
    merged["repetition_penalty"] = float(penalty)
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
            "make glm-ocr-model and run with --cache (README.md)"
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
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration

    device = params["device"]
    dtype = torch.bfloat16 if device.startswith("cuda:") else torch.float32
    path = CACHE / params["model_dir"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = GlmOcrForConditionalGeneration.from_pretrained(path, dtype=dtype, local_files_only=True)
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


def stop_tokens(model: Any) -> list[int]:
    """The end tokens of the checkpoint's generation_config.json: <|endoftext|> and <|user|>
    (the start of the next turn)."""
    eos = model.generation_config.eos_token_id
    return list(eos) if isinstance(eos, list) else [eos]


def generate(processor: Any, model: Any, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """Generate one transcription greedily; returns the decoded text and the token and timing
    details.

    Generation stops at an end token, at ``max_new_tokens`` or at the end of the model's context,
    whichever comes first; without an end token the output is truncated.
    """
    import torch

    device = params["device"]
    start = time.monotonic()
    prompt = processor.apply_chat_template(CONVERSATION, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[prompt],
        images=[image],
        size={"shortest_edge": MIN_PIXELS, "longest_edge": params["max_pixels"]},
        return_tensors="pt",
    )
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
    stops = stop_tokens(model)
    prepared = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
            repetition_penalty=params["repetition_penalty"],
            eos_token_id=stops,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    finished = time.monotonic()
    ids = output[0, len(prompt_ids) :].tolist()
    _, rows, columns = inputs["image_grid_thw"][0].tolist()  # patches, after resizing
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
        "model_image": {"width": int(columns) * PATCH, "height": int(rows) * PATCH},
        "preprocess_seconds": round(prepared - start, 3),
        "generate_seconds": round(finished - prepared, 3),
        "tokens_per_second": round(len(generated) / max(finished - prepared, 1e-9), 1),
    }
    if device.startswith("cuda:"):
        details["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) >> 20
    text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"text": text, **details}


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Transcribe one sample, write its outputs and return ``result.json`` content."""
    processor, model = models
    image = read_image(source / sample["image"])
    generation = generate(processor, model, params, image)
    (out / "page.md").write_text(generation.pop("text"), encoding="utf-8")
    truncated = generation["stop_reason"] != "end-token"
    result: dict[str, Any] = {"status": "truncated" if truncated else "ok", "primary": "page.md"}
    if truncated:
        limit = (
            "the model's context" if generation["stop_reason"] == "context" else "max_new_tokens"
        )
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
        init_error = f"cannot load GLM-OCR: {type(error).__name__}: {error}"
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
    print(f"glm-ocr adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
