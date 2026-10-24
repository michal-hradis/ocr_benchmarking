"""olmOCR-2 page-image to Markdown engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the model
from the /cache volume once (checked against models.sha256), scale every sample image so that its
longest side is 1288 pixels, generate its transcription in job order with the official prompt
(prompt.txt) and write $OCRBENCH_OUTPUT/<id>/ with the model's output unchanged (output.md: YAML
front matter and Markdown), the Markdown after the front matter (page.md, the primary output,
converted to plain text on the host) and, last, result.json with the front matter, the token
counts, the stop reason and the timing. The model gives no positions, so the prediction has none.
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

REPOSITORY = "allenai/olmOCR-2-7B-1025"
REVISION = "e52d6f090b7a9007afffbbd6ce510876222fea93"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_OLMOCR2_CACHE", "/cache"))
PINNED_DIR = f"models/olmocr2/olmOCR-2-7B-1025/{REVISION}"  # relative to the cache
CHECKSUMS = Path(os.environ.get("OCRBENCH_OLMOCR2_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# The model's whole context (max_position_embeddings): prompt plus generated tokens.
CONTEXT_TOKENS = 128000
# The official instruction (build_no_anchoring_v4_yaml_prompt in olmocr/prompts/prompts.py of
# olmocr v0.4.27), byte for byte; it precedes the image in one user turn, as in the pipeline.
PROMPT = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
CONVERSATION = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image"}]}]
# The front matter fields of olmocr's PageResponse, in the order the model writes them.
FRONT_MATTER = (
    "primary_language",
    "is_rotation_valid",
    "rotation_correction",
    "is_table",
    "is_diagram",
)
PATCH = 14  # one image token per 28 x 28 pixels (14-pixel patches, merged 2 x 2)
# Engine parameters (job.json "params"), README.md. The image size, max_new_tokens and the
# temperature are the olmocr pipeline's (its first attempt); the repetition penalty is the
# checkpoint's generation_config.json, which vLLM applies by default.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "model_dir": PINNED_DIR,
    "target_longest_image_dim": 1288,
    "max_new_tokens": 8000,
    "temperature": 0.1,
    "repetition_penalty": 1.05,
    "seed": 0,
}
INT_PARAMS = {"target_longest_image_dim": 56, "max_new_tokens": 1, "seed": 0}  # minimum


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
    temperature, penalty = merged["temperature"], merged["repetition_penalty"]
    if not _number(temperature) or temperature < 0:
        raise AdapterError(f"temperature must be a number of at least 0: {temperature!r}")
    if not _number(penalty) or penalty <= 0:
        raise AdapterError(f"repetition_penalty must be a number above 0: {penalty!r}")
    model_dir = merged["model_dir"]
    if not (isinstance(model_dir, str) and _relative(model_dir)):
        raise AdapterError(f"model_dir must be a relative path in the cache: {model_dir!r}")
    return merged


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


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
    """SHA-256 of the model files, and whether they are the official pinned checkpoint: every
    file of models.sha256 present and matching. The pinned directory must match."""
    directory = CACHE / model_dir
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make olmocr2-model and run with --cache (README.md)"
        )
    files = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()}
    pinned = dict(
        reversed(line.split()) for line in CHECKSUMS.read_text().splitlines() if line.strip()
    )
    bad = sorted(name for name, digest in pinned.items() if files.get(name) != digest)
    if bad and Path(model_dir) == Path(PINNED_DIR):
        raise AdapterError(f"model files missing or not matching models.sha256: {bad}")
    return files, not bad


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
    """The processor and the bfloat16 model (float32 on the CPU), from local files only. The
    16.6 GB of weights are loaded directly onto the device, not first into host memory."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = params["device"]
    dtype = torch.bfloat16 if device.startswith("cuda:") else torch.float32
    path = CACHE / params["model_dir"]
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, dtype=dtype, device_map=device, local_files_only=True
    )
    return processor, model.eval()


def read_image(path: Path, longest: int) -> tuple[Any, dict[str, int]]:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions),
    scaled with Lanczos resampling so that its longest side is ``longest`` pixels, as olmocr
    renders a page; returns it with its original size."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    image = image if image.mode == "RGB" else image.convert("RGB")
    width, height = image.size
    scale = longest / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image, {"width": width, "height": height}


def stop_tokens(model: Any) -> set[int]:
    eos = model.generation_config.eos_token_id
    return set(eos if isinstance(eos, list) else [eos])


def generate(processor: Any, model: Any, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """Generate one transcription; returns the decoded text and the token and timing details.

    Generation stops at an end token, at ``max_new_tokens`` or at the end of the model's context,
    whichever comes first; without an end token the output is truncated.
    """
    import torch

    device = params["device"]
    start = time.monotonic()
    prompt = processor.apply_chat_template(CONVERSATION, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
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
    # vLLM's sampling defaults: no top-p or top-k truncation (transformers' default top_k is 50).
    sampling: dict[str, Any] = {"do_sample": False}
    if params["temperature"] > 0:
        sampling = {
            "do_sample": True,
            "temperature": params["temperature"],
            "top_p": 1.0,
            "top_k": 0,
        }
    torch.manual_seed(params["seed"])
    prepared = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            repetition_penalty=params["repetition_penalty"],
            **sampling,
        )
    finished = time.monotonic()
    ids = output[0, len(prompt_ids) :].tolist()
    _, rows, columns = inputs["image_grid_thw"][0].tolist()  # patches, after resizing
    stops = stop_tokens(model)
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


def split_front_matter(output: str) -> tuple[dict[str, Any] | None, str, str | None]:
    """The front matter, the Markdown after it and a problem with the front matter, if any.

    The split is olmocr's FrontMatterParser: a front matter starts the output with ``---`` and
    ends at the next line starting with ``---``; without one, or if its YAML is invalid, the whole
    output is the text. The fields must be exactly those of olmocr's PageResponse.
    """
    import yaml

    if output.startswith("---\n"):
        end = output.find("\n---", 4)
        if end != -1:
            try:
                fields = yaml.safe_load(output[4:end]) or {}
            except yaml.YAMLError as error:
                return None, output.strip(), f"invalid YAML front matter: {error}"
            text = output[end + 4 :].strip()
            if not isinstance(fields, dict):
                return None, text, "front matter is not a mapping"
            fields = {str(k): v if _scalar(v) else str(v) for k, v in fields.items()}
            missing = [name for name in FRONT_MATTER if name not in fields]
            unexpected = sorted(set(fields) - set(FRONT_MATTER))
            problem = None
            if missing or unexpected:
                problem = f"front matter fields missing: {missing}, unexpected: {unexpected}"
            return fields, text, problem
    return None, output.strip(), "no front matter"


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int)) or _number(value)


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Transcribe one sample, write its outputs and return ``result.json`` content."""
    processor, model = models
    image, original = read_image(source / sample["image"], params["target_longest_image_dim"])
    generation = generate(processor, model, params, image)
    output = generation.pop("text")
    (out / "output.md").write_text(output, encoding="utf-8")
    front_matter, text, problem = split_front_matter(output)
    (out / "page.md").write_text(text, encoding="utf-8")
    generation |= {"source_image": original, "front_matter": front_matter}
    if problem:
        generation["front_matter_error"] = problem
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
        init_error = f"cannot load olmOCR-2: {type(error).__name__}: {error}"
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
    print(f"olmocr2 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
