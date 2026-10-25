"""HunyuanOCR-1.5 page-image to text/Markdown engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the model
from the /cache volume once (checked against models.sha256), and for every sample in job order
generate its transcription from the image as stored with one of the official fixed task prompts,
stopping early when the end of the output repeats itself as the official clients do. It writes
$OCRBENCH_OUTPUT/<id>/ with the model's output unchanged (output.md), the output with a repeated
tail trimmed (page.md, the primary output, converted to plain text on the host) and, last,
result.json with the token counts, the stop reason and the timing. The model gives no positions,
so the prediction has none.
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

REPOSITORY = "tencent/HunyuanOCR"
REVISION = "47644ecc4fc854efa4f505155158831f36773ee4"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_HUNYUANOCR_CACHE", "/cache"))
PINNED_DIR = f"models/hunyuanocr/HunyuanOCR/{REVISION}"  # relative to the cache
CHECKSUMS = Path(os.environ.get("OCRBENCH_HUNYUANOCR_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# The model's whole context (max_position_embeddings): prompt plus generated tokens.
CONTEXT_TOKENS = 131072
# The official task prompts (TASK_PROMPTS in inference/utils/tasks.py of
# github.com/Tencent-Hunyuan/HunyuanOCR at 1ef4179e53cf860f7e6fd8276a292e5d05b3e927), verbatim.
TASK_PROMPTS = {
    # End-to-end document parsing to Markdown; page headers and footers are left out.
    "doc_parse": (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    ),
    # "Extract the text in the image", for non-document scenes such as ancient texts.
    "structured_parse": "提取图中的文字。",
}
PATCH = 16  # one image token per 32 x 32 pixels (16-pixel patches, merged 2 x 2)
# The official clients' tail-repetition early stop: after 4,000 decoded characters and then every
# 1,000 more, the last 8,000 characters are checked for a unit of at most 256 characters repeated
# 8 times at the end; the decoded text is checked every 64 tokens.
REPEAT_START, REPEAT_STEP, REPEAT_WINDOW, REPEAT_TIMES, REPEAT_UNIT = 4000, 1000, 8000, 8, 256
REPEAT_PROBE_TOKENS = 64
# Their final clean-up: a tail of 2,000+ characters repeated 10 times is cut to one copy.
TRIM_MIN_CHARS, TRIM_TIMES = 2000, 10
# Engine parameters (job.json "params"), README.md. The decoding is the official clients'.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "model_dir": PINNED_DIR,
    "task": "doc_parse",
    "max_new_tokens": 32768,
    "repetition_penalty": 1.08,
    "repetition_stop": True,
}


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
    if merged["task"] not in TASK_PROMPTS:
        raise AdapterError(f"task must be one of {', '.join(TASK_PROMPTS)}: {merged['task']!r}")
    tokens = merged["max_new_tokens"]
    if type(tokens) is not int or not 1 <= tokens < CONTEXT_TOKENS:
        raise AdapterError(
            f"max_new_tokens must be an integer from 1 to below the context of {CONTEXT_TOKENS}: "
            f"{tokens!r}"
        )
    penalty = merged["repetition_penalty"]
    if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty <= 0:
        raise AdapterError(f"repetition_penalty must be a number above 0: {penalty!r}")
    if type(merged["repetition_stop"]) is not bool:
        raise AdapterError(f"repetition_stop must be true or false: {merged['repetition_stop']!r}")
    model_dir = merged["model_dir"]
    if not (isinstance(model_dir, str) and _relative(model_dir)):
        raise AdapterError(f"model_dir must be a relative path in the cache: {model_dir!r}")
    return merged


def _relative(value: str) -> bool:
    path = Path(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def conversation(task: str) -> list[dict[str, Any]]:
    """The official request: an empty system message, then the image before the instruction."""
    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": TASK_PROMPTS[task]}],
        },
    ]


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
            "make hunyuanocr-model and run with --cache (README.md)"
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
    """The processor (the PIL image processor, the official ``use_fast=False``) and the
    bfloat16 model (float32 on the CPU), from local files only, with PyTorch's SDPA attention."""
    import torch
    from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

    device = params["device"]
    dtype = torch.bfloat16 if device.startswith("cuda:") else torch.float32
    path = CACHE / params["model_dir"]
    processor = AutoProcessor.from_pretrained(path, backend="pil", local_files_only=True)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        path, dtype=dtype, device_map=device, attn_implementation="sdpa", local_files_only=True
    )
    return processor, model.eval()


def read_image(path: Path) -> tuple[Any, dict[str, int]]:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions), in
    RGB, as the official clients send it; the processor scales it. Returns it with its size."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    image = image if image.mode == "RGB" else image.convert("RGB")
    return image, {"width": image.width, "height": image.height}


def tail_repeats(text: str) -> bool:
    """Whether the text ends with a non-blank unit of at most REPEAT_UNIT characters repeated
    REPEAT_TIMES times in a row (the official ``has_tail_repetition``)."""
    for length in range(1, min(REPEAT_UNIT, len(text) // REPEAT_TIMES) + 1):
        unit = text[-length:]
        if unit.strip() and text.endswith(unit * REPEAT_TIMES):
            return True
    return False


def trim_repeated_tail(text: str) -> str:
    """Cut a tail unit repeated at least TRIM_TIMES times in a row to a single copy, in a text of
    at least TRIM_MIN_CHARS characters (the official ``clean_repeated_substrings``)."""
    if len(text) < TRIM_MIN_CHARS:
        return text
    for length in range(2, len(text) // TRIM_TIMES + 1):
        unit, count = text[-length:], 1
        while text[: len(text) - length * count].endswith(unit):
            count += 1
        if count >= TRIM_TIMES:
            return text[: len(text) - length * (count - 1)]
    return text


def repetition_stop(tokenizer: Any, prompt_length: int) -> Any:
    """A stopping criterion that ends generation when the decoded output repeats itself at the
    end, checked as the official clients do (REPEAT_* above); ``triggered`` records it."""
    from transformers import StoppingCriteria

    class TailRepetition(StoppingCriteria):
        def __init__(self) -> None:
            self.triggered = False
            self.probed = 0
            self.next_check = REPEAT_START

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            if self.triggered:
                return True
            generated = input_ids[0, prompt_length:].tolist()
            if len(generated) - self.probed < REPEAT_PROBE_TOKENS:
                return False
            self.probed = len(generated)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            if len(text) < self.next_check:
                return False
            self.next_check = len(text) + REPEAT_STEP
            self.triggered = tail_repeats(text[-REPEAT_WINDOW:])
            return self.triggered

    return TailRepetition()


def stop_tokens(model: Any) -> set[int]:
    eos = model.generation_config.eos_token_id
    return set(eos if isinstance(eos, list) else [eos])


def generate(processor: Any, model: Any, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """Generate one transcription; returns the decoded text and the token and timing details.

    Generation stops at an end token, when the output repeats itself, at ``max_new_tokens`` or at
    the end of the model's context, whichever comes first; only an end token is a complete output.
    """
    import torch

    device = params["device"]
    start = time.monotonic()
    prompt = processor.apply_chat_template(
        conversation(params["task"]), add_generation_prompt=True, tokenize=False
    )
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
    criteria = []
    if params["repetition_stop"]:
        criteria = [repetition_stop(processor.tokenizer, len(prompt_ids))]
    prepared = time.monotonic()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
            repetition_penalty=params["repetition_penalty"],
            stopping_criteria=criteria,
        )
    finished = time.monotonic()
    ids = output[0, len(prompt_ids) :].tolist()
    _, rows, columns = inputs["image_grid_thw"][0].tolist()  # patches, after resizing
    stops = stop_tokens(model)
    stopped = next((i for i, token in enumerate(ids) if token in stops), None)
    generated = ids if stopped is None else ids[:stopped]
    if stopped is not None:
        reason = "end-token"
    elif criteria and criteria[0].triggered:
        reason = "repetition"
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


STOPPED = {
    "repetition": "generation stopped because the output repeats itself",
    "max_new_tokens": "generation stopped at max_new_tokens",
    "context": "generation stopped at the model's context",
}


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Transcribe one sample, write its outputs and return ``result.json`` content."""
    processor, model = models
    image, original = read_image(source / sample["image"])
    generation = generate(processor, model, params, image)
    output = generation.pop("text")
    (out / "output.md").write_text(output, encoding="utf-8")
    text = trim_repeated_tail(output)
    (out / "page.md").write_text(text.strip(), encoding="utf-8")
    generation |= {"source_image": original, "trimmed_chars": len(output) - len(text)}
    reason = generation["stop_reason"]
    result: dict[str, Any] = {"status": "ok" if reason == "end-token" else "truncated"}
    if reason != "end-token":
        result["message"] = (
            f"{STOPPED[reason]} "
            f"({generation['generated_tokens']} of {generation['max_new_tokens']} tokens)"
        )
    return result | {"primary": "page.md", "extra": generation}


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
        init_error = f"cannot load HunyuanOCR-1.5: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    # The official repository and revision are claimed only for verified files; a custom
    # checkpoint is identified by its directory, its file hashes and the configured model name.
    model_info: dict[str, Any] = {"verified_official_checkpoint": official}
    if official:
        model_info |= {"repository": REPOSITORY, "revision": REVISION}
    model_info |= {"model_dir": params and params["model_dir"], "files": files}
    model_info["job_model"] = job.get("model")
    if params:
        prompt = TASK_PROMPTS[params["task"]]
        model_info |= {
            "prompt": conversation(params["task"]),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
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
    print(f"hunyuanocr adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
