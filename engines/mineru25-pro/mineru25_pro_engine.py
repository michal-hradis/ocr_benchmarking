"""MinerU2.5-Pro coarse-to-fine page parsing engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the
MinerU2.5-Pro-2605-1.2B model from the /cache volume once (checked against models.sha256) and run
the official two-step extraction of mineru-vl-utils on every sample image in job order: layout
detection on the whole page scaled to 1036 x 1036 pixels, then recognition of every layout block
cropped from the original image. Writes $OCRBENCH_OUTPUT/<id>/ with the native blocks
(mineru.json), the official Markdown (page.md), the standardized prediction.json and, last,
result.json. Every block becomes a region in the model's reading order with its box mapped to
input-image pixels; there are no line positions. A sample is truncated when any generation (the
layout or a block) stops without its end token.
"""

from __future__ import annotations

import copy
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

REPOSITORY = "opendatalab/MinerU2.5-Pro-2605-1.2B"
REVISION = "08aaea840498d49ce16247b6263196cf02814885"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_MINERU25_PRO_CACHE", "/cache"))
PINNED_DIR = f"models/mineru25-pro/MinerU2.5-Pro-2605-1.2B/{REVISION}"  # relative to the cache
CHECKSUMS = Path(os.environ.get("OCRBENCH_MINERU25_PRO_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# Engine parameters (job.json "params"), README.md. max_new_tokens None keeps mineru-vl-utils'
# default: every generation may run to the end of the model's context.
DEFAULT_PARAMS: dict[str, Any] = {"device": "cuda:0", "max_new_tokens": None}
# MinerU block type: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "text": "text",
    "code": "text",
    "algorithm": "text",
    "aside_text": "text",
    "ref_text": "text",
    "index": "text",
    "phonetic": "text",
    "list_item": "list-item",
    "title": "heading",
    "doc_title": "heading",
    "paragraph_title": "heading",
    "caption": "caption",
    "table_caption": "caption",
    "image_caption": "caption",
    "code_caption": "caption",
    "footnote": "footnote",
    "table_footnote": "footnote",
    "image_footnote": "footnote",
    "page_footnote": "footnote",
    "header": "page-header",
    "footer": "page-footer",
    "page_number": "page-number",
    "table": "table",
    "equation": "formula",
    "equation_block": "formula",
    "formula_number": "formula",
    "image": "figure",
    "chart": "figure",
    "image_block": "figure",
}


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    device = merged["device"]
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
        raise AdapterError(f"device must be 'cuda:<n>' (the image is a CUDA build): {device!r}")
    limit = merged["max_new_tokens"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise AdapterError(f"max_new_tokens must be an integer of at least 1: {limit!r}")
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model() -> dict[str, str]:
    """SHA-256 of the files of the pinned model directory, which must hold exactly the files of
    models.sha256, all matching: transformers may load other files it finds there."""
    directory = CACHE / PINNED_DIR
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make mineru25-pro-model and run with --cache (README.md)"
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
    if bad:
        raise AdapterError(f"model files missing or not matching models.sha256: {bad}")
    extra = sorted(set(files) - set(pinned))
    if extra:
        raise AdapterError(f"model files not in models.sha256: {extra}")
    return files


def environment(device: str) -> dict[str, Any]:
    """Versions and GPU diagnostics; raises AdapterError when the requested GPU is unusable."""
    try:
        import mineru_vl_utils
        import torch
        import transformers  # noqa: F401
    except Exception as error:
        raise AdapterError(f"cannot import mineru-vl-utils or PyTorch: {error}") from error
    info: dict[str, Any] = {
        "mineru_vl_utils": mineru_vl_utils.__version__,
        "transformers": metadata.version("transformers"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": device,
    }
    index = int(device.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
    info["gpu"] = torch.cuda.get_device_name(index)
    info["gpu_memory_mb"] = torch.cuda.get_device_properties(index).total_memory >> 20
    return info


class Generations:
    """Records every generation of the model: prompt image tokens, generated tokens, whether it
    ended and how long it took.

    mineru-vl-utils' transformers client removes the end tokens before decoding, so the adapter
    wraps the model's ``generate`` to see whether a generation stopped at its token limit without
    an end token (the output is then cut off). The first call of a page is layout detection, the
    others read one block each.
    """

    def __init__(self, model: Any, stop_ids: list[int], image_token_id: int) -> None:
        self.calls: list[dict[str, Any]] = []
        generate = model.generate

        def recording(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            output = generate(*args, **kwargs)
            rows = output.tolist()  # prompt and generated tokens, one row per crop
            seconds = (time.monotonic() - start) / len(rows)
            limit = kwargs.get("max_new_tokens")
            for prompt, ids in zip(kwargs["input_ids"].tolist(), rows, strict=True):
                generated = ids[len(prompt) :]
                stop = next((i for i, token in enumerate(generated) if token in stop_ids), None)
                if stop is not None:
                    reason = "end-token"
                elif limit is not None and len(generated) >= limit:
                    reason = "max_new_tokens"
                else:
                    reason = "context"
                self.calls.append(
                    {
                        "image_tokens": prompt.count(image_token_id),
                        "generated_tokens": len(generated) if stop is None else stop,
                        "stop_reason": reason,
                        "seconds": seconds,
                    }
                )
            return output

        model.generate = recording

    def summary(self) -> dict[str, Any]:
        if not self.calls:
            raise AdapterError("MinerU ran no layout detection")
        layout, *blocks = self.calls
        tokens = [call["generated_tokens"] for call in blocks]
        return {
            "layout": {
                "image_tokens": layout["image_tokens"],
                "generated_tokens": layout["generated_tokens"],
                "stop_reason": layout["stop_reason"],
                "seconds": round(layout["seconds"], 3),
            },
            "vl_calls": len(blocks),
            "generated_tokens": sum(tokens),
            "max_generated_tokens": max(tokens, default=0),
            "image_tokens": sum(call["image_tokens"] for call in blocks),
            "max_image_tokens": max((call["image_tokens"] for call in blocks), default=0),
            "truncated_blocks": sum(1 for call in blocks if call["stop_reason"] != "end-token"),
            "recognition_seconds": round(sum(call["seconds"] for call in blocks), 3),
        }


def load_client(params: dict[str, Any]) -> tuple[Any, Generations]:
    """The mineru-vl-utils client with the transformers backend and the model from local files.

    The model and processor are loaded by mineru-vl-utils' own loaders (they fix the tied output
    embeddings for transformers 5 and select the torchvision image processor), in the checkpoint's
    bfloat16, on ``device``. Prompts and sampling are the client's defaults; only ``max_new_tokens``
    is set on every task when given.
    """
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.mineru_client import DEFAULT_SAMPLING_PARAMS
    from mineru_vl_utils.transformers_loading import (
        load_transformers_model,
        load_transformers_processor,
    )

    path = CACHE / PINNED_DIR
    model = load_transformers_model(path, device_map=params["device"])
    processor = load_transformers_processor(path)
    sampling = DEFAULT_SAMPLING_PARAMS
    if params["max_new_tokens"] is not None:
        sampling = {}
        for task, task_params in DEFAULT_SAMPLING_PARAMS.items():
            sampling[task] = copy.copy(task_params)
            sampling[task].max_new_tokens = params["max_new_tokens"]
    client = MinerUClient(
        backend="transformers",
        model=model,
        processor=processor,
        sampling_params=sampling,
        use_tqdm=False,
    )
    eos = model.generation_config.eos_token_id
    stop_ids = list(eos) if isinstance(eos, list) else [eos]
    return client, Generations(model, stop_ids, model.config.image_token_id)


def read_image(path: Path) -> Any:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode == "RGB" else image.convert("RGB")


class _TableText(HTMLParser):
    """Rows of an HTML table as lines, cells separated by tabs, as the host converter does."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append([])
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self.cell is not None:
            self.rows[-1].append(" ".join("".join(self.cell).split()))
            self.cell = None

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)


def table_text(content: str) -> str:
    """Plain text of a table block: mineru-vl-utils converts the model's OTSL table to HTML;
    content that is not an HTML table is kept."""
    if "<table" not in content.lower():
        return content.strip()
    parser = _TableText()
    parser.feed(content)
    parser.close()
    return "\n".join("\t".join(row) for row in parser.rows if any(row))


def formula_text(content: str) -> str:
    """TeX source of an equation block, without the ``\\[ … \\]`` (or ``$$``) delimiters that
    mineru-vl-utils adds."""
    text = content.strip()
    for start, end in (("\\[", "\\]"), ("$$", "$$")):
        if len(text) >= len(start) + len(end) and text.startswith(start) and text.endswith(end):
            return text[len(start) : -len(end)].strip()
    return text


def block_text(block_type: str, content: str) -> str:
    if block_type == "table":
        return table_text(content)
    if block_type == "equation":
        return formula_text(content)
    return content.strip()


def blocks_to_prediction(
    blocks: list[dict[str, Any]], sample_id: str, size: tuple[int, int]
) -> dict[str, Any]:
    """Standardized prediction (docs/PREDICTIONS.md) from the blocks of one page.

    Every block becomes a region in the order the layout step produced them, which is the model's
    reading order. Its box is relative to the page (0–1, the layout model sees the whole page
    scaled to a square) and is mapped to input-image pixels. Its text is the block's recognized
    content after mineru-vl-utils' post-processing, in plain text: an HTML table becomes one line
    per row with tab-separated cells, an equation loses its delimiters. Blocks without text
    (images, containers) stay regions without text. A text block that continues the previous one
    (``merge_prev``) stays a region of its own: the flag is kept in extra.
    """
    width, height = size
    regions = []
    for number, block in enumerate(blocks):
        x1, y1, x2, y2 = block["bbox"]
        x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
        region: dict[str, Any] = {
            "id": f"b{number + 1}",
            "type": REGION_TYPES.get(block["type"], "other"),
            "native_type": block["type"],
            "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        }
        text = block_text(block["type"], block.get("content") or "")
        if text:
            region["text"] = text
        extra = {"angle": block.get("angle")}
        for key in ("merge_prev", "sub_type"):
            if key in block:
                extra[key] = block[key]
        region["extra"] = extra
        regions.append(region)
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": width, "height": height},
        "text": "\n\n".join(region["text"] for region in regions if "text" in region),
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    return prediction


def process(
    models: tuple[Any, Generations],
    params: dict[str, Any],
    sample: dict[str, Any],
    source: Path,
    out: Path,
) -> dict[str, Any]:
    """Parse one sample, write its outputs and return the ``result.json`` content."""
    import torch
    from mineru_vl_utils.post_process import json2md

    client, generations = models
    image = read_image(source / sample["image"])
    generations.calls.clear()
    device = params["device"]
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    blocks = [dict(block) for block in client.two_step_extract(image)]
    seconds = time.monotonic() - start
    (out / "mineru.json").write_text(json.dumps(blocks, ensure_ascii=False), encoding="utf-8")
    (out / "page.md").write_text(json2md(blocks), encoding="utf-8")
    prediction = blocks_to_prediction(blocks, sample["id"], image.size)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    extra = {"blocks": len(blocks), **generations.summary()}
    generate_seconds = extra["layout"]["seconds"] + extra["recognition_seconds"]
    extra |= {
        "other_seconds": round(max(seconds - generate_seconds, 0.0), 3),
        "max_new_tokens": params["max_new_tokens"],
        "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) >> 20,
    }
    result: dict[str, Any] = {"status": "ok", "primary": "prediction.json", "extra": extra}
    problems = []
    if extra["layout"]["stop_reason"] != "end-token":
        problems.append(f"layout detection stopped at {extra['layout']['stop_reason']}")
    if extra["truncated_blocks"]:
        problems.append(
            f"{extra['truncated_blocks']} of {extra['vl_calls']} blocks stopped without an end "
            "token"
        )
    if problems:
        result["status"] = "truncated"
        result["message"] = "; ".join(problems)
    return result


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    models, params, info, files, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        files = check_model()
        models = load_client(params)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load MinerU2.5-Pro: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": files,
        "job_model": job.get("model"),
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
    print(f"mineru25-pro adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
