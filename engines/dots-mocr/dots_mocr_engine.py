"""dots.mocr page-image to layout engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, load the model
and its own code from the /cache volume once (checked against models.sha256), and for every
sample in job order generate the layout of the image as stored with the official
prompt_layout_all_en prompt: a JSON list of layout elements, each with a box, a category and its
text, in reading order. It writes $OCRBENCH_OUTPUT/<id>/ with the model's output unchanged
(output.json), the standardized prediction.json (the primary output: one region per element, its
box mapped to input-image pixels) and, last, result.json with the token counts, the stop reason
and the timing. An output cut off inside the list keeps its complete elements.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import json
import os
import re
import sys
import time
import types
from html.parser import HTMLParser
from importlib import metadata
from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

REPOSITORY = "dots-studio/dots.mocr"
REVISION = "e539fbb52280393adc081b289ec597430a0f9031"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_DOTS_MOCR_CACHE", "/cache"))
# Relative to the cache; without periods, as the model card asks for the remote code.
PINNED_DIR = f"models/dots-mocr/dots-mocr/{REVISION}"
CHECKSUMS = Path(os.environ.get("OCRBENCH_DOTS_MOCR_CHECKSUMS", "/opt/ocrbench/models.sha256"))
# The model's whole context (max_position_embeddings): prompt plus generated tokens.
CONTEXT_TOKENS = 131072
# The official prompt_layout_all_en (dots_mocr/utils/prompts.py of github.com/rednote-hilab/
# dots.mocr at 23f3e5612fb8066d4034d5ecfc8f33a9243533eb), verbatim; it follows the image.
PROMPT = (
    "Please output the layout information from the PDF image, including each layout element's "
    "bbox, its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', "
    "'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', "
    "'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object.\n"
)
CONVERSATION = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
PATCH = 14  # 14-pixel patches, merged 2 x 2 into one image token
# dots.mocr category: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "Caption": "caption",
    "Footnote": "footnote",
    "Formula": "formula",
    "List-item": "list-item",
    "Page-footer": "page-footer",
    "Page-header": "page-header",
    "Picture": "figure",
    "Section-header": "heading",
    "Table": "table",
    "Text": "text",
    "Title": "heading",
}
# Engine parameters (job.json "params"), README.md. max_new_tokens is the official
# transformers inference's (dots_mocr/parser.py, demo/demo_hf.py).
DEFAULT_PARAMS: dict[str, Any] = {"device": "cuda:0", "max_new_tokens": 24000}


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
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
        raise AdapterError(f"device must be 'cuda:<n>' (the model code needs a GPU): {device!r}")
    tokens = merged["max_new_tokens"]
    if type(tokens) is not int or not 1 <= tokens < CONTEXT_TOKENS:
        raise AdapterError(
            f"max_new_tokens must be an integer from 1 to below the context of {CONTEXT_TOKENS}: "
            f"{tokens!r}"
        )
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model() -> dict[str, str]:
    """SHA-256 of the files of the pinned model directory, which must hold exactly the files of
    models.sha256, all matching: transformers imports the Python files it finds there."""
    directory = CACHE / PINNED_DIR
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make dots-mocr-model and run with --cache (README.md)"
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
    index = int(device.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
    info["gpu"] = torch.cuda.get_device_name(index)
    info["gpu_memory_mb"] = torch.cuda.get_device_properties(index).total_memory >> 20
    return info


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    """The one flash-attn function the model's vision encoder calls, computed with PyTorch's
    scaled-dot-product attention: every sequence of the packed batch (``cu_seqlens``, one per
    image) attends only to itself, with the default 1/sqrt(head size) scale. ``q``, ``k`` and
    ``v`` are (tokens, heads, head size)."""
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    if not torch.equal(cu_seqlens_q, cu_seqlens_k):
        raise ValueError("different query and key sequences are not supported")
    output = torch.empty_like(q)
    bounds = cu_seqlens_q.tolist()
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        output[start:end] = scaled_dot_product_attention(
            q[start:end].transpose(0, 1),
            k[start:end].transpose(0, 1),
            v[start:end].transpose(0, 1),
            is_causal=causal,
        ).transpose(0, 1)
    return output


def install_flash_attn() -> None:
    """Make ``from flash_attn import flash_attn_varlen_func`` in the model code import the
    function above. flash-attn itself is not installed, and transformers does not consider it
    available (it has no package metadata), so nothing else uses it."""
    module = types.ModuleType("flash_attn")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    module.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = module


def load_model(params: dict[str, Any]) -> tuple[Any, Any]:
    """The processor and the bfloat16 model, with the checkpoint's own code, from local files
    only. The language model uses PyTorch's SDPA attention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    install_flash_attn()
    path = CACHE / PINNED_DIR
    processor = AutoProcessor.from_pretrained(
        path, trust_remote_code=True, use_fast=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        local_files_only=True,
    )
    return processor, model.to(params["device"]).eval()


def read_image(path: Path) -> Any:
    """The image as stored (EXIF orientation is not applied, docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode == "RGB" else image.convert("RGB")


def stop_tokens(model: Any) -> set[int]:
    eos = model.generation_config.eos_token_id
    return set(eos if isinstance(eos, list) else [eos])


def generate(processor: Any, model: Any, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """Generate the layout of one page greedily; returns the decoded text and the token and
    timing details.

    Generation stops at an end token, at ``max_new_tokens`` or at the end of the model's context,
    whichever comes first; without an end token the output is truncated.
    """
    import torch

    device = params["device"]
    start = time.monotonic()
    prompt = processor.apply_chat_template(CONVERSATION, tokenize=False, add_generation_prompt=True)
    # The processor scales the image to whole 28-pixel units (Qwen2-VL smart_resize).
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    torch.cuda.reset_peak_memory_stats(device)  # the peak includes moving the inputs to the GPU
    dtype = model.dtype
    inputs = {
        key: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        for key, value in dict(inputs).items()
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
            eos_token_id=sorted(stops),
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
        "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) >> 20,
    }
    text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"text": text, **details}


def parse_layout(text: str) -> tuple[list[Any], bool]:
    """The layout elements of the output and whether it is one complete JSON list.

    An output that is not a complete list, typically cut off at the token limit, keeps the
    elements of its list that are complete; one that does not start a list has none."""
    try:
        elements = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        if isinstance(elements, list):
            return elements, True
        return [], False
    decoder = json.JSONDecoder()
    stripped = text.lstrip()
    if not stripped.startswith("["):
        return [], False
    position, elements = len(text) - len(stripped) + 1, []
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        try:
            element, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return elements, False
        elements.append(element)


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
    """Plain text of a Table element: its HTML table as one line per row with tab-separated
    cells; content that is not an HTML table is kept."""
    if "<table" not in content.lower():
        return content.strip()
    parser = _TableText()
    parser.feed(content)
    parser.close()
    return "\n".join("\t".join(row) for row in parser.rows if any(row))


def formula_text(content: str) -> str:
    """TeX source of a Formula element, without ``$$``, ``\\[ … \\]`` or ``$`` delimiters."""
    text = content.strip()
    for start, end in (("$$", "$$"), ("\\[", "\\]"), ("$", "$")):
        if len(text) > len(start) + len(end) and text.startswith(start) and text.endswith(end):
            return text[len(start) : -len(end)].strip()
    return text


HEADING = re.compile(r"#{1,6}[ \t]+")
BULLET = re.compile(r"[-*+•][ \t]+")
# The host Markdown converter's parser (src/ocrbench/convert.py, same pinned versions).
MARKDOWN = MarkdownIt("commonmark").enable(["table", "strikethrough"])
MARKDOWN.use(dollarmath_plugin, allow_space=True, double_inline=True)
BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
INLINE_TEXT = ("text", "code_inline", "math_inline", "math_inline_double")


def inline_text(content: str) -> str:
    """Text of inline Markdown by the host converter's inline rules (docs/PREDICTIONS.md):
    emphasis, links, strikethrough and code keep only their text, inline math its TeX source,
    escapes and entities are decoded, images and HTML tags add nothing except ``<br>``, a line
    break; spaces are collapsed and blank lines dropped."""
    parts = []
    for child in MARKDOWN.parseInline(content)[0].children or []:
        if child.type in INLINE_TEXT:
            parts.append(child.content)
        elif child.type in ("softbreak", "hardbreak") or (
            child.type == "html_inline" and BREAK.fullmatch(child.content)
        ):
            parts.append("\n")
    lines = (re.sub(r"[ \t\r\f]+", " ", line).strip() for line in "".join(parts).split("\n"))
    return "\n".join(line for line in lines if line)


def markdown_text(category: str, content: str) -> str:
    """Plain text of a Markdown element: the heading marks of a Title or Section-header and the
    bullet of a List-item are removed, then inline Markdown is converted (``inline_text``)."""
    text = content.strip()
    if category in ("Title", "Section-header"):
        text = HEADING.sub("", text, count=1) if HEADING.match(text) else text
    elif category == "List-item":
        text = BULLET.sub("", text, count=1) if BULLET.match(text) else text
    return inline_text(text)


def element_text(category: str, content: str) -> str:
    if category == "Picture":
        return ""  # the prompt asks for no text; what the model writes is not used
    if category == "Table":
        return table_text(content)
    if category == "Formula":
        return formula_text(content)
    return markdown_text(category, content)


def box(element: dict[str, Any]) -> list[float] | None:
    """The element's box [x1, y1, x2, y2] in model-image pixels, or None when it is not four
    finite numbers with x1 < x2 and y1 < y2 (the official is_legal_bbox)."""
    value = element.get("bbox")
    if not (isinstance(value, list) and len(value) == 4):
        return None
    if not all(type(v) in (int, float) and abs(v) < float("inf") for v in value):
        return None
    x1, y1, x2, y2 = (float(v) for v in value)
    return [x1, y1, x2, y2] if x1 < x2 and y1 < y2 else None


def layout_to_prediction(
    elements: list[Any],
    sample_id: str,
    size: tuple[int, int],
    model_size: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Standardized prediction (docs/PREDICTIONS.md) from the layout elements of one page, and
    counts of what was not used as given.

    Every element becomes a region in output order, which the prompt asks to be the reading
    order. Its box is in pixels of the image the model saw (``model_size``) and is mapped to
    input-image pixels as floats; an element without a valid box becomes a region without a
    polygon (no box is invented). Its text is the element's ``text`` in plain text
    (``element_text``). An element that is not a JSON object is skipped.
    """
    width, height = size
    scale_x, scale_y = width / model_size[0], height / model_size[1]
    regions: list[dict[str, Any]] = []
    counts = {"skipped_elements": 0, "invalid_boxes": 0, "repeated_elements": 0}
    previous = None
    for element in elements:
        if not isinstance(element, dict):
            counts["skipped_elements"] += 1
            continue
        category = element.get("category")
        category = category if isinstance(category, str) else None
        content = element.get("text")
        content = content if isinstance(content, str) else ""
        region: dict[str, Any] = {
            "id": f"e{len(regions) + 1}",
            "type": REGION_TYPES.get(category, "other"),
        }
        if category is not None:
            region["native_type"] = category
        bbox = box(element)
        if bbox is None:
            counts["invalid_boxes"] += 1
            region["extra"] = {"bbox": element.get("bbox")}
        else:
            x1, y1, x2, y2 = bbox
            x1, x2, y1, y2 = x1 * scale_x, x2 * scale_x, y1 * scale_y, y2 * scale_y
            region["polygon"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        text = element_text(category or "", content)
        if text:
            region["text"] = text
        if (category, content) == previous and content.strip():
            counts["repeated_elements"] += 1
        previous = (category, content)
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
    return prediction, counts


STOPPED = {
    "max_new_tokens": "generation stopped at max_new_tokens",
    "context": "generation stopped at the model's context",
}


def process(models: tuple, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """Parse one sample, write its outputs and return ``result.json`` content."""
    processor, model = models
    image = read_image(source / sample["image"])
    generation = generate(processor, model, params, image)
    text = generation.pop("text")
    (out / "output.json").write_text(text, encoding="utf-8")
    elements, complete = parse_layout(text)
    model_image = generation["model_image"]
    prediction, counts = layout_to_prediction(
        elements, sample["id"], image.size, (model_image["width"], model_image["height"])
    )
    generation |= {
        "source_image": {"width": image.width, "height": image.height},
        "json_complete": complete,
        "elements": len(elements),
        **counts,
    }
    reason = generation["stop_reason"]
    if reason == "end-token" and not complete and text.strip():
        if not elements:
            raise AdapterError("the output is not a JSON list of layout elements (output.json)")
        result: dict[str, Any] = {
            "status": "partial",
            "message": f"the output is not a complete JSON list; {len(elements)} complete "
            "elements are used",
        }
    elif reason != "end-token":
        result = {
            "status": "truncated",
            "message": f"{STOPPED[reason]} ({generation['generated_tokens']} of "
            f"{generation['max_new_tokens']} tokens); {len(elements)} complete elements are used",
        }
    else:
        result = {"status": "ok"}
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result | {"primary": "prediction.json", "extra": generation}


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
        models = load_model(params)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot load dots.mocr: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {
        "repository": REPOSITORY,
        "revision": REVISION,
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
    print(f"dots-mocr adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
