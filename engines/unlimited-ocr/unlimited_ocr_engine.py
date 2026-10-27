"""Unlimited-OCR page parsing engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, check the pinned
baidu/Unlimited-OCR checkpoint in the /cache volume (models.sha256) and start vLLM on it as a server
process inside the container, reachable on the loopback interface only. Every sample is parsed in
job order by one greedy request with the model card's document prompt, the whole page image in
vLLM's crop ("gundam") mode and the official no-repeat n-gram logits processor. The model writes
the page as layout blocks, each a ``<|det|>label [x1, y1, x2, y2]<|/det|>`` tag (box scaled to
[0, 999] of the image) followed by its Markdown; grounding references and unterminated tags
are handled as README.md describes. It writes $OCRBENCH_OUTPUT/<id>/ with the model's
output unchanged (output.txt), the parsed blocks (layout.json), the model card's Markdown rendering
(page.md), prediction.json (the primary output: one region per block, its box in input-image
pixels and its Markdown as region markup, converted to text on the host) and, last, result.json
with the token counts, the stop reason, the repetition check and the timing.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

REPOSITORY = "baidu/Unlimited-OCR"
# The last revision whose config.json has model_type "unlimited-ocr", the name vLLM 0.26 registers;
# the next one (8249d20b) renames it for Transformers and has the same weights (README.md).
REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_UNLIMITED_OCR_CACHE", "/cache"))
PINNED_DIR = f"models/unlimited-ocr/Unlimited-OCR/{REVISION}"
CHECKSUMS = Path(os.environ.get("OCRBENCH_UNLIMITED_OCR_CHECKSUMS", "/opt/ocrbench/models.sha256"))
SERVER_LOG = Path(os.environ.get("OCRBENCH_UNLIMITED_OCR_LOG", "/tmp/vllm-server.log"))
SERVED_MODEL = "unlimited-ocr"
# The language model's context (language_config.max_position_embeddings): the image and prompt
# tokens plus the generated ones, the max_length of the model card's infer().
CONTEXT = 32768
# The vLLM recipe's server settings (its n-gram logits processor, no prefix caching, no
# multimodal processor cache) with one image per request, sent one at a time. The checkpoint
# holds no remote code: vLLM's own model, processor, configuration and chat template are used.
SERVER_ARGS = [
    "--served-model-name", SERVED_MODEL,
    "--dtype", "bfloat16",
    "--distributed-executor-backend", "uni",
    "--tensor-parallel-size", "1",
    "--max-model-len", str(CONTEXT),
    "--no-enable-prefix-caching",
    "--mm-processor-cache-gb", "0",
    "--limit-mm-per-prompt", json.dumps({"image": 1}),
    "--logits-processors", "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
]  # fmt: skip
# The model card's single-image document prompt; it must start with the image placeholder.
PROMPT = "<image>document parsing."
# The model card's and the recipe's decoding for a single image: greedy, with no repeated 35-token
# n-gram within the last 128 generated tokens.
SAMPLING = {"temperature": 0.0, "ngram_size": 35, "window_size": 128}
# A layout tag: <|det|>label [x1, y1, x2, y2]<|/det|>, the box scaled to [0, 999]. A tag ends at
# its closing delimiter or, when that is missing, at the end of its line (README.md).
OPEN, CLOSE = "<|det|>", "<|/det|>"
HEADER = re.compile(r"\s*([A-Za-z_][\w-]*)\s*(?:\[([^\]]*)\])?\s*", re.DOTALL)
# The recipe's grounding wrapper <|ref|>…<|/ref|>. Directly before a tag, as in DeepSeek-OCR's
# <|ref|>label<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>, it holds the tag's label and the tag only
# the box; anywhere else it is unwrapped, keeping its text.
REF = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
REF_BEFORE_TAG = re.compile(r"<\|ref\|>((?:(?!<\|).)*)<\|/ref\|>\s*$", re.DOTALL)
LABEL = re.compile(r"[A-Za-z_][\w-]*")
BOX_ONLY = re.compile(r"\s*\[\s*(?:\[([^\[\]]*)\]|([^\[\]]*))\s*\]\s*")
# Grounding delimiters left without a partner are removed from the Markdown.
STRAY = re.compile(r"<\|/?(?:ref|det)\|>|<\|grounding\|>")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
BOX_SCALE = 999
IMAGE_LABEL = "image"
# A tail unit of at most REPEAT_UNIT characters repeated REPEAT_TIMES times in a row is reported
# as a repetition loop (the n-gram processor only forbids exact 35-token repeats within 128).
REPEAT_TIMES, REPEAT_UNIT = 8, 256
# Unlimited-OCR label: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "text": "text",
    "title": "heading",
    "header": "page-header",
    "footer": "page-footer",
    "page_number": "page-number",
    "table": "table",
    "image": "figure",
    "figure": "figure",
    "chart": "figure",
    "seal": "figure",
    "image_caption": "caption",
    "table_caption": "caption",
    "image_footnote": "footnote",
    "table_footnote": "footnote",
    "page_footnote": "footnote",
    "equation": "formula",
}
# Engine parameters (job.json "params"), README.md. max_tokens null generates up to the end of
# the context, as the model card's infer(max_length=32768); gpu_memory_utilization is the
# benchmark's limit for an in-container server (ADR 0012).
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "max_tokens": None,
    "gpu_memory_utilization": 0.6,
    "max_num_seqs": 8,
    "startup_seconds": 900,
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
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
        raise AdapterError(f"device must be 'cuda:<n>' (vLLM needs a GPU): {device!r}")
    tokens = merged["max_tokens"]
    if tokens is not None and (type(tokens) is not int or not 1 <= tokens < CONTEXT):
        raise AdapterError(
            f"max_tokens must be an integer from 1 to below the {CONTEXT}-token context "
            f"(or absent): {tokens!r}"
        )
    share = merged["gpu_memory_utilization"]
    if type(share) not in (int, float) or not 0 < share <= 0.6:
        raise AdapterError(f"gpu_memory_utilization must be in (0, 0.6]: {share!r}")
    seqs = merged["max_num_seqs"]
    if type(seqs) is not int or seqs < 1:
        raise AdapterError(f"max_num_seqs must be a positive integer: {seqs!r}")
    startup = merged["startup_seconds"]
    if type(startup) not in (int, float) or startup <= 0:
        raise AdapterError(f"startup_seconds must be a positive number: {startup!r}")
    return merged


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model() -> dict[str, str]:
    """SHA-256 of the files of the pinned model directory, which must hold exactly the files of
    models.sha256, all matching."""
    directory = CACHE / PINNED_DIR
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make unlimited-ocr-model and run with --cache (README.md)"
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
    except Exception as error:
        raise AdapterError(f"cannot import PyTorch: {error}") from error
    info: dict[str, Any] = {
        name: metadata.version(name) for name in ("vllm", "transformers", "huggingface-hub")
    }
    info |= {"torch": torch.__version__, "cuda": torch.version.cuda, "device": device}
    index = int(device.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
    info["gpu"] = torch.cuda.get_device_name(index)
    info["gpu_memory_mb"] = torch.cuda.get_device_properties(index).total_memory >> 20
    info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(index)))
    return info


def server_command(params: dict[str, Any], port: int) -> list[str]:
    directory = CACHE / PINNED_DIR
    command = ["vllm", "serve", str(directory), "--host", "127.0.0.1", "--port", str(port)]
    command += [*SERVER_ARGS, "--max-num-seqs", str(params["max_num_seqs"])]
    command += ["--gpu-memory-utilization", str(params["gpu_memory_utilization"])]
    return command


def server_env(params: dict[str, Any]) -> dict[str, str]:
    """FlashInfer's top-k/top-p sampler would be compiled at start-up with nvcc, which the image
    does not have; vLLM's PyTorch sampler is used instead (README.md)."""
    return {
        "CUDA_VISIBLE_DEVICES": params["device"].split(":", 1)[1],
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def log_tail(lines: int = 30) -> str:
    try:
        return "\n".join(SERVER_LOG.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no server log)"


class Server:
    """vLLM's OpenAI-compatible server as a child process on the loopback interface. The
    container has no network; the server only serves this adapter."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.port = free_port()
        self.command = server_command(params, self.port)
        self.env = server_env(params)
        with SERVER_LOG.open("wb") as log:
            self.process = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT,
                env=os.environ | self.env, start_new_session=True,
            )  # fmt: skip
        self.url = f"http://127.0.0.1:{self.port}"
        self.wait(params["startup_seconds"])

    def wait(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AdapterError(
                    f"vLLM exited with status {self.process.returncode} during start-up:\n"
                    + log_tail()
                )
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(1)
        self.stop()
        raise AdapterError(f"vLLM did not become ready within {seconds} s:\n" + log_tail())

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)


def read_image(path: Path) -> Any:
    """The image as stored, in RGB (EXIF orientation is not applied,
    docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode == "RGB" else image.convert("RGB")


def payload(image: Any, max_tokens: int | None) -> dict[str, Any]:
    """The recipe's chat request: the prompt, then the whole image as PNG, greedy decoding with the
    n-gram processor's arguments, and the layout tags kept (skip_special_tokens false). Without
    max_tokens vLLM generates up to the end of the context."""
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    request: dict[str, Any] = {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": SAMPLING["temperature"],
        "skip_special_tokens": False,
        "vllm_xargs": {
            "ngram_size": SAMPLING["ngram_size"],
            "window_size": SAMPLING["window_size"],
        },
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    return request


def generate(server_url: str, image: Any, max_tokens: int | None) -> dict[str, Any]:
    """One chat completion request; returns the output, its stop reason, usage and time."""
    request = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload(image, max_tokens)).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request) as reply:
            body = json.loads(reply.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise AdapterError(f"request failed: HTTP {error.code}: {detail}") from None
    seconds = time.monotonic() - start
    choice = body["choices"][0]
    return {
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
        "seconds": seconds,
    }


def tail_repeats(text: str) -> bool:
    """Whether the text ends with a non-blank unit of at most REPEAT_UNIT characters repeated
    REPEAT_TIMES times in a row."""
    for length in range(1, min(REPEAT_UNIT, len(text) // REPEAT_TIMES) + 1):
        unit = text[-length:]
        if unit.strip() and text.endswith(unit * REPEAT_TIMES):
            return True
    return False


def parse_box(text: str | None) -> list[int | float] | None:
    """The four numbers of a box as written, or None when it is not exactly four numbers. The
    box is read, never evaluated: no box is invented or computed."""
    if text is None:
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4 or not all(NUMBER.fullmatch(part) for part in parts):
        return None
    return [float(part) if "." in part else int(part) for part in parts]


def parse_header(det: str, ref: str | None) -> tuple[str | None, list[int | float] | None]:
    """The label and box of a tag: ``label [box]`` in the tag, or, after a reference, the label
    in the reference (when it is a label, not text) and only the box, single or double bracketed,
    in the tag."""
    if ref is None:
        header = HEADER.fullmatch(det)
        return (header.group(1), parse_box(header.group(2))) if header else (None, None)
    label = ref.strip() if LABEL.fullmatch(ref.strip()) else None
    box = BOX_ONLY.fullmatch(det)
    if box is None:
        return label, None
    return label, parse_box(box.group(1) if box.group(1) is not None else box.group(2))


def clean(content: str) -> str:
    """Markdown without grounding delimiters: references unwrapped, stray delimiters removed."""
    return STRAY.sub("", REF.sub(r"\1", content)).strip()


def layout_blocks(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """The blocks of the output in order: the label and box as written (``det``), a reference
    before the tag (``ref``, only when present), the label (``label``, None when the tag has
    none), the box (``box``, four numbers scaled to [0, 999], or None), the same box in input-image
    pixels (``box_px``, float pixel edges sorted and clipped to the image), the Markdown up to the
    next tag (``content``, stripped, grounding delimiters removed) and the character offsets of the
    tag with its reference (``span``). A tag without a closing delimiter (``unterminated`` true)
    ends at the end of its line or at the next tag. Text before the first tag is a block without
    a tag."""
    opens = [match.start() for match in re.finditer(re.escape(OPEN), text)]
    head = text[: opens[0]] if opens else text
    tags: list[dict[str, Any]] = []
    for index, start in enumerate(opens):
        end = opens[index + 1] if index + 1 < len(opens) else len(text)
        body = text[start + len(OPEN) : end]
        close = body.find(CLOSE)
        if close >= 0:
            det, content = body[:close], body[close + len(CLOSE) :]
            span = [start, start + len(OPEN) + close + len(CLOSE)]
        else:
            det, _, content = body.partition("\n")
            span = [start, start + len(OPEN) + len(det)]
        tags.append(
            {"det": det, "ref": None, "span": span, "content": content, "closed": close >= 0}
        )
    # A reference that ends the text before a tag belongs to that tag.
    for index, tag in enumerate(tags):
        before = tags[index - 1]["content"] if index else head
        ref = REF_BEFORE_TAG.search(before)
        if ref:
            tag["ref"], tag["span"][0] = ref.group(1), tag["span"][0] - len(before) + ref.start()
            if index:
                tags[index - 1]["content"] = before[: ref.start()]
            else:
                head = before[: ref.start()]
    blocks: list[dict[str, Any]] = []
    if clean(head):
        blocks.append({"det": None, "label": None, "box": None, "content": clean(head)})
    for tag in tags:
        label, box = parse_header(tag["det"], tag["ref"])
        block: dict[str, Any] = {"det": tag["det"]}
        if tag["ref"] is not None:
            block["ref"] = tag["ref"]
        block |= {"label": label, "box": box}
        if box is not None:
            x1, y1, x2, y2 = box
            xs = sorted(min(max(x * width / BOX_SCALE, 0), width) for x in (x1, x2))
            ys = sorted(min(max(y * height / BOX_SCALE, 0), height) for y in (y1, y2))
            block["box_px"] = [xs[0], ys[0], xs[1], ys[1]]
        content = clean(tag["content"])
        if tag["ref"] is not None and label is None and clean(tag["ref"]):
            content = "\n".join(filter(None, [clean(tag["ref"]), content]))  # referenced text
        block["content"] = content
        block["span"] = tag["span"]
        if not tag["closed"]:
            block["unterminated"] = True
        blocks.append(block)
    return blocks


def markdown(text: str) -> str:
    """The model card's OmniDocBench rendering (remove_det): every tag starts a block, image
    blocks are dropped, blank lines are removed and trailing spaces stripped, the lines of a block
    are joined by line breaks and blocks by blank lines."""
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        match = re.match(r"<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)", line, re.DOTALL)
        if match:
            category, content = match.group(1).strip(), match.group(2).strip()
            if category == IMAGE_LABEL:
                continue
            if current is not None:
                blocks.append(current)
            current = [content] if content else []
            continue
        if current is None:
            current = []
        current.append(line)
    if current is not None:
        blocks.append(current)
    return "\n\n".join("\n".join(block) for block in blocks).strip()


def blocks_to_prediction(blocks: list[dict[str, Any]], sample_id: str, image: Any) -> dict:
    """Standardized prediction (docs/PREDICTIONS.md): one region per block, in model order, the
    reading order. A block's box is its polygon in input-image pixels; a block without a valid
    box has none. Its Markdown (tables as HTML) is region markup, converted on the host."""
    regions: list[dict[str, Any]] = []
    for block in blocks:
        region: dict[str, Any] = {
            "id": f"b{len(regions) + 1}",
            "type": REGION_TYPES.get(block["label"], "other"),
        }
        if block["label"] is not None:
            region["native_type"] = block["label"]
        if "box_px" in block:
            x0, y0, x1, y1 = block["box_px"]
            region["polygon"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        if block["content"]:
            region["markup"] = {"format": "markdown", "content": block["content"]}
        regions.append(region)
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": image.width, "height": image.height},
        "text": "",  # built on the host from the converted region markup
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    return prediction


def process(server_url: str, params: dict, sample: dict, source: Path, out: Path) -> dict:
    """Parse one sample, write its outputs and return ``result.json`` content."""
    image = read_image(source / sample["image"])
    response = generate(server_url, image, params["max_tokens"])
    output = response["text"]
    (out / "output.txt").write_text(output, encoding="utf-8")  # the native output, unchanged
    blocks = layout_blocks(output, image.width, image.height)
    layout = {"image": {"width": image.width, "height": image.height}, "blocks": blocks}
    (out / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), "utf-8")
    (out / "page.md").write_text(markdown(output), encoding="utf-8")
    prediction = blocks_to_prediction(blocks, sample["id"], image)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    usage = response["usage"]
    prompt, generated = usage.get("prompt_tokens"), usage.get("completion_tokens")
    reason, repeats = response["finish_reason"], tail_repeats(output)
    result: dict[str, Any] = {"status": "ok", "primary": "prediction.json"}
    if reason == "length":
        full = prompt is not None and generated is not None and prompt + generated >= CONTEXT
        limit = f"the end of the {CONTEXT}-token context" if full else "max_tokens"
        result["status"] = "truncated"
        result["message"] = f"generation stopped at {limit} ({generated} tokens)" + (
            "; the output ends in a repetition loop" if repeats else ""
        )
    elif reason != "stop":  # failed, with the request's diagnostics kept
        result = {"status": "failed", "message": f"generation failed (finish_reason {reason!r})"}
    elif not output.strip():
        result["message"] = "the model returned an empty page"
    seconds = response["seconds"]
    extra = {
        "finish_reason": reason,
        "tail_repetition": repeats,
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "max_tokens": params["max_tokens"],
        "request_seconds": round(seconds, 3),
        "tokens_per_second": round((generated or 0) / max(seconds, 1e-9), 1),
        "source_image": {"width": image.width, "height": image.height},
        "blocks": len(blocks),
        "labels": sorted({b["label"] for b in blocks if b["label"] is not None}),
        "blocks_without_box": sum("box_px" not in b for b in blocks),
        "unterminated_tags": sum(b.get("unterminated", False) for b in blocks),
    }
    return result | {"extra": extra}


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    server, params, info, files, init_error = None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        files = check_model()
        server = Server(params)
        info["server_command"] = server.command
        info["server_env"] = server.env
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot start Unlimited-OCR: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": files,
        "job_model": job.get("model"),
        "prompt": PROMPT,
        "sampling": SAMPLING,
        "context": CONTEXT,
    }
    try:
        for sample in job["samples"]:
            out = target / sample["id"]
            out.mkdir()
            start = time.monotonic()
            try:
                if init_error:
                    raise AdapterError(init_error)
                if not server.alive():
                    raise AdapterError(f"the vLLM server has exited:\n{log_tail()}")
                result = process(server.url, params, sample, source, out)
            except AdapterError as error:
                result = {"status": "failed", "message": str(error)}
            except Exception as error:  # e.g. a request error: fail the sample, keep going
                result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
            result["seconds"] = round(time.monotonic() - start, 3)
            extra = result.setdefault("extra", {})
            extra.update(environment=info, models=model_info, params=params)
            write_result(out, result)
    finally:
        if server:
            server.stop()
    print(f"unlimited-ocr adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
