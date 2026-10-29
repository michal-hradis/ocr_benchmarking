"""LightOnOCR-3 engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, check the pinned
checkpoint named by the ``checkpoint`` parameter (LightOnOCR-3-0.8B, -1B or -4B) in the /cache
volume (models-<checkpoint>.sha256) and start vLLM on it as a server process inside the container,
reachable on the loopback interface only. Every sample is sent in job order as one request of
LightOn's client (lightonocr.client.LightOnOCR.ocr): the image, scaled down to the checkpoint's
longest edge, and in ``grounding`` mode the text "grounding". The ``mode`` parameter is explicit:

- ``plain``: the model writes the page as Markdown (page.md, the primary output, converted to text
  on the host);
- ``grounding``: the model writes one ``![label](x1,y1,x2,y2)`` marker before every layout block,
  its box normalized to 0-1000 of the image. It writes $OCRBENCH_OUTPUT/<id>/ with the model's
  output unchanged (output.md), the parsed blocks (layout.json) and prediction.json (the primary
  output: one region per block, its box in input-image pixels and its Markdown as region markup,
  converted to text on the host).

result.json comes last, with the token counts, the stop reason, the repetition check and the timing.
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

# The three LightOnOCR-3 checkpoints (README.md). 0.8B and 4B are Qwen3.5 vision-language models
# that take the image as sent and need thinking disabled; 1B keeps the LightOnOCR-2 architecture
# (Pixtral encoder, Qwen3 decoder), whose processor caps images at 1540 px. ``context`` and
# ``longest_edge`` are the values of LightOn's README and client (lightonocr/models.py).
CHECKPOINTS: dict[str, dict[str, Any]] = {
    "0.8B": {
        "repository": "lightonai/LightOnOCR-3-0.8B",
        "revision": "36d8636925327c671bd8e479c517a004f7566bd9",
        "context": 18000,
        "longest_edge": 2048,
        "thinking_flag": True,
    },
    "1B": {
        "repository": "lightonai/LightOnOCR-3-1B",
        "revision": "23d544085c886bbdace647bcfab91203299417e2",
        "context": 16384,
        "longest_edge": 1540,
        "thinking_flag": False,
    },
    "4B": {
        "repository": "lightonai/LightOnOCR-3-4B",
        "revision": "b71010f095dc3735de7ee6b9969bff8f8f50b39a",
        "context": 18000,
        "longest_edge": 2048,
        "thinking_flag": True,
    },
}
MODES = ("plain", "grounding")
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_LIGHTONOCR3_CACHE", "/cache"))
CHECKSUMS = Path(os.environ.get("OCRBENCH_LIGHTONOCR3_CHECKSUMS", "/opt/ocrbench"))
SERVER_LOG = Path(os.environ.get("OCRBENCH_LIGHTONOCR3_LOG", "/tmp/vllm-server.log"))
SERVED_MODEL = "lightonocr3"
# LightOn's server settings (README.md of lightonai/LightOnOCR): one image per prompt, no multimodal
# processor cache, no prefix caching; the Qwen3.5 checkpoints also get thinking disabled.
SERVER_ARGS = [
    "--served-model-name", SERVED_MODEL,
    "--dtype", "bfloat16",
    "--distributed-executor-backend", "uni",
    "--tensor-parallel-size", "1",
    "--limit-mm-per-prompt", json.dumps({"image": 1}),
    "--mm-processor-cache-gb", "0",
    "--no-enable-prefix-caching",
]  # fmt: skip
NO_THINKING = {"enable_thinking": False}
GROUNDING_PROMPT = "grounding"
# A grounding marker: ![label](x1,y1,x2,y2), a trailing "+" on the label marking a continuation,
# the box exactly four integers as in LightOn's format. Anything else, such as a Markdown image
# ![Fig](a.png) or a box of three numbers, is not a marker and stays text of the block it is in.
INT = r"\s*(\d+)\s*"
MARKER = re.compile(rf"!\[([A-Za-z_][\w-]*?)(\+?)\]\({INT},{INT},{INT},{INT}\)[ \t]*")
BOX_SCALE = 1000
# Descriptions of pictures and data read from charts are not text on the page: these blocks keep
# their box but no text (README.md; the maintainer's decision for Chandra OCR 2 in #53).
UNSCORED = {"image", "header_image", "footer_image", "chart"}
# The output is reported as ending in a repetition loop when its tail is a unit repeated in a row:
# SHORT_TIMES times for a unit of fewer than LONG_UNIT characters, LONG_TIMES times for a longer
# one (a looping sentence or paragraph) of at most LONG_MAX characters.
SHORT_TIMES, LONG_TIMES, LONG_UNIT, LONG_MAX = 8, 3, 64, 4096
# LightOnOCR-3 label: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "text": "text",
    "aside_text": "text",
    "title": "heading",
    "section": "heading",
    "list": "list-item",
    "caption": "caption",
    "footnote": "footnote",
    "header": "page-header",
    "footer": "page-footer",
    "page_number": "page-number",
    "table": "table",
    "formula": "formula",
    "image": "figure",
    "header_image": "figure",
    "footer_image": "figure",
    "chart": "figure",
}
# Engine parameters (job.json "params"), README.md. checkpoint and mode have no default: every
# preset names both. max_tokens null generates up to the end of the context, as LightOn's client;
# the sampling is the client's for LightOnOCR-3, with a fixed seed per request; longest_edge null is
# the checkpoint's; gpu_memory_utilization is the benchmark's limit for an in-container server
# (ADR 0012).
DEFAULT_PARAMS: dict[str, Any] = {
    "checkpoint": None,
    "mode": None,
    "device": "cuda:0",
    "max_tokens": None,
    "temperature": 0.2,
    "top_p": 1.0,
    "seed": 0,
    "longest_edge": None,
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
    if merged["checkpoint"] not in CHECKPOINTS:
        raise AdapterError(
            f"checkpoint must be one of {', '.join(CHECKPOINTS)}: {merged['checkpoint']!r}"
        )
    if merged["mode"] not in MODES:
        raise AdapterError(f"mode must be one of {', '.join(MODES)}: {merged['mode']!r}")
    context = CHECKPOINTS[merged["checkpoint"]]["context"]
    device = merged["device"]
    if not isinstance(device, str) or not re.fullmatch(r"cuda:[0-9]+", device):
        raise AdapterError(f"device must be 'cuda:<n>' (vLLM needs a GPU): {device!r}")
    tokens = merged["max_tokens"]
    if tokens is not None and (type(tokens) is not int or not 1 <= tokens < context):
        raise AdapterError(
            f"max_tokens must be an integer from 1 to below the {context}-token context "
            f"(or absent): {tokens!r}"
        )
    temperature = merged["temperature"]
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise AdapterError(f"temperature must be in [0, 2]: {temperature!r}")
    top_p = merged["top_p"]
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise AdapterError(f"top_p must be in (0, 1]: {top_p!r}")
    if type(merged["seed"]) is not int:
        raise AdapterError(f"seed must be an integer: {merged['seed']!r}")
    edge = merged["longest_edge"]
    if edge is not None and (type(edge) is not int or edge < 64):
        raise AdapterError(f"longest_edge must be an integer of at least 64 (or absent): {edge!r}")
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


def pinned_dir(checkpoint: str) -> str:
    spec = CHECKPOINTS[checkpoint]
    return f"models/lightonocr3/{spec['repository'].split('/')[1]}/{spec['revision']}"


def checksums_file(checkpoint: str) -> Path:
    return CHECKSUMS / f"models-{checkpoint}.sha256"


def longest_edge(params: dict[str, Any]) -> int:
    return params["longest_edge"] or CHECKPOINTS[params["checkpoint"]]["longest_edge"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model(checkpoint: str) -> dict[str, str]:
    """SHA-256 of the files of the pinned model directory, which must hold exactly the files of
    the checkpoint's checksum list, all matching."""
    directory = CACHE / pinned_dir(checkpoint)
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            f"make lightonocr3-model CHECKPOINT={checkpoint} and run with --cache (README.md)"
        )
    files = {
        p.relative_to(directory).as_posix(): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    listing = checksums_file(checkpoint)
    pinned = dict(
        reversed(line.split()) for line in listing.read_text().splitlines() if line.strip()
    )
    bad = sorted(name for name, digest in pinned.items() if files.get(name) != digest)
    if bad:
        raise AdapterError(f"model files missing or not matching {listing.name}: {bad}")
    extra = sorted(set(files) - set(pinned))
    if extra:
        raise AdapterError(f"model files not in {listing.name}: {extra}")
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
    spec = CHECKPOINTS[params["checkpoint"]]
    directory = CACHE / pinned_dir(params["checkpoint"])
    command = ["vllm", "serve", str(directory), "--host", "127.0.0.1", "--port", str(port)]
    command += [*SERVER_ARGS, "--max-model-len", str(spec["context"])]
    if spec["thinking_flag"]:
        command += ["--default-chat-template-kwargs", json.dumps(NO_THINKING)]
    command += ["--max-num-seqs", str(params["max_num_seqs"])]
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


def fit(image: Any, edge: int) -> Any:
    """LightOn's lightonocr.render.fit: scale down with Lanczos so that the longest side is at
    most ``edge`` pixels, keeping the aspect ratio; never scale up."""
    from PIL import Image

    width, height = image.size
    scale = edge / max(width, height)
    if scale >= 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def payload(image: Any, params: dict[str, Any]) -> dict[str, Any]:
    """LightOn's client request: one user message with the image as PNG and, in grounding mode,
    the text "grounding"; the client's sampling with a fixed seed and thinking disabled. Without
    max_tokens vLLM generates up to the end of the context."""
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": data_url}}]
    if params["mode"] == "grounding":
        content.append({"type": "text", "text": GROUNDING_PROMPT})
    request: dict[str, Any] = {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "seed": params["seed"],
        "chat_template_kwargs": NO_THINKING,
    }
    if params["max_tokens"] is not None:
        request["max_tokens"] = params["max_tokens"]
    return request


def generate(server_url: str, image: Any, params: dict[str, Any]) -> dict[str, Any]:
    """One chat completion request; returns the output, its stop reason, usage and time."""
    request = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=json.dumps(payload(image, params)).encode(),
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
    """Whether the text ends with a non-blank unit repeated in a row: SHORT_TIMES times for a unit
    shorter than LONG_UNIT characters, LONG_TIMES times for one of up to LONG_MAX characters."""
    for length in range(1, min(LONG_MAX, len(text) // LONG_TIMES) + 1):
        unit = text[-length:]
        times = SHORT_TIMES if length < LONG_UNIT else LONG_TIMES
        if unit.strip() and text.endswith(unit * times):
            return True
    return False


def layout_blocks(text: str, width: int, height: int) -> list[dict[str, Any]]:
    """The blocks of a grounding output in order, as LightOn's lightonocr.grounding.parse_blocks
    splits them (each marker owns the text up to the next marker): the marker as written
    (``marker``), its label without and with the continuation "+" (``label``, ``continues``), the
    box (``box``, four integers normalized to 0-1000), the same box in input-image pixels
    (``box_px``, float pixel edges sorted and clipped to the image), the text (``content``,
    stripped) and the character offsets of the marker (``span``). Text before the first marker is a
    block without a marker."""
    markers = list(MARKER.finditer(text))
    head = text[: markers[0].start()] if markers else text
    blocks: list[dict[str, Any]] = []
    if head.strip():
        blocks.append({"marker": None, "label": None, "box": None, "content": head.strip()})
    for index, match in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        x1, y1, x2, y2 = box = [int(value) for value in match.group(3, 4, 5, 6)]
        xs = sorted(min(max(x * width / BOX_SCALE, 0), width) for x in (x1, x2))
        ys = sorted(min(max(y * height / BOX_SCALE, 0), height) for y in (y1, y2))
        block: dict[str, Any] = {
            "marker": match.group(0).rstrip(),
            "label": match.group(1),
            "continues": match.group(2) == "+",
            "box": box,
            "box_px": [xs[0], ys[0], xs[1], ys[1]],
        }
        block["content"] = text[match.end() : end].strip()
        block["span"] = [match.start(), match.start() + len(block["marker"])]
        blocks.append(block)
    return blocks


def blocks_to_prediction(blocks: list[dict[str, Any]], sample_id: str, image: Any) -> dict:
    """Standardized prediction (docs/PREDICTIONS.md): one region per block, in model order, the
    reading order. A block's box is its polygon in input-image pixels; text before the first
    marker has none. Its Markdown (tables as HTML) is region markup, converted on the host; picture
    and chart blocks have no text."""
    regions: list[dict[str, Any]] = []
    for block in blocks:
        region: dict[str, Any] = {
            "id": f"b{len(regions) + 1}",
            "type": REGION_TYPES.get(block["label"], "other"),
        }
        if block["label"] is not None:
            region["native_type"] = block["label"] + ("+" if block["continues"] else "")
        if "box_px" in block:
            x0, y0, x1, y1 = block["box_px"]
            region["polygon"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        if block["content"] and block["label"] not in UNSCORED:
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
    """OCR one sample, write its outputs and return ``result.json`` content."""
    image = read_image(source / sample["image"])
    sent = fit(image, longest_edge(params))
    response = generate(server_url, sent, params)
    output = response["text"]
    extra: dict[str, Any] = {}
    if params["mode"] == "plain":
        (out / "page.md").write_text(output, encoding="utf-8")  # native and primary, unchanged
        result: dict[str, Any] = {"status": "ok", "primary": "page.md"}
    else:
        (out / "output.md").write_text(output, encoding="utf-8")  # the native output, unchanged
        blocks = layout_blocks(output, image.width, image.height)
        layout = {"image": {"width": image.width, "height": image.height}, "blocks": blocks}
        (out / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=1), "utf-8")
        prediction = blocks_to_prediction(blocks, sample["id"], image)
        (out / "prediction.json").write_text(
            json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        result = {"status": "ok", "primary": "prediction.json"}
        extra = {
            "blocks": len(blocks),
            "labels": sorted({b["label"] for b in blocks if b["label"] is not None}),
            "blocks_without_box": sum("box_px" not in b for b in blocks),
            "continuation_blocks": sum(b.get("continues", False) for b in blocks),
            "unscored_blocks": sum(b["label"] in UNSCORED for b in blocks),
        }
    usage = response["usage"]
    prompt, generated = usage.get("prompt_tokens"), usage.get("completion_tokens")
    reason, repeats = response["finish_reason"], tail_repeats(output)
    context = CHECKPOINTS[params["checkpoint"]]["context"]
    if reason == "length":
        full = prompt is not None and generated is not None and prompt + generated >= context
        limit = f"the end of the {context}-token context" if full else "max_tokens"
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
        "sent_image": {"width": sent.width, "height": sent.height},
    } | extra
    return result | {"extra": extra}


def check_job_model(model: Any, checkpoint: str) -> None:
    """A configured model must be the checkpoint's pinned repository@revision, so that a preset
    naming one checkpoint never runs another."""
    spec = CHECKPOINTS[checkpoint]
    expected = f"{spec['repository']}@{spec['revision']}"
    if model is not None and model != expected:
        raise AdapterError(f"model {model!r} does not match checkpoint {checkpoint} ({expected})")


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
        check_job_model(job.get("model"), params["checkpoint"])
        info = environment(params["device"])
        files = check_model(params["checkpoint"])
        server = Server(params)
        info["server_command"] = server.command
        info["server_env"] = server.env
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot start LightOnOCR-3: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info: dict[str, Any] = {"job_model": job.get("model"), "files": files}
    if params:
        spec = CHECKPOINTS[params["checkpoint"]]
        model_info |= {
            "repository": spec["repository"],
            "revision": spec["revision"],
            "mode": params["mode"],
            "prompt": GROUNDING_PROMPT if params["mode"] == "grounding" else "",
            "chat_template_kwargs": NO_THINKING,
            "context": spec["context"],
            "longest_edge": longest_edge(params),
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
    print(f"lightonocr3 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
