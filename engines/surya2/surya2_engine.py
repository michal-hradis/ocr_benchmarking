"""Surya 2 full-page OCR engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, check the pinned
surya-ocr-2 checkpoint in the /cache volume (models.sha256) and start vLLM on it as a server
process inside the container, reachable on the loopback interface only, with Surya's server
settings. For every sample in job order it sends Surya's full-page OCR request (Surya's prompt and
image scaling, one greedy request per page) and parses the output with Surya's parser into layout
blocks: a box, a label and the block's HTML, in reading order. It writes $OCRBENCH_OUTPUT/<id>/
with the model's output unchanged (output.html), prediction.json (the primary output: one region
per block, its box in input-image pixels and its HTML as region markup, converted to text on the
host) and, last, result.json with the token counts, the stop reason and the timing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from importlib import metadata
from pathlib import Path
from typing import Any

REPOSITORY = "datalab-to/surya-ocr-2"
REVISION = "3b3d4cdf88d6928b0acdc75181b13206ea67c4a3"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_SURYA2_CACHE", "/cache"))
PINNED_DIR = f"models/surya2/surya-ocr-2/{REVISION}"
CHECKSUMS = Path(os.environ.get("OCRBENCH_SURYA2_CHECKSUMS", "/opt/ocrbench/models.sha256"))
SERVER_LOG = Path(os.environ.get("OCRBENCH_SURYA2_LOG", "/tmp/vllm-server.log"))
# The vLLM server settings of Surya 0.22.1 (surya/inference/backends/vllm.py and settings.py) for
# its default GPU type, the RTX 4090 (24 GB): bfloat16, an 18,000-token context, prefix caching,
# CUDA graphs, the image pixel limits and multi-token prediction (MTP) with two speculative tokens.
SERVER_ARGS = [
    "--no-enforce-eager",
    "--max-num-seqs", "32",
    "--dtype", "bfloat16",
    "--max-model-len", "18000",
    "--max-num-batched-tokens", "8192",
    "--enable-prefix-caching",
    "--mm-processor-kwargs", json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
    "--served-model-name", REPOSITORY,
]  # fmt: skip
MTP_ARGS = ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 2})]
# Surya's client defaults (openai_client.chat_completions_batch): temperature 0 is greedy.
TEMPERATURE, TOP_P = 0.0, 0.1
# Surya 2 layout label: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "Caption": "caption",
    "Footnote": "footnote",
    "Equation-Block": "formula",
    "List-Group": "text",
    "Page-Header": "page-header",
    "Page-Footer": "page-footer",
    "Image": "figure",
    "Section-Header": "heading",
    "Table": "table",
    "Text": "text",
    "Complex-Block": "figure",
    "Figure": "figure",
    "Diagram": "figure",
    "Bibliography": "text",
}
# Engine parameters (job.json "params"), README.md. max_tokens is Surya's
# SURYA_MAX_TOKENS_FULL_PAGE, gpu_memory_utilization and mtp its VLLM_* defaults.
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "max_tokens": 12288,
    "gpu_memory_utilization": 0.85,
    "mtp": True,
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
    if type(tokens) is not int or not 1 <= tokens < 18000:
        raise AdapterError(
            f"max_tokens must be an integer from 1 to below the 18000-token context: {tokens!r}"
        )
    share = merged["gpu_memory_utilization"]
    if type(share) not in (int, float) or not 0 < share <= 1:
        raise AdapterError(f"gpu_memory_utilization must be in (0, 1]: {share!r}")
    if type(merged["mtp"]) is not bool:
        raise AdapterError(f"mtp must be true or false: {merged['mtp']!r}")
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
            "make surya2-model and run with --cache (README.md)"
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
        name: metadata.version(name) for name in ("vllm", "surya-ocr", "transformers", "openai")
    }
    info |= {"torch": torch.__version__, "cuda": torch.version.cuda, "device": device}
    index = int(device.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index >= count:
        raise AdapterError(f"{device} requested but {count} CUDA devices are visible")
    info["gpu"] = torch.cuda.get_device_name(index)
    info["gpu_memory_mb"] = torch.cuda.get_device_properties(index).total_memory >> 20
    return info


def server_command(params: dict[str, Any], port: int) -> list[str]:
    command = ["vllm", "serve", str(CACHE / PINNED_DIR), "--host", "127.0.0.1"]
    command += ["--port", str(port), *SERVER_ARGS]
    command += ["--gpu-memory-utilization", str(params["gpu_memory_utilization"])]
    return command + (MTP_ARGS if params["mtp"] else [])


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
        device = params["device"].split(":", 1)[1]
        env = os.environ | {"CUDA_VISIBLE_DEVICES": device}
        with SERVER_LOG.open("wb") as log:
            self.process = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT, env=env,
                start_new_session=True,
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
    """The image as stored, in RGB, as Surya's load_image (EXIF orientation is not applied,
    docs/PREDICTIONS.md#positions)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error
    return image if image.mode == "RGB" else image.convert("RGB")


def request(client: Any, params: dict[str, Any], image: Any) -> dict[str, Any]:
    """One greedy full-page OCR request, built as Surya's client builds it (openai_client
    _generate_one): the image scaled by scale_to_fit, sent as PNG before the prompt."""
    from surya.inference.backends.openai_client import _build_messages
    from surya.inference.prompts import HIGH_ACCURACY_BBOX_PROMPT
    from surya.inference.util import scale_to_fit

    start = time.monotonic()
    scaled = scale_to_fit(image)
    messages = _build_messages(scaled, HIGH_ACCURACY_BBOX_PROMPT)
    prepared = time.monotonic()
    completion = client.chat.completions.create(
        model=REPOSITORY,
        messages=messages,
        max_tokens=params["max_tokens"],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    finished = time.monotonic()
    choice = completion.choices[0]
    usage = completion.usage
    generated = usage.completion_tokens if usage else None
    return {
        "text": choice.message.content or "",
        "stop_reason": "end-token" if choice.finish_reason == "stop" else choice.finish_reason,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "generated_tokens": generated,
        "max_tokens": params["max_tokens"],
        "model_image": {"width": scaled.width, "height": scaled.height},
        "preprocess_seconds": round(prepared - start, 3),
        "generate_seconds": round(finished - prepared, 3),
        "tokens_per_second": round((generated or 0) / max(finished - prepared, 1e-9), 1),
    }


def parse_blocks(text: str) -> list[Any]:
    """Surya's parser for full-page output: the top-level <div data-label data-bbox> blocks with
    their inner HTML; a div without a label or a four-number box is left out."""
    from surya.inference.parsers import parse_full_page_html

    return parse_full_page_html(text)


def blocks_to_prediction(
    blocks: list[Any], sample_id: str, image: Any
) -> tuple[dict[str, Any], dict[str, int]]:
    """Standardized prediction (docs/PREDICTIONS.md) from the blocks of one page, and counts of
    what Surya's post-processing changed.

    As in Surya's full-page OCR (recognition._full_page_ocr): boxes are normalized to 0-1000 of
    the image size, the blocks of labels without text (Image, Figure, Diagram, Blank-Page) keep
    no HTML, and a text-like block over an essentially blank part of the page is dropped. Every
    remaining block becomes a region in output order, the reading order, with its box in
    input-image pixels and its HTML as region markup, which the host converts to text.
    """
    from surya.common.blank import is_blank_region
    from surya.inference.prompts import SKIP_OCR_LABELS
    from surya.layout.label import LAYOUT_PRED_RELABEL, TEXT_LABELS
    from surya.recognition import _crop_block
    from surya.settings import settings

    width, height = image.size
    regions: list[dict[str, Any]] = []
    counts = {"blank_blocks": 0, "skipped_blocks": 0}
    for block in blocks:
        x0, y0, x1, y1 = (v / settings.BBOX_SCALE for v in block.bbox)
        polygon = [
            [x0 * width, y0 * height],
            [x1 * width, y0 * height],
            [x1 * width, y1 * height],
            [x0 * width, y1 * height],
        ]
        canonical = LAYOUT_PRED_RELABEL.get(block.label, block.label)
        if canonical in TEXT_LABELS and is_blank_region(_crop_block(image, polygon)):
            counts["blank_blocks"] += 1
            continue
        region: dict[str, Any] = {
            "id": f"b{len(regions) + 1}",
            "type": REGION_TYPES.get(block.label, "other"),
            "native_type": block.label,
            "polygon": polygon,
        }
        if block.label in SKIP_OCR_LABELS:
            counts["skipped_blocks"] += 1
        elif block.html.strip():
            region["markup"] = {"format": "html", "content": block.html}
        regions.append(region)
    prediction: dict[str, Any] = {
        "format": "ocrbench.prediction",
        "version": 1,
        "sample_id": sample_id,
        "image": {"width": width, "height": height},
        "text": "",  # built on the host from the converted region markup
    }
    if regions:
        prediction["regions"] = regions
        prediction["reading_order"] = [region["id"] for region in regions]
    return prediction, counts


def process(client: Any, params: dict, sample: dict, source: Path, out: Path) -> dict[str, Any]:
    """OCR one sample, write its outputs and return ``result.json`` content."""
    from surya.recognition import _detect_repeat_loop

    image = read_image(source / sample["image"])
    generation = request(client, params, image)
    text = generation.pop("text")
    (out / "output.html").write_text(text, encoding="utf-8")
    try:
        blocks = parse_blocks(text)
    except Exception as error:
        raise AdapterError(f"Surya's parser cannot read the output: {error}") from error
    prediction, counts = blocks_to_prediction(blocks, sample["id"], image)
    generation |= {
        "source_image": {"width": image.width, "height": image.height},
        "blocks": len(blocks),
        **counts,
        # Surya would regenerate such a page in block mode; here it is only recorded (README.md).
        "repeat_loop": _detect_repeat_loop(text),
    }
    reason = generation["stop_reason"]
    if reason != "end-token":
        result: dict[str, Any] = {
            "status": "truncated",
            "message": f"generation stopped: {reason} ({generation['generated_tokens']} of "
            f"{params['max_tokens']} tokens); {len(blocks)} blocks are used",
        }
    elif text.strip() and not blocks:
        raise AdapterError("the output has no layout blocks (output.html)")
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
    server, client, params, info, files, init_error = None, None, None, {}, None, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        files = check_model()
        server = Server(params)
        info["server_command"] = server.command
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=f"{server.url}/v1", max_retries=0, timeout=None)
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot start Surya 2: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    from_surya = {}
    if not init_error:
        from surya.inference.prompts import HIGH_ACCURACY_BBOX_PROMPT

        from_surya = {
            "prompt": HIGH_ACCURACY_BBOX_PROMPT,
            "prompt_sha256": hashlib.sha256(HIGH_ACCURACY_BBOX_PROMPT.encode()).hexdigest(),
        }
    model_info = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": files,
        "job_model": job.get("model"),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        **from_surya,
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
                result = process(client, params, sample, source, out)
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
    print(f"surya2 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
