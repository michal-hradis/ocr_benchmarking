"""Nanonets-OCR2-3B page-image to Markdown engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, check the pinned
checkpoint in the /cache volume (models.sha256) and start vLLM on it as a server process inside the
container, reachable on the loopback interface only. Every sample is sent in job order as one
request of the model card's vLLM example: one user message with the page image, as stored and not
resized by the adapter (the model's Qwen2.5-VL processor resizes it to its own pixel limits), then
the official prompt (prompt.txt); greedy decoding, max_tokens 15000 and the checkpoint's
repetition penalty. The model's Markdown is written unchanged to $OCRBENCH_OUTPUT/<id>/page.md (the
primary output, converted to plain text on the host) and, last, result.json with the token counts,
the stop reason, the repetition check and the timing. The model gives no positions, so the
prediction has none.
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

REPOSITORY = "nanonets/Nanonets-OCR2-3B"
REVISION = "c3886ff00bb037ce7da24988c9eafaf1fe2bed72"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_NANONETS_OCR2_CACHE", "/cache"))
PINNED_DIR = f"models/nanonets-ocr2/Nanonets-OCR2-3B/{REVISION}"  # relative to the cache
CHECKSUMS = Path(os.environ.get("OCRBENCH_NANONETS_OCR2_CHECKSUMS", "/opt/ocrbench/models.sha256"))
SERVER_LOG = Path(os.environ.get("OCRBENCH_NANONETS_OCR2_LOG", "/tmp/vllm-server.log"))
SERVED_MODEL = "nanonets-ocr2"
# The official OCR instruction of the model card and the cookbook (image2md.ipynb), byte for byte;
# it follows the image in one user turn. The chat template adds the default system message.
PROMPT = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
# The server's context: the largest image the processor makes (max_pixels 12,845,056 = 16,384
# image tokens of 28 x 28 pixels), the prompt and the model card's 15,000 generated tokens fit.
CONTEXT = 32768
SERVER_ARGS = [
    "--served-model-name", SERVED_MODEL,
    "--dtype", "bfloat16",
    "--distributed-executor-backend", "uni",
    "--tensor-parallel-size", "1",
    "--limit-mm-per-prompt", json.dumps({"image": 1}),
    "--max-model-len", str(CONTEXT),
]  # fmt: skip
# The output is reported as ending in a repetition loop when its tail is a unit repeated in a row:
# SHORT_TIMES times for a unit of fewer than LONG_UNIT characters, LONG_TIMES times for a longer
# one (a looping sentence or paragraph) of at most LONG_MAX characters.
SHORT_TIMES, LONG_TIMES, LONG_UNIT, LONG_MAX = 8, 3, 64, 4096
# Engine parameters (job.json "params"), README.md. max_tokens and the greedy decoding are the
# model card's; the repetition penalty is the checkpoint's generation_config.json, which vLLM and
# transformers apply by default and the adapter sends explicitly; gpu_memory_utilization is the
# benchmark's limit for an in-container server (ADR 0012).
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "max_tokens": 15000,
    "temperature": 0.0,
    "repetition_penalty": 1.05,
    "seed": 0,
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
    if type(tokens) is not int or not 1 <= tokens < CONTEXT:
        raise AdapterError(
            f"max_tokens must be an integer from 1 to below the {CONTEXT}-token context: "
            f"{tokens!r}"
        )
    temperature = merged["temperature"]
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise AdapterError(f"temperature must be in [0, 2]: {temperature!r}")
    penalty = merged["repetition_penalty"]
    if type(penalty) not in (int, float) or not 0 < penalty <= 2:
        raise AdapterError(f"repetition_penalty must be in (0, 2]: {penalty!r}")
    if type(merged["seed"]) is not int:
        raise AdapterError(f"seed must be an integer: {merged['seed']!r}")
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
            "make nanonets-ocr2-model and run with --cache (README.md)"
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


def check_job_model(model: Any) -> None:
    """A configured model must be the pinned repository@revision."""
    expected = f"{REPOSITORY}@{REVISION}"
    if model is not None and model != expected:
        raise AdapterError(f"model {model!r} is not the pinned checkpoint {expected}")


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


def payload(image: Any, params: dict[str, Any]) -> dict[str, Any]:
    """The model card's vLLM request: one user message with the image (lossless PNG at its stored
    size) and then the prompt, greedy decoding and max_tokens, plus the checkpoint's repetition
    penalty and a fixed seed, sent explicitly so that they do not depend on server defaults."""
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    content = [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": PROMPT},
    ]
    return {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
        "repetition_penalty": params["repetition_penalty"],
        "seed": params["seed"],
    }


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


def process(server_url: str, params: dict, sample: dict, source: Path, out: Path) -> dict:
    """OCR one sample, write its outputs and return ``result.json`` content."""
    image = read_image(source / sample["image"])
    response = generate(server_url, image, params)
    output = response["text"]
    (out / "page.md").write_text(output, encoding="utf-8")  # native and primary, unchanged
    result: dict[str, Any] = {"status": "ok", "primary": "page.md"}
    usage = response["usage"]
    prompt, generated = usage.get("prompt_tokens"), usage.get("completion_tokens")
    reason, repeats = response["finish_reason"], tail_repeats(output)
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
        check_job_model(job.get("model"))
        info = environment(params["device"])
        files = check_model()
        server = Server(params)
        info["server_command"] = server.command
        info["server_env"] = server.env
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot start Nanonets-OCR2: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info: dict[str, Any] = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "job_model": job.get("model"),
        "files": files,
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
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
    print(f"nanonets-ocr2 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
