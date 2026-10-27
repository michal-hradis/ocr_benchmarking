"""Nemotron 3.5 Parse page parsing engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, check the pinned
nvidia/Nemotron-3.5-Parse checkpoint in the /cache volume (models.sha256) and start vLLM on it, with
NVIDIA's nemotron-parse plugin, as a server process inside the container, reachable on the loopback
interface only. Every sample is parsed in job order by NVIDIA's DocumentParser (nemotron_parse
package): the full page in one greedy request with the model card's prompt and, in the double mode,
one more request per predicted Table or Chart crop. The model's blocks (class, box normalized to the
page and text, in model order) are rendered by NVIDIA's postprocessing. It writes
$OCRBENCH_OUTPUT/<id>/ with the full-page model output unchanged (output.txt), all stage responses
and the rendered document (nemotron.json, page.md), prediction.json (the primary output: one
region per block, its box in input-image pixels and its Markdown as region markup, converted to
text on the host) and, last, result.json with the token counts, the stop reasons and the timing.
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
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

REPOSITORY = "nvidia/Nemotron-3.5-Parse"
REVISION = "62cb1e4ba153d113ff64aaf2be6e068c377cf53c"
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_NEMOTRON_CACHE", "/cache"))
PINNED_DIR = f"models/nemotron-parse/Nemotron-3.5-Parse/{REVISION}"
CHECKSUMS = Path(os.environ.get("OCRBENCH_NEMOTRON_CHECKSUMS", "/opt/ocrbench/models.sha256"))
SERVER_LOG = Path(os.environ.get("OCRBENCH_NEMOTRON_LOG", "/tmp/vllm-server.log"))
SERVED_MODEL = "nemotron-parse"
# The server settings of the model's docs/serving.md (bfloat16, a 16,384-token context, no prefix
# caching, one image per request, its chat template) except the H100 throughput settings: fewer
# concurrent sequences, as pages are sent one at a time, and no FlashAttention 3 by default.
SERVER_ARGS = [
    "--served-model-name", SERVED_MODEL,
    "--trust-remote-code",
    "--dtype", "bfloat16",
    "--distributed-executor-backend", "uni",
    "--tensor-parallel-size", "1",
    "--max-model-len", "16384",
    "--max-num-batched-tokens", "16384",
    "--no-enable-prefix-caching",
    "--limit-mm-per-prompt", json.dumps({"image": 1}),
]  # fmt: skip
CONTEXT = 16384
# The model card's generation contract for the full page: greedy decoding (temperature 0, top-k 1)
# with repetition penalty 1.1. DocumentParser's crop stage keeps its repetition penalty 1.0 and is
# made greedy as well; its own defaults sample (README.md).
FULLPAGE = {"temperature": 0.0, "top_k": 1, "repetition_penalty": 1.1}
CROP = {"temperature": 0.0, "top_k": 1, "repetition_penalty": 1.0}
# Of two overlapping predictions of a Table or Chart from the full page and its crop, the double
# mode keeps one (DocumentParser's "guarded" policy) rather than both ("alternatives"), so the
# text is not counted twice; the other is kept in nemotron.json (README.md).
MERGE_POLICY = "guarded"
# Nemotron 3.5 Parse class: region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "Text": "text",
    "Title": "heading",
    "Section-header": "heading",
    "List-item": "list-item",
    "Caption": "caption",
    "Footnote": "footnote",
    "Page-header": "page-header",
    "Page-footer": "page-footer",
    "Table": "table",
    "Chart": "figure",
    "Picture": "figure",
    "Formula": "formula",
    "Bibliography": "text",
    "TOC": "text",
    "Code": "other",
}
# Engine parameters (job.json "params"), README.md. max_tokens is the model card's 8,744;
# gpu_memory_utilization is the benchmark's limit for an in-container server (ADR 0012).
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "mode": "single",
    "max_tokens": 8744,
    "gpu_memory_utilization": 0.6,
    "max_num_seqs": 16,
    "mtp": True,
    "flash_attn_version": None,
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
    if merged["mode"] not in ("single", "double"):
        raise AdapterError(f"mode must be 'single' or 'double': {merged['mode']!r}")
    tokens = merged["max_tokens"]
    if type(tokens) is not int or not 1 <= tokens < CONTEXT:
        raise AdapterError(
            f"max_tokens must be an integer from 1 to below the {CONTEXT}-token context: {tokens!r}"
        )
    share = merged["gpu_memory_utilization"]
    if type(share) not in (int, float) or not 0 < share <= 0.6:
        raise AdapterError(f"gpu_memory_utilization must be in (0, 0.6]: {share!r}")
    seqs = merged["max_num_seqs"]
    if type(seqs) is not int or seqs < 1:
        raise AdapterError(f"max_num_seqs must be a positive integer: {seqs!r}")
    if type(merged["mtp"]) is not bool:
        raise AdapterError(f"mtp must be true or false: {merged['mtp']!r}")
    if merged["flash_attn_version"] not in (None, 2, 3):
        raise AdapterError(
            f"flash_attn_version must be 2 or 3 (or absent): {merged['flash_attn_version']!r}"
        )
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
            "make nemotron-parse-model and run with --cache (README.md)"
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
        name: metadata.version(name)
        for name in ("vllm", "nemotron-parse", "transformers", "huggingface-hub")
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
    command += [*SERVER_ARGS, "--chat-template", str(directory / "chat_template.jinja")]
    command += ["--max-num-seqs", str(params["max_num_seqs"])]
    command += ["--gpu-memory-utilization", str(params["gpu_memory_utilization"])]
    if params["flash_attn_version"]:
        attention = {"flash_attn_version": params["flash_attn_version"]}
        command += ["--attention-config", json.dumps(attention)]
    return command


def server_env(params: dict[str, Any]) -> dict[str, str]:
    """The plugin enables one-token MTP unless NP_VLLM_MTP=0 (the model's docs/serving.md).
    FlashInfer's top-k/top-p sampler would be compiled at start-up with nvcc, which the image does
    not have; vLLM's PyTorch sampler is used instead (README.md)."""
    return {
        "CUDA_VISIBLE_DEVICES": params["device"].split(":", 1)[1],
        "NP_VLLM_MTP": "1" if params["mtp"] else "0",
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


class Backend:
    """DocumentParser's backend: the chat completion request of DocumentParser's own HTTP client
    (nemotron_parse/inference.py: the prompt, then the image as PNG, the stage's sampling settings
    and skip_special_tokens false), sent to the server. Every response is recorded, so that the
    stages before a failed or truncated one are kept."""

    def __init__(self, url: str) -> None:
        self.url = f"{url}/v1/chat/completions"
        self.calls: list[dict[str, Any]] = []

    def __call__(self, image: Any, *, prompt: str, stage: str, settings: Any) -> dict[str, Any]:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
        payload = {
            "model": SERVED_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            **asdict(settings),
            "skip_special_tokens": False,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request) as reply:
                body = json.loads(reply.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise AdapterError(f"{stage} request failed: HTTP {error.code}: {detail}") from None
        choice = body["choices"][0]
        response = {
            "raw_out": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage") or {},
        }
        self.calls.append(
            {
                "stage": stage,
                **response,
                "image": {"width": image.width, "height": image.height},
                "seconds": round(time.monotonic() - start, 3),
            }
        )
        return response


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


def make_parser(backend: Backend, params: dict[str, Any]) -> Any:
    from nemotron_parse.inference import DocumentParser, GenerationSettings

    tokens = params["max_tokens"]
    return DocumentParser(
        backend=backend,
        fullpage=GenerationSettings(**FULLPAGE, max_tokens=tokens),
        crop=GenerationSettings(**CROP, max_tokens=tokens),
        merge_policy=MERGE_POLICY,
    )


def parse_page(parser: Any, backend: Backend, image: Any, mode: str) -> tuple[dict, dict]:
    """The rendered document of one page and the sample's status. A full-page output cut off at
    max_tokens is used as far as its blocks are complete, which may be none ("truncated"); so is
    the full page when a
    crop is cut off, without crop refinement. An empty full-page output is an empty page."""
    from nemotron_parse import postprocess_document
    from nemotron_parse.inference import GenerationError

    try:
        return parser.parse(image, mode=mode), {"status": "ok"}
    except GenerationError as error:
        response, stage = error.response, error.stage
        reason = response.get("finish_reason")
        tokens = (response.get("usage") or {}).get("completion_tokens")
        if reason == "length":
            what = "the page" if stage == "fullpage" else "a Table/Chart crop"
            status = {
                "status": "truncated",
                "message": f"generation of {what} stopped at max_tokens ({tokens} tokens)"
                + ("; the full page is used without crop refinement" if stage == "crop" else ""),
            }
        elif stage == "fullpage" and reason == "stop" and not response["raw_out"].strip():
            status = {"status": "ok", "message": "the model returned an empty page"}
        else:
            raise AdapterError(f"{stage} generation failed (finish_reason {reason!r})") from error
    fullpage = next(call for call in backend.calls if call["stage"] == "fullpage")
    document = postprocess_document(
        fullpage["raw_out"], image_size=image.size, options=parser.options
    )
    return document, status


def document_to_prediction(document: dict[str, Any], sample_id: str, image: Any) -> dict:
    """Standardized prediction (docs/PREDICTIONS.md) from NVIDIA's rendered document: one region per
    block, in model order, the reading order. Boxes are normalized to the page content and are
    mapped to input-image pixels by the postprocessing (bbox_pixels, clipped to the image). The
    rendered block content (Markdown, tables as HTML) is region markup, converted on the host."""
    regions: list[dict[str, Any]] = []
    for block in document["blocks"]:
        x0, y0, x1, y1 = block["bbox_pixels"]
        region: dict[str, Any] = {
            "id": f"b{len(regions) + 1}",
            "type": REGION_TYPES.get(block["class"], "other"),
            "native_type": block["class"],
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        }
        if block["content"].strip():
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
    backend = Backend(server_url)
    parser = make_parser(backend, params)
    try:
        document, result = parse_page(parser, backend, image, params["mode"])
    finally:
        # The native outputs are written even when the sample fails.
        fullpage = next((c for c in backend.calls if c["stage"] == "fullpage"), None)
        if fullpage:
            (out / "output.txt").write_text(fullpage["raw_out"], encoding="utf-8")
        native = {"stages": backend.calls}
        (out / "nemotron.json").write_text(
            json.dumps(native, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    if result["status"] == "ok" and fullpage["raw_out"].strip() and not document["blocks"]:
        raise AdapterError("the output has no layout blocks (output.txt)")
    native |= {key: value for key, value in document.items() if key != "stages"}
    (out / "nemotron.json").write_text(
        json.dumps(native, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "page.md").write_text(document["markdown"], encoding="utf-8")
    prediction = document_to_prediction(document, sample["id"], image)
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    usage = [call["usage"] for call in backend.calls]
    extra = {
        "mode": params["mode"],
        "requests": len(backend.calls),
        "crop_requests": sum(call["stage"] == "crop" for call in backend.calls),
        "stop_reasons": [call["finish_reason"] for call in backend.calls],
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "generated_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "fullpage_generated_tokens": fullpage["usage"].get("completion_tokens"),
        "max_tokens": params["max_tokens"],
        "source_image": {"width": image.width, "height": image.height},
        "blocks": len(document["blocks"]),
        "postprocessing_warnings": len(document.get("warnings") or []),
    }
    return result | {"primary": "prediction.json", "extra": extra}


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
        init_error = f"cannot start Nemotron 3.5 Parse: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    from_package = {}
    if not init_error:
        from nemotron_parse import PostprocessOptions
        from nemotron_parse.inference import PROMPT

        from_package = {
            "prompt": PROMPT,
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "postprocessing": asdict(PostprocessOptions()),
        }
    model_info = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "files": files,
        "job_model": job.get("model"),
        "fullpage_sampling": FULLPAGE,
        "crop_sampling": CROP,
        "merge_policy": MERGE_POLICY,
        **from_package,
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
    print(f"nemotron-parse adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
