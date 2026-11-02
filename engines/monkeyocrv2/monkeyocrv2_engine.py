"""MonkeyOCRv2 parsing engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, check the pinned
zenosai/MonkeyOCRv2-B-Parsing checkpoint (and, for the DFlash preset, the pinned
MonkeyOCRv2-B-Parsing-DFlash draft) in the /cache volume, and start upstream's ``parsing/serve.py``
(vLLM with upstream's MonkeyOCRv2 model plugin) as a server process inside the container,
reachable on the loopback interface only. Every sample is parsed in job order by upstream's
two-stage pipeline at the pinned source commit, without its dewarping step (upstream's
``--skip-preprocess``): one layout request for the whole page (layout blocks with labels and boxes
scaled to [0, 1000], in reading order), then one recognition request per block crop with the
block label's prompt, sent concurrently. The pipeline functions (image resizing, the prompts,
the tolerant layout parser, the box mapping, the crops and the block formatting) are imported
from upstream's ``core_runner``; the adapter sends the requests itself to record their stop
reasons and token counts. It writes $OCRBENCH_OUTPUT/<id>/ with the model outputs unchanged
(layout.txt and the raw block outputs in blocks.json), upstream's page record (monkeyocrv2.json)
and Markdown (page.md), prediction.json (the primary output: one region per layout block, its
box in input-image pixels and its formatted content as region markup, converted to text on the
host) and, last, result.json.
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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path
from typing import Any

# Upstream source (github.com/Yuliang-Liu/MonkeyOCRv2), checked against source.sha256 at build.
SOURCE_REPOSITORY = "Yuliang-Liu/MonkeyOCRv2"
SOURCE_COMMIT = "6cc0c0bb4864f2541fa6bc3a8f976cffaac8f359"
SOURCE = Path(os.environ.get("OCRBENCH_MONKEYOCRV2_SOURCE", "/opt/monkeyocrv2"))
# The weights are not in the image: they are read from the runner's /cache mount (README.md).
CACHE = Path(os.environ.get("OCRBENCH_MONKEYOCRV2_CACHE", "/cache"))
CHECKSUMS = Path(os.environ.get("OCRBENCH_MONKEYOCRV2_CHECKSUMS", "/opt/ocrbench"))
SERVER_LOG = Path(os.environ.get("OCRBENCH_MONKEYOCRV2_LOG", "/tmp/vllm-server.log"))
CHECKPOINTS: dict[str, dict[str, str]] = {
    "B-Parsing": {
        "repository": "zenosai/MonkeyOCRv2-B-Parsing",
        "revision": "e8871a0057c98378219b369f924a25ba10f392bf",
    },
    "B-Parsing-DFlash": {
        "repository": "zenosai/MonkeyOCRv2-B-Parsing-DFlash",
        "revision": "5e87f46698d698e971039168c01c34e044445a9b",
    },
}
TARGET, DRAFT = "B-Parsing", "B-Parsing-DFlash"
SERVED_MODEL = "MonkeyOCRv2"  # serve.py's and parse.py's default served model name
# serve.py's defaults, passed explicitly so that the command records them: context, scheduler
# token budget, concurrent sequences and the DFlash proposal block (the draft's block_size 16).
SERVER_SETTINGS = {
    "max_model_len": 16384,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 128,
    "dflash_num_speculative_tokens": 16,
}
# parse.py's and core_runner's defaults (README.md): every request image is resized to at most
# MAX_PIXELS; the layout image is also enlarged to at least LAYOUT_MIN_PIXELS. Greedy decoding.
PIPELINE = {
    "max_pixels": 1003520,
    "layout_min_pixels": 1003520,
    "layout_max_tokens": 4096,
    "recognition_max_tokens": 5000,
    "temperature": 0,
    "dewarp": False,  # upstream's --skip-preprocess: boxes are in input-image pixels
    "retry_repeat": False,  # parse.py's default: no resampling of repeated outputs
}
REQUEST_TIMEOUT = 300  # seconds, parse.py's --request-timeout
# Upstream layout label: region type (docs/PREDICTIONS.md#region-types); others: "other".
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
    "Formula": "formula",
    "Picture": "figure",
}
# Engine parameters (job.json "params"), README.md. gpu_memory_utilization defaults to serve.py's
# 0.5; 0.6 is the benchmark's limit for an in-container server (ADR 0012).
DEFAULT_PARAMS: dict[str, Any] = {
    "device": "cuda:0",
    "dflash": False,
    "gpu_memory_utilization": 0.5,
    "startup_seconds": 900,
}
SPEC_DECODE = re.compile(r"^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total\b")


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
    if type(merged["dflash"]) is not bool:
        raise AdapterError(f"dflash must be true or false: {merged['dflash']!r}")
    share = merged["gpu_memory_utilization"]
    if type(share) not in (int, float) or not 0 < share <= 0.6:
        raise AdapterError(f"gpu_memory_utilization must be in (0, 0.6]: {share!r}")
    startup = merged["startup_seconds"]
    if type(startup) not in (int, float) or startup <= 0:
        raise AdapterError(f"startup_seconds must be a positive number: {startup!r}")
    return merged


def pinned_dir(checkpoint: str) -> str:
    return f"models/monkeyocrv2/MonkeyOCRv2-{checkpoint}/{CHECKPOINTS[checkpoint]['revision']}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_model(checkpoint: str) -> dict[str, str]:
    """SHA-256 of the files of a pinned checkpoint directory, which must hold exactly the files of
    models-<checkpoint>.sha256, all matching."""
    directory = CACHE / pinned_dir(checkpoint)
    if not directory.is_dir():
        raise AdapterError(
            f"model directory {directory} does not exist; download the weights with "
            "make monkeyocrv2-model and run with --cache (README.md)"
        )
    files = {
        p.relative_to(directory).as_posix(): sha256(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }
    checksums = CHECKSUMS / f"models-{checkpoint}.sha256"
    pinned = dict(
        reversed(line.split()) for line in checksums.read_text().splitlines() if line.strip()
    )
    bad = sorted(name for name, digest in pinned.items() if files.get(name) != digest)
    if bad:
        raise AdapterError(f"{checkpoint} files missing or not matching {checksums.name}: {bad}")
    extra = sorted(set(files) - set(pinned))
    if extra:
        raise AdapterError(f"{checkpoint} files not in {checksums.name}: {extra}")
    return files


def environment(device: str) -> dict[str, Any]:
    """Versions and GPU diagnostics; raises AdapterError when the requested GPU is unusable."""
    try:
        import torch
    except Exception as error:
        raise AdapterError(f"cannot import PyTorch: {error}") from error
    info: dict[str, Any] = {name: metadata.version(name) for name in ("vllm", "transformers")}
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
    """Upstream's serve.py: it imports upstream's vLLM model plugin, then runs ``vllm serve`` with
    ``--trust-remote-code`` (for the checkpoint's configuration and processor classes) and, with
    a draft, DFlash speculative decoding."""
    settings = SERVER_SETTINGS
    command = [sys.executable, str(SOURCE / "parsing" / "serve.py")]
    command += ["--model-path", str(CACHE / pinned_dir(TARGET))]
    command += ["--host", "127.0.0.1", "--port", str(port)]
    command += ["--served-model-name", SERVED_MODEL, "--tensor-parallel-size", "1"]
    command += ["--gpu-memory-utilization", str(params["gpu_memory_utilization"])]
    command += ["--max-model-len", str(settings["max_model_len"])]
    command += ["--max-num-seqs", str(settings["max_num_seqs"])]
    command += ["--max-num-batched-tokens", str(settings["max_num_batched_tokens"])]
    if params["dflash"]:
        command += ["--draft-model", str(CACHE / pinned_dir(DRAFT))]
        command += [
            "--dflash-num-speculative-tokens",
            str(settings["dflash_num_speculative_tokens"]),
        ]
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
    """Upstream's vLLM server as a child process on the loopback interface. The container has no
    network; the server only serves this adapter."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.port = free_port()
        self.command = server_command(params, self.port)
        self.env = server_env(params)
        with SERVER_LOG.open("wb") as log:
            self.process = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT,
                env=os.environ | self.env, start_new_session=True, cwd=SOURCE / "parsing",
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

    def spec_decode(self) -> dict[str, float]:
        """vLLM's speculative decoding counters (drafts, draft and accepted tokens) so far."""
        counters = {"drafts": 0.0, "draft_tokens": 0.0, "accepted_tokens": 0.0}
        with urllib.request.urlopen(f"{self.url}/metrics", timeout=10) as response:
            for line in response.read().decode(errors="replace").splitlines():
                match = SPEC_DECODE.match(line)
                if match:
                    counters[match.group(1)] += float(line.rsplit(" ", 1)[1])
        return counters

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)


def upstream() -> Any:
    """Upstream's parsing/core_runner module at the pinned commit (the pipeline functions)."""
    parsing = str(SOURCE / "parsing")
    if parsing not in sys.path:
        sys.path.insert(0, parsing)
    import core_runner

    return core_runner


def read_image(path: Path) -> Any:
    """The image as stored, as a new RGB image (EXIF orientation is not applied,
    docs/PREDICTIONS.md#positions; upstream applies it only to images in JPEG format)."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except Exception as error:
        raise AdapterError(f"cannot decode image {path.name}: {error}") from error


class Client:
    """Chat completion requests as upstream's MonkeyOCRv2_ServerParsing sends them (the image as a
    PNG data URL before the prompt, temperature 0, max_tokens), recording each request's stop
    reason, token counts, image size and time. Requests are not retried."""

    def __init__(self, url: str, runner: Any) -> None:
        self.url, self.runner = url, runner

    def request(self, image: Any, prompt: str, max_tokens: int, **resize: int) -> dict[str, Any]:
        sent = self.runner.load_image(image, **resize)  # upstream's resizing
        body = {
            "model": SERVED_MODEL,
            "temperature": PIPELINE["temperature"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self.runner.image_to_png_data_uri(sent)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as reply:
                answer = json.loads(reply.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise AdapterError(f"request failed: HTTP {error.code}: {detail}") from None
        choice = answer["choices"][0]
        usage = answer.get("usage") or {}
        return {
            "output": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "generated_tokens": usage.get("completion_tokens"),
            "image": list(sent.size),
            "seconds": round(time.monotonic() - start, 3),
        }


def layout(runner: Any, client: Client, image: Any) -> tuple[dict, list[dict]]:
    """Upstream's get_layout for one page: the layout request, the tolerant parse of its output
    (a list of {"bbox", "label"} items) and every box mapped from [0, 1000] to the input image
    (rounded to whole pixels and clipped, as upstream crops)."""
    reply = client.request(
        image,
        runner.ALL_PROMPT["LAYOUT"],
        PIPELINE["layout_max_tokens"],
        max_pixels=PIPELINE["max_pixels"],
        min_pixels=PIPELINE["layout_min_pixels"],
    )
    items = runner._parse_tolerant_items(
        reply["output"], lambda item: runner._normalize_model_item(item, include_content=False)
    )
    width, height = image.size
    blocks = [
        {
            "label": item["label"],
            "box": item["bbox"],
            "bbox": runner._map_bbox_to_image(item["bbox"], width, height),
        }
        for item in items
    ]
    return reply, blocks


def recognize(runner: Any, client: Client, image: Any, blocks: list[dict]) -> list[dict]:
    """Upstream's per-block recognition: a crop of every block whose label has a prompt, sent
    concurrently (at most the server's max_num_seqs at a time) with its label's prompt, and the
    output formatted as upstream's pipeline does (titles and section headers as Markdown
    headings, formulas as $$ blocks, OTSL tables as HTML). Other labels (Picture, Footnote,
    unknown ones) are not recognized and have no content."""
    tasks = runner._build_page_tasks(
        0, image, [{"bbox": b["bbox"], "label": b["label"]} for b in blocks]
    )
    todo = [task for task in tasks if task["need_infer"]]

    def run(task: dict) -> dict:
        return client.request(
            task["image"],
            task["question"],
            PIPELINE["recognition_max_tokens"],
            max_pixels=PIPELINE["max_pixels"],
        )

    with ThreadPoolExecutor(
        max_workers=max(1, min(len(todo), SERVER_SETTINGS["max_num_seqs"]))
    ) as pool:
        replies = dict(zip((t["block_idx"] for t in todo), pool.map(run, todo), strict=True))
    records = []
    for task in tasks:
        record = {"index": task["block_idx"], "label": task["label"], "bbox": task["bbox"]}
        reply = replies.get(task["block_idx"])
        if reply is not None:
            fields = runner._format_block_fields(task, reply["output"], None, None, True, None)
            record |= {"request": reply, "content": fields["content"]}
            if "otsl" in fields:
                record["otsl"] = fields["otsl"]
            record["repeated"] = bool(runner._should_retry_repeat_output(reply["output"]))
        records.append(record)
    return records


def records_to_prediction(records: list[dict], sample_id: str, image: Any) -> dict:
    """Standardized prediction (docs/PREDICTIONS.md): one region per layout block, in layout
    order, the reading order the model gives. A block's box is its polygon in input-image pixels;
    its formatted content (Markdown, HTML tables) is region markup, converted on the host."""
    regions: list[dict[str, Any]] = []
    for record in records:
        x0, y0, x1, y1 = record["bbox"]
        region: dict[str, Any] = {
            "id": f"b{len(regions) + 1}",
            "type": REGION_TYPES.get(record["label"], "other"),
            "native_type": record["label"],
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        }
        if (record.get("content") or "").strip():
            region["markup"] = {"format": "markdown", "content": record["content"]}
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


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def process(server: Any, params: dict, sample: dict, source: Path, out: Path) -> dict:
    """Parse one sample, write its outputs and return ``result.json`` content."""
    runner = upstream()
    image = read_image(source / sample["image"])
    client = Client(server.url, runner)
    before = server.spec_decode() if params["dflash"] else None
    start = time.monotonic()
    reply, blocks = layout(runner, client, image)
    layout_seconds = time.monotonic() - start
    (out / "layout.txt").write_text(reply["output"], encoding="utf-8")  # native, unchanged
    extra: dict[str, Any] = {"layout": {k: v for k, v in reply.items() if k != "output"}}
    if reply["finish_reason"] not in ("stop", "length"):
        return {
            "status": "failed",
            "message": f"layout request failed (finish_reason {reply['finish_reason']!r})",
            "extra": extra,
        }
    start = time.monotonic()
    records = recognize(runner, client, image, blocks)
    recognition_seconds = time.monotonic() - start
    write_json(
        out / "blocks.json",
        {
            "image": {"width": image.width, "height": image.height},
            "blocks": [
                {**block, **{k: v for k, v in record.items() if k not in ("label", "bbox")}}
                | ({"output": record["request"]["output"]} if "request" in record else {})
                for block, record in zip(blocks, records, strict=True)
            ],
        },
    )
    # Upstream's page record and Markdown, as its pipeline writes them for an image.
    page = [
        {"bbox": r["bbox"], "label": r["label"], "content": r.get("content", "")} for r in records
    ]
    for item, record in zip(page, records, strict=True):
        if "otsl" in record:
            item["otsl"] = record["otsl"]
    write_json(
        out / "monkeyocrv2.json",
        {"image_name": sample["image"], "image_size": [image.width, image.height], "layouts": page},
    )
    (out / "page.md").write_text(runner.result2md([sample["id"]], [page])[0], encoding="utf-8")
    write_json(out / "prediction.json", records_to_prediction(records, sample["id"], image))

    requests = [r["request"] for r in records if "request" in r]
    reasons = [r["finish_reason"] for r in requests]
    cut = sum(reason == "length" for reason in reasons)
    result: dict[str, Any] = {"status": "ok", "primary": "prediction.json"}
    messages = []
    if reply["finish_reason"] == "length":
        messages.append(
            f"the layout output stopped at max_tokens ({PIPELINE['layout_max_tokens']} tokens); "
            f"{len(blocks)} blocks were parsed from it"
        )
    if cut:
        messages.append(
            f"{cut} of {len(requests)} block outputs stopped at max_tokens "
            f"({PIPELINE['recognition_max_tokens']} tokens)"
        )
    failed = [r for r in reasons if r not in ("stop", "length")]
    if failed:
        result = {
            "status": "failed",
            "message": f"block recognition failed (finish_reason {failed[0]!r})",
        }
    elif messages:
        result |= {"status": "truncated", "message": "; ".join(messages)}
    elif not blocks:
        result["message"] = "the layout output has no blocks"
    generated = sum(r["generated_tokens"] or 0 for r in requests)
    extra |= {
        "blocks": len(blocks),
        "labels": sorted({b["label"] for b in blocks}),
        "recognized_blocks": len(requests),
        "truncated_blocks": cut,
        "repeated_blocks": sum(r.get("repeated", False) for r in records),
        "recognition_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in requests),
        "recognition_generated_tokens": generated,
        "layout_seconds": round(layout_seconds, 3),
        "recognition_seconds": round(recognition_seconds, 3),
        "source_image": {"width": image.width, "height": image.height},
    }
    if before is not None:
        after = server.spec_decode()
        delta = {key: after[key] - before[key] for key in after}
        drafted = delta["draft_tokens"]
        delta["acceptance_rate"] = round(delta["accepted_tokens"] / drafted, 4) if drafted else None
        extra["spec_decode"] = delta
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
    server, params, info, files, init_error = None, None, {}, {}, None
    try:
        params = parse_params(job.get("params") or {})
        info = environment(params["device"])
        files[TARGET] = check_model(TARGET)
        if params["dflash"]:
            files[DRAFT] = check_model(DRAFT)
        upstream()  # fail early when the pipeline cannot be imported
        server = Server(params)
        info["server_command"] = server.command
        info["server_env"] = server.env
    except AdapterError as error:
        init_error = str(error)
    except Exception as error:
        init_error = f"cannot start MonkeyOCRv2: {type(error).__name__}: {error}"
    info["load_seconds"] = round(time.monotonic() - start, 3)
    model_info = {
        "checkpoints": {
            name: CHECKPOINTS[name] | {"files": files.get(name)}
            for name in ([TARGET, DRAFT] if params and params["dflash"] else [TARGET])
        },
        "source": {"repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT},
        "job_model": job.get("model"),
        "pipeline": PIPELINE,
        "server": SERVER_SETTINGS,
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
                result = process(server, params, sample, source, out)
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
    print(f"monkeyocrv2 adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
