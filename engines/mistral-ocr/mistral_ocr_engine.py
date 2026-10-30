"""Mistral OCR 4.1 hosted OCR engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json and, for every
sample in job order, send the sample's image file unchanged (base64, with its own media type; no
PDF conversion, resizing or re-encoding) to Mistral's OCR API with the pinned model and request
options. Nothing else from the job is sent: no sample ID, metadata or file names. This is the only
network access, permitted by the runner's --allow-network (ADR 0013); the API key comes from the
MISTRAL_API_KEY environment variable and is never written anywhere.

It writes $OCRBENCH_OUTPUT/<id>/ with the API response body unchanged (response.json), the page's
Markdown (page.md) and, for the blocks mode, prediction.json: one region per returned block in
reading order, its box scaled to input-image pixels, its Markdown as region markup (converted to
text on the host) and its confidence. Last comes result.json with the request settings, usage,
estimated cost, HTTP attempts, rate-limit events and latency. API errors, throttling that outlasts
the retries and malformed responses fail the sample; an empty page transcription is a valid
result, and an output that ends in a repetition loop is reported as truncated.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = os.environ.get("OCRBENCH_MISTRAL_URL", "https://api.mistral.ai/v1/ocr")
# The pinned model identifier, never a "-latest" alias; the response's model must match it.
MODEL = "mistral-ocr-4-1"
KEY_VARIABLE = "MISTRAL_API_KEY"
# Image formats sent as stored (Pillow format -> media type). Others fail the sample: the
# benchmark does not convert images (ADR 0001).
MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "AVIF": "image/avif"}
# Mistral block type -> region type (docs/PREDICTIONS.md#region-types); others: "other".
REGION_TYPES = {
    "text": "text",
    "title": "heading",
    "list": "text",
    "table": "table",
    "image": "figure",
    "equation": "formula",
    "caption": "caption",
    "code": "text",
    "references": "text",
    "aside_text": "text",
    "header": "page-header",
    "footer": "page-footer",
}
# The API reports no stop reason. An output that ends in a repetition loop was cut off by the
# service and is reported as truncated: its tail is a unit repeated in a row, SHORT_TIMES times for
# a unit of fewer than LONG_UNIT characters, LONG_TIMES times for a longer one of at most LONG_MAX
# characters (the rule of the LightOnOCR-3 adapter).
SHORT_TIMES, LONG_TIMES, LONG_UNIT, LONG_MAX = 8, 3, 64, 4096
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
# Response headers kept as provenance: request IDs and rate-limit state.
KEPT_HEADERS = ("x-request-id", "mistral-correlation-id", "x-ratelimit", "ratelimit")
# Engine parameters (job.json "params"), README.md.
DEFAULT_PARAMS: dict[str, Any] = {
    "mode": "blocks",
    "table_format": None,  # tables inline in the page Markdown
    "extract_header": False,  # headers and footers stay in the transcription
    "extract_footer": False,
    "confidence_scores_granularity": "block",
    "request_timeout": 300,
    "max_retries": 4,
    "retry_base_seconds": 2.0,
    "min_interval_seconds": 1.0,
    "usd_per_1000_pages": 4.0,  # Mistral's list price for OCR 4.1, for the cost estimate only
}


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    if merged["mode"] not in ("blocks", "markdown"):
        raise AdapterError(f"mode must be 'blocks' or 'markdown': {merged['mode']!r}")
    if merged["table_format"] not in (None, "markdown", "html"):
        raise AdapterError(
            f"table_format must be null, 'markdown' or 'html': {merged['table_format']!r}"
        )
    for key in ("extract_header", "extract_footer"):
        if type(merged[key]) is not bool:
            raise AdapterError(f"{key} must be true or false: {merged[key]!r}")
    if merged["confidence_scores_granularity"] not in (None, "page", "block", "word"):
        raise AdapterError("confidence_scores_granularity must be null, page, block or word")
    for key in ("request_timeout", "retry_base_seconds", "usd_per_1000_pages"):
        if type(merged[key]) not in (int, float) or merged[key] <= 0:
            raise AdapterError(f"{key} must be a positive number: {merged[key]!r}")
    if (
        type(merged["min_interval_seconds"]) not in (int, float)
        or merged["min_interval_seconds"] < 0
    ):
        raise AdapterError("min_interval_seconds must be a number of at least 0")
    if type(merged["max_retries"]) is not int or not 0 <= merged["max_retries"] <= 10:
        raise AdapterError(
            f"max_retries must be an integer from 0 to 10: {merged['max_retries']!r}"
        )
    return merged


def request_body(params: dict[str, Any], media_type: str, data: bytes) -> dict[str, Any]:
    """The OCR request: the image as a base64 data URL, the pinned model and the options."""
    body: dict[str, Any] = {
        "model": MODEL,
        "document": {
            "type": "image_url",
            "image_url": f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}",
        },
        "include_image_base64": False,
        "include_blocks": params["mode"] == "blocks",
        "extract_header": params["extract_header"],
        "extract_footer": params["extract_footer"],
    }
    for key in ("table_format", "confidence_scores_granularity"):
        if params[key] is not None:
            body[key] = params[key]
    return body


def read_image(path: Path) -> tuple[bytes, str, int, int]:
    """The file's bytes as stored, its media type and its size in pixels."""
    from PIL import Image

    data = path.read_bytes()
    try:
        with Image.open(path) as image:
            kind, width, height = image.format, image.width, image.height
    except Exception as error:
        raise AdapterError(f"cannot read image {path.name}: {error}") from error
    if kind not in MEDIA_TYPES:
        raise AdapterError(
            f"image format {kind} is not sent: the adapter sends only {sorted(MEDIA_TYPES)} "
            "files as stored and does not convert images"
        )
    return data, MEDIA_TYPES[kind], width, height


class Client:
    """POSTs OCR requests with retries on throttling and transient errors, and spaces requests at
    least ``min_interval_seconds`` apart."""

    def __init__(self, params: dict[str, Any], key: str, url: str = API_URL) -> None:
        self.params = params
        self.key = key
        self.url = url
        self.last_start: float | None = None

    def post(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """The parsed response and the transport record; raises AdapterError on failure."""
        payload = json.dumps(body).encode("utf-8")
        attempts: list[dict[str, Any]] = []
        record: dict[str, Any] = {"attempts": attempts, "request_bytes": len(payload)}
        for attempt in range(self.params["max_retries"] + 1):
            self.throttle()
            start = time.monotonic()
            status, headers, data, error = self.send(payload)
            seconds = round(time.monotonic() - start, 3)
            attempts.append({"status": status, "seconds": seconds, "error": error})
            record["headers"] = {
                k: v for k, v in headers.items() if k.lower().startswith(KEPT_HEADERS)
            }
            if status == 200:
                record |= {"latency_seconds": seconds, "response_bytes": len(data)}
                try:
                    return json.loads(data.decode("utf-8")), record
                except ValueError as exc:
                    raise TransportError(f"the response is not JSON: {exc}", record) from exc
            retry = status is None or status in RETRY_STATUSES
            if not retry or attempt == self.params["max_retries"]:
                detail = error or data[:500].decode("utf-8", "replace")
                what = f"HTTP {status}" if status is not None else "request failed"
                raise TransportError(f"{what} after {len(attempts)} attempts: {detail}", record)
            time.sleep(self.delay(attempt, headers))
        raise AssertionError("unreachable")

    def throttle(self) -> None:
        if self.last_start is not None:
            wait = self.last_start + self.params["min_interval_seconds"] - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        self.last_start = time.monotonic()

    def delay(self, attempt: int, headers: dict[str, str]) -> float:
        """Retry-After when the API gives one (at most 120 s), else exponential backoff."""
        try:
            return min(float(headers.get("Retry-After", "")), 120.0)
        except ValueError:
            return self.params["retry_base_seconds"] * 2**attempt

    def send(self, payload: bytes) -> tuple[int | None, dict[str, str], bytes, str | None]:
        request = urllib.request.Request(
            self.url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.params["request_timeout"]) as reply:
                return reply.status, dict(reply.headers), reply.read(), None
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers or {}), error.read(), None
        except (OSError, TimeoutError) as error:  # URLError, connection reset, timeout
            return None, {}, b"", f"{type(error).__name__}: {error}"


class TransportError(AdapterError):
    """The request failed; ``record`` holds the attempts made."""

    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


def tail_repeats(text: str) -> bool:
    """Whether the text ends with a non-blank unit repeated in a row: SHORT_TIMES times for a unit
    shorter than LONG_UNIT characters, LONG_TIMES times for one of up to LONG_MAX characters."""
    for length in range(1, min(LONG_MAX, len(text) // LONG_TIMES) + 1):
        unit = text[-length:]
        times = SHORT_TIMES if length < LONG_UNIT else LONG_TIMES
        if unit.strip() and text.endswith(unit * times):
            return True
    return False


def repetition(page: dict[str, Any]) -> str | None:
    """Where the page's output ends in a repetition loop: the page Markdown or blocks; None if
    nowhere."""
    blocks = page.get("blocks") if isinstance(page.get("blocks"), list) else []
    looping = [
        str(index + 1)
        for index, block in enumerate(blocks)
        if isinstance(block, dict)
        and isinstance(block.get("content"), str)
        and tail_repeats(block["content"])
    ]
    where = ["the page Markdown"] if tail_repeats(page["markdown"]) else []
    if looping:
        where.append(f"block{'s' if len(looping) > 1 else ''} {', '.join(looping)}")
    return " and ".join(where) or None


def single_page(response: dict[str, Any]) -> dict[str, Any]:
    pages = response.get("pages")
    if not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict):
        count = len(pages) if isinstance(pages, list) else "no"
        raise AdapterError(f"expected one page in the response, got {count} (response.json)")
    page = pages[0]
    if not isinstance(page.get("markdown"), str):
        raise AdapterError("the page has no Markdown (response.json)")
    return page


def blocks_to_prediction(
    page: dict[str, Any], sample_id: str, width: int, height: int
) -> dict[str, Any]:
    """Standardized prediction (docs/PREDICTIONS.md) from the page's blocks: one region per block
    in response order, the reading order, with its box scaled from the page dimensions the API
    reports to input-image pixels, its Markdown as region markup (none for an image block, whose
    content only references the picture) and its average content confidence."""
    blocks = page.get("blocks")
    dimensions = page.get("dimensions") or {}
    page_w, page_h = dimensions.get("width"), dimensions.get("height")
    if not isinstance(blocks, list):
        raise AdapterError("the page has no blocks list (response.json)")
    if not all(type(v) in (int, float) and v > 0 for v in (page_w, page_h)):
        raise AdapterError(f"the page has no valid dimensions: {dimensions!r} (response.json)")
    sx, sy = width / page_w, height / page_h
    regions: list[dict[str, Any]] = []
    for block in blocks:
        try:
            x0, y0 = float(block["top_left_x"]) * sx, float(block["top_left_y"]) * sy
            x1, y1 = float(block["bottom_right_x"]) * sx, float(block["bottom_right_y"]) * sy
        except (KeyError, TypeError, ValueError) as error:
            raise AdapterError(f"a block has no valid box: {error} (response.json)") from error
        native = block.get("type")
        region: dict[str, Any] = {
            "id": f"b{len(regions) + 1}",
            "type": REGION_TYPES.get(native, "other"),
            "native_type": str(native),
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        }
        content = block.get("content")
        # An image block's content is only a reference to the extracted picture, not text.
        if native != "image" and isinstance(content, str) and content.strip():
            region["markup"] = {"format": "markdown", "content": content}
        scores = block.get("confidence_scores")
        if isinstance(scores, dict):
            average = scores.get("average_content_confidence_score")
            if type(average) in (int, float):
                region["confidence"] = average
            region["extra"] = {"confidence_scores": scores}
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
    return prediction


def process(
    client: Client, params: dict, sample: dict, source: Path, out: Path, extra: dict[str, Any]
) -> dict:
    """OCR one sample, write its outputs and return ``result.json`` content. What is known about
    the request and response is added to ``extra``, also when the sample fails."""
    data, media_type, width, height = read_image(source / sample["image"])
    body = request_body(params, media_type, data)
    extra |= {
        "image": {
            "media_type": media_type,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "width": width,
            "height": height,
        },
    }
    try:
        response, transport = client.post(body)
    except TransportError as error:
        extra["transport"] = error.record
        raise
    extra["transport"] = transport
    (out / "response.json").write_text(json.dumps(response, ensure_ascii=False), "utf-8")
    usage = response.get("usage_info") or {}
    pages = usage.get("pages_processed")
    extra |= {
        "response_model": response.get("model"),
        "usage": usage,
        "estimated_cost_usd": (
            round(pages * params["usd_per_1000_pages"] / 1000, 6) if type(pages) is int else None
        ),
    }
    if response.get("model") != MODEL:
        raise AdapterError(
            f"the response is from model {response.get('model')!r}, not the pinned {MODEL!r}"
        )
    page = single_page(response)
    markdown = page["markdown"]
    (out / "page.md").write_text(markdown, encoding="utf-8")
    loop = repetition(page)
    extra |= {
        "dimensions": page.get("dimensions"),
        "confidence_scores": page.get("confidence_scores"),
        "empty": not markdown.strip(),
        "tail_repetition": loop,
    }
    result: dict[str, Any] = {"status": "ok"}
    if loop:
        result = {"status": "truncated", "message": f"repetition loop at the end of {loop}"}
    if params["mode"] == "markdown":
        return result | {"primary": "page.md"}
    prediction = blocks_to_prediction(page, sample["id"], width, height)
    extra["blocks"] = len(prediction.get("regions", []))
    if markdown.strip() and not prediction.get("regions"):
        raise AdapterError("the page has Markdown but no blocks (response.json)")
    (out / "prediction.json").write_text(
        json.dumps(prediction, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return result | {"primary": "prediction.json"}


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    client, params, init_error = None, None, None
    try:
        params = parse_params(job.get("params") or {})
        if job.get("model") != MODEL:
            raise AdapterError(f"the job's model {job.get('model')!r} is not {MODEL!r}")
        key = os.environ.get(KEY_VARIABLE)
        if not key:
            raise AdapterError(f"{KEY_VARIABLE} is not set")
        client = Client(params, key)
    except AdapterError as error:
        init_error = str(error)
    settings = {"api_url": API_URL, "model": MODEL, "params": params}
    if params:
        settings["request"] = {
            k: v for k, v in request_body(params, "image/jpeg", b"").items() if k != "document"
        }
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        extra: dict[str, Any] = {}
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(client, params, sample, source, out, extra)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result["extra"] = extra | {"settings": settings}
        write_result(out, result)
    print(f"mistral-ocr adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
