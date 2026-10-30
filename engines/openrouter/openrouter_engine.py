"""OpenRouter hosted vision-model engine adapter; runs in the engine image (README.md).

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json and, for every
sample in job order, send the sample's image file unchanged (base64, with its own media type; no
resizing or re-encoding) and the preset's fixed prompt to OpenRouter's chat completions API, with
the job's pinned model ID, one pinned provider without fallbacks and fixed generation settings.
Nothing else from the job is sent: no sample ID, metadata or file names. This is the only network
access, permitted by the runner's --allow-network (ADR 0013); the API key comes from the
OPENROUTER_API_KEY environment variable and is never written anywhere.

It writes $OCRBENCH_OUTPUT/<id>/ with the request without the image data (request.json), the API
response body unchanged (response.json) and the model's answer (page.md, converted to text on the
host). Last comes result.json with the image, prompt and request fingerprints, the effective
provider and model, usage, cost, HTTP attempts and latency. API errors, throttling that outlasts
the retries, refusals and malformed responses fail the sample; an empty answer is a valid result,
and an answer cut off by the token limit or ending in a repetition loop is reported as truncated.
"""

from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

API_URL = os.environ.get("OCRBENCH_OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
KEY_VARIABLE = "OPENROUTER_API_KEY"
# Image formats sent as stored (Pillow format -> media type). Others fail the sample: the
# benchmark does not convert images (ADR 0001).
MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
# An answer that ends in a repetition loop is reported as truncated: its tail is a unit repeated in
# a row, SHORT_TIMES times for a unit of fewer than LONG_UNIT characters, LONG_TIMES times for a
# longer one of at most LONG_MAX characters (the rule of the LightOnOCR-3 adapter).
SHORT_TIMES, LONG_TIMES, LONG_UNIT, LONG_MAX = 8, 3, 64, 4096
# Retried: no HTTP response (network error or timeout), 408, 429 and any 5xx, also when OpenRouter
# reports it as an error object in a 200 response.
RETRY_STATUSES = {408, 429} | set(range(500, 600))
MAX_RETRY_AFTER = 120.0  # seconds; a longer Retry-After is shortened to this
# Response headers kept as provenance: request IDs and rate-limit state.
KEPT_HEADERS = ("x-request-id", "x-generation-id", "x-ratelimit", "ratelimit")
# Engine parameters (job.json "params"), README.md. provider and prompt have no default.
DEFAULT_PARAMS: dict[str, Any] = {
    "provider": None,
    "prompt": None,
    "max_tokens": 32768,
    "temperature": 0.0,
    "seed": 0,
    "reasoning_effort": None,
    "request_timeout": 300,
    "max_retries": 4,
    "retry_base_seconds": 2.0,
    "min_interval_seconds": 1.0,
}
REASONING_EFFORTS = (None, "none", "minimal", "low", "medium", "high")


class AdapterError(Exception):
    """A sample cannot be processed; the message is reported as the engine failure."""


def check_model(model: Any) -> str:
    """The job's model must be one exact OpenRouter model ID: no router, alias or variant."""
    if not isinstance(model, str) or model.count("/") != 1 or not all(model.split("/")):
        raise AdapterError(f"the job's model must be an OpenRouter ID 'vendor/model': {model!r}")
    if (
        model.startswith("~")
        or ":" in model
        or "latest" in model
        or model.startswith("openrouter/")
    ):
        raise AdapterError(
            f"the job's model {model!r} is a router, alias or variant, not one pinned model"
        )
    return model


def parse_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    if unknown:
        raise AdapterError(f"unknown parameters: {', '.join(unknown)}")
    merged = {**DEFAULT_PARAMS, **params}
    if not isinstance(merged["provider"], str) or not merged["provider"].strip():
        raise AdapterError("provider must name one OpenRouter provider, e.g. 'google-ai-studio'")
    if not isinstance(merged["prompt"], str) or not merged["prompt"].strip():
        raise AdapterError("prompt must be a non-empty string")
    if type(merged["max_tokens"]) is not int or merged["max_tokens"] < 16:
        raise AdapterError(
            f"max_tokens must be an integer of at least 16: {merged['max_tokens']!r}"
        )
    if type(merged["temperature"]) not in (int, float) or not 0 <= merged["temperature"] <= 2:
        raise AdapterError(f"temperature must be from 0 to 2: {merged['temperature']!r}")
    if merged["seed"] is not None and type(merged["seed"]) is not int:
        raise AdapterError(f"seed must be an integer or null: {merged['seed']!r}")
    if merged["reasoning_effort"] not in REASONING_EFFORTS:
        raise AdapterError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
    for key in ("request_timeout", "retry_base_seconds"):
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


def request_body(model: str, params: dict[str, Any], image_url: str) -> dict[str, Any]:
    """The chat request: one user message with the prompt and the image, the pinned model, one
    provider without fallbacks or providers that ignore a setting, and the generation settings."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": params["prompt"]},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "provider": {
            "only": [params["provider"]],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
        },
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "stream": False,
    }
    if params["seed"] is not None:
        body["seed"] = params["seed"]
    if params["reasoning_effort"] is not None:
        body["reasoning"] = {"effort": params["reasoning_effort"]}
    return body


def fingerprint(value: Any) -> str:
    """SHA-256 of a string, or of a JSON value in canonical form."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def settings_of(model: str, params: dict[str, Any]) -> dict[str, Any]:
    """The request without the image and its fingerprint, the same for every sample."""
    request = request_body(model, params, "<image>")
    return {
        "api_url": API_URL,
        "model": model,
        "params": params,
        "request": request,
        "request_sha256": fingerprint(request),
        "prompt_sha256": fingerprint(params["prompt"]),
    }


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
    """POSTs requests with retries on throttling and transient errors, and spaces requests at
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
            started = datetime.now(UTC).isoformat(timespec="milliseconds")
            start = time.monotonic()
            status, headers, data, error = self.send(payload)
            seconds = round(time.monotonic() - start, 3)
            attempts.append({"started": started, "status": status, "seconds": seconds})
            record["headers"] = {k: v for k, v in headers.items() if k.startswith(KEPT_HEADERS)}
            response = None
            if status == 200:
                try:
                    response = json.loads(data.decode("utf-8"))
                except ValueError as exc:
                    raise TransportError(f"the response is not JSON: {exc}", record) from exc
                # OpenRouter may answer 200 with an error object, e.g. a provider error.
                failed = body_error(response)
                if failed:
                    status, error = failed
                    attempts[-1]["error_in_body"] = status
                else:
                    record |= {"latency_seconds": seconds, "response_bytes": len(data)}
                    return response, record
            attempts[-1]["error"] = error
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
        """Retry-After when the API gives a valid one (at most MAX_RETRY_AFTER), else exponential
        backoff. ``headers`` has lower-case names."""
        wait = retry_after(headers.get("retry-after"))
        if wait is None:
            return self.params["retry_base_seconds"] * 2**attempt
        return min(wait, MAX_RETRY_AFTER)

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
                return reply.status, lower_names(reply.headers), reply.read(), None
        except urllib.error.HTTPError as error:
            return error.code, lower_names(error.headers), error.read(), None
        except (OSError, TimeoutError) as error:  # URLError, connection reset, timeout
            return None, {}, b"", f"{type(error).__name__}: {error}"


def body_error(response: Any) -> tuple[int, str] | None:
    """The error a 200 response reports instead of an answer: a top-level error object, or one
    choice that stopped with ``finish_reason`` "error" and an error object, as when the provider
    fails during generation. None when the response has no such error."""
    if not isinstance(response, dict):
        return None
    if "error" in response and "choices" not in response:
        return embedded_error(response["error"])
    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
        choice = choices[0]
        if choice.get("finish_reason") == "error" and isinstance(choice.get("error"), dict):
            return embedded_error(choice["error"])
    return None


def embedded_error(error: Any) -> tuple[int, str]:
    """The HTTP-like status and message of an error object in a 200 response; 502 when it names
    no valid status, as OpenRouter does for provider errors."""
    code = error.get("code") if isinstance(error, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    status = code if type(code) is int and 400 <= code < 600 else 502
    return status, f"error in the response body: {message or repr(error)}"


def lower_names(headers: Any) -> dict[str, str]:
    """Response headers with lower-case names: HTTP header names are case-insensitive."""
    return {name.lower(): value for name, value in (headers or {}).items()}


def retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds to wait from a Retry-After value, either delay-seconds or an HTTP date (RFC 9110);
    None when it is missing or invalid. A date in the past means no wait."""
    if value is None:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:  # "-0000": UTC
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


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


def model_matches(requested: str, reported: Any) -> bool:
    """Whether the response's model is the requested one or a dated snapshot of it, such as
    ``google/gemini-3.8-flash-20260902`` for ``google/gemini-3.8-flash``."""
    if not isinstance(reported, str) or reported == requested:
        return reported == requested
    suffix = reported.removeprefix(f"{requested}-")
    return suffix != reported and suffix.isdigit() and len(suffix) == 8


def single_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        count = len(choices) if isinstance(choices, list) else "no"
        raise AdapterError(f"expected one choice in the response, got {count} (response.json)")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content") or "", str):
        raise AdapterError("the choice has no text message (response.json)")
    return choice


def process(
    client: Client, model: str, params: dict, sample: dict, source: Path, out: Path, extra: dict
) -> dict:
    """OCR one sample, write its outputs and return ``result.json`` content. What is known about
    the request and response is added to ``extra``, also when the sample fails."""
    data, media_type, width, height = read_image(source / sample["image"])
    image_sha256 = hashlib.sha256(data).hexdigest()
    url = f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"
    body = request_body(model, params, url)
    extra["image"] = {
        "media_type": media_type,
        "bytes": len(data),
        "sha256": image_sha256,
        "width": width,
        "height": height,
    }
    # The request as sent, with the image data replaced by its fingerprint.
    logged = request_body(model, params, f"data:{media_type};base64,<sha256 {image_sha256}>")
    (out / "request.json").write_text(json.dumps(logged, ensure_ascii=False), "utf-8")
    try:
        response, transport = client.post(body)
    except TransportError as error:
        extra["transport"] = error.record
        raise
    extra["transport"] = transport
    (out / "response.json").write_text(json.dumps(response, ensure_ascii=False), "utf-8")
    if not isinstance(response, dict):
        raise AdapterError("the response is not a JSON object (response.json)")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    extra |= {
        "generation_id": response.get("id"),
        "created": response.get("created"),
        "response_model": response.get("model"),
        "provider": response.get("provider"),
        "service_tier": response.get("service_tier"),
        "system_fingerprint": response.get("system_fingerprint"),
        "usage": usage,
        "cost_usd": usage.get("cost"),
    }
    if not model_matches(model, response.get("model")):
        raise AdapterError(
            f"the response is from model {response.get('model')!r}, not the pinned {model!r}"
        )
    choice = single_choice(response)
    finish, native = choice.get("finish_reason"), choice.get("native_finish_reason")
    text = choice["message"].get("content") or ""
    extra |= {"finish_reason": finish, "native_finish_reason": native, "empty": not text.strip()}
    if finish not in ("stop", "length"):
        detail = choice.get("error") or choice["message"].get("refusal")
        raise AdapterError(
            f"the model stopped with {finish!r} (native {native!r})"
            + (f": {detail}" if detail else "")
        )
    (out / "page.md").write_text(text, encoding="utf-8")
    loop = tail_repeats(text)
    extra["tail_repetition"] = loop
    result: dict[str, Any] = {"status": "ok", "primary": "page.md"}
    if finish == "length":
        result |= {"status": "truncated", "message": "the answer reached max_tokens"}
    elif loop:
        result |= {"status": "truncated", "message": "repetition loop at the end of the answer"}
    return result


def write_result(directory: Path, result: dict[str, Any]) -> None:
    tmp = directory / ".result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main() -> None:
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    client, model, params, settings, init_error = None, job.get("model"), None, {}, None
    try:
        model = check_model(model)
        params = parse_params(job.get("params") or {})
        settings = settings_of(model, params)
        key = os.environ.get(KEY_VARIABLE)
        if not key:
            raise AdapterError(f"{KEY_VARIABLE} is not set")
        client = Client(params, key)
    except AdapterError as error:
        init_error = str(error)
    for sample in job["samples"]:
        out = target / sample["id"]
        out.mkdir()
        start = time.monotonic()
        extra: dict[str, Any] = {}
        try:
            if init_error:
                raise AdapterError(init_error)
            result = process(client, model, params, sample, source, out, extra)
        except AdapterError as error:
            result = {"status": "failed", "message": str(error)}
        except Exception as error:  # fail the sample, keep going
            result = {"status": "failed", "message": f"{type(error).__name__}: {error}"}
        result["seconds"] = round(time.monotonic() - start, 3)
        result["extra"] = extra | {"settings": settings}
        write_result(out, result)
    print(f"openrouter adapter processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
