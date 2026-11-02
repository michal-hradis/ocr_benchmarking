#!/usr/bin/env python3
"""Fake upstream serve.py (vLLM with the MonkeyOCRv2 plugin): an OpenAI-compatible chat
completions server on the loopback interface with fixed outputs (../../README.md). It checks the
command line and every request as the adapter must send them."""

import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

FIXTURES = Path(__file__).resolve().parents[3]
TEXT = "Please output the text content from the image."
FORMULA = "Please write out the expression of the formula in the image using LaTeX format."
TABLE = "Please extract the table from the image and represent it in OTSL format."
LAYOUT = "Please output the categories and coordinates of the document elements in reading order."
MAX_TOKENS = {LAYOUT: 4096, TEXT: 5000, FORMULA: 5000, TABLE: 5000}
# Recognition outputs of the blocks of ../../layout.txt, by block number (the crop's marker).
BLOCKS = {
    1: "THE CHRONICLE",
    2: "The **Daily** Chronicle",
    3: "The quick brown fox jumps over\nthe lazy dog & the river.",
    4: "News",
    5: "<fcel>Year<fcel>Copies<nl><fcel>1923<fcel>2O26<nl>",
    6: "$E = mc^2$",
    9: "Footer",
    10: "A block with its corners swapped.",
}
# Three blocks; the recognition request of the second one fails.
REC500 = str([{"bbox": [100, y, 900, y + 200], "label": "Text"} for y in (100, 400, 700)])
STATE = {"kind": None, "draft": False}
COUNTERS = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0}
LOCK = threading.Lock()


def news(limit):
    items = [f"{{'bbox': [50, {n * 9}, 950, {n * 9 + 8}], 'label': 'Text'}}" for n in range(1, 101)]
    text = "[" + ", ".join(items) + "]"
    return (text[:limit], "length") if len(text) > limit else (text, "stop")


def layout(gray, limit):
    """The layout output for a page, chosen by the gray level of its top-left corner."""
    kinds = [
        (190, "page", lambda: ((FIXTURES / "layout.txt").read_text(encoding="utf-8"), "stop")),
        (165, "invalid", lambda: ("[{'bbox': [1, 2], 'label': 'Text'}]", "stop")),
        (155, "overflow", lambda: ("[{'bbox': [0, 0, 1e999, 10], 'label': 'Text'}]", "stop")),
        (145, "news", lambda: news(limit)),
        (130, "rec500", lambda: (REC500, "stop")),
        (110, "repeat", lambda: ("[{'bbox': [0, 0, 1000, 1000], 'label': 'Text'}]", "stop")),
        (90, "garbage", lambda: ("There is no layout here.", "stop")),
        (84, "blank", lambda: ("  \n", "stop")),
        (75, "abort", lambda: ("[{'bbox': [0, 0, 1000, 1000], 'label': 'Text'}", "abort")),
        (0, "empty", lambda: ("[]", "stop")),
    ]
    for threshold, kind, make in kinds:
        if gray > threshold:
            STATE["kind"] = kind
            return make()


def recognize(marker, limit):
    kind = STATE["kind"]
    if kind == "repeat":
        text = "Intro " + "word " * limit
        return text[:limit], "length"
    if kind == "news" and 100 < marker <= 200:
        return f"Line {marker - 100} of a very long newspaper page.", "stop"
    return BLOCKS.get(marker - 100, ""), "stop"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body, content_type="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/metrics":
            lines = ["# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts."]
            if STATE["draft"]:
                with LOCK:
                    lines += [
                        f'vllm:spec_decode_num_{name}_total{{engine="0"}} {value}.0'
                        for name, value in COUNTERS.items()
                    ]
                lines.append('vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 7.0')
            return self.reply(200, "\n".join(lines) + "\n", "text/plain")
        self.reply(200 if self.path == "/health" else 404, {})

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        (message,) = request["messages"]
        image_part, text_part = message["content"]
        prompt = text_part["text"]
        assert message["role"] == "user"
        assert text_part["type"] == "text"
        assert request["model"] == "MonkeyOCRv2"
        assert request["temperature"] == 0
        assert request["max_tokens"] == MAX_TOKENS[prompt]
        url = image_part["image_url"]["url"]
        assert image_part["type"] == "image_url"
        assert url.startswith("data:image/png;base64,")
        image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("L")
        if prompt == LAYOUT:
            assert image.width * image.height >= 1003520 * 0.99  # enlarged to at least 1 MP
            gray = image.getpixel((image.width // 40, image.height // 40))
            if 35 < gray < 65:  # 50: a request error
                return self.reply(500, {"error": {"message": "fake CUDA out of memory"}})
            if gray < 35:  # 20: the server dies
                os._exit(1)
            text, reason = layout(gray, request["max_tokens"])
        else:
            assert image.width * image.height <= 1003520
            marker = image.getpixel((2, 2))
            if STATE["kind"] == "rec500" and marker == 102:
                return self.reply(500, {"error": {"message": "fake recognition error"}})
            text, reason = recognize(marker, request["max_tokens"])
        if STATE["draft"]:
            with LOCK:
                COUNTERS["drafts"] += len(text) // 16 + 1
                COUNTERS["draft_tokens"] += 16 * (len(text) // 16 + 1)
                COUNTERS["accepted_tokens"] += len(text) // 2
        self.reply(
            200,
            {
                "choices": [{"message": {"content": text}, "finish_reason": reason}],
                "usage": {
                    "prompt_tokens": image.width * image.height // 784 + len(prompt),
                    "completion_tokens": len(text),
                },
            },
        )


def option(args, name):
    return args[args.index(name) + 1] if name in args else None


def main():
    args = sys.argv[1:]
    assert option(args, "--model-path").endswith("e8871a0057c98378219b369f924a25ba10f392bf")
    assert option(args, "--host") == "127.0.0.1"
    assert option(args, "--served-model-name") == "MonkeyOCRv2"
    assert option(args, "--max-model-len") == "16384"
    assert option(args, "--max-num-seqs") == "128"
    draft = option(args, "--draft-model")
    if draft is not None:
        assert draft.endswith("5e87f46698d698e971039168c01c34e044445a9b")
        assert option(args, "--dflash-num-speculative-tokens") == "16"
        STATE["draft"] = True
    if os.environ.get("FAKE_VLLM_EXIT"):
        print("fake vLLM start-up failure", flush=True)
        sys.exit(3)
    port = int(option(args, "--port"))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
