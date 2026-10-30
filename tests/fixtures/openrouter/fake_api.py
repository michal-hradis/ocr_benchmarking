"""A fake OpenRouter chat completions API on the loopback interface for the offline adapter tests.

It prints its port, then serves POST /api/v1/chat/completions. The answer depends on the width of
the posted image: 900 (and any other) returns response.json; 910 the same answer in Markdown;
302 redirects to the URL given as the optional second argument; 300 an empty answer; 401 an
authentication error; 402 an out-of-credits error; 429 throttles the first request with
Retry-After; 430 answers the first request with HTTP 200 and an error object in the body; 431
answers the first request with HTTP 200 and a choice that failed during generation; 503
always fails; 555 returns an answer that ends in a repetition loop; 600 stops at the token limit;
601 is blocked by a content filter; 776 answers from a dated snapshot of the model; 777 from
another model; 778 returns two choices; 779 returns a body that is not JSON. Every request is
logged, without the image data, to the JSON-lines file given as the first argument; a GET, which
only a followed redirect sends, is logged with its Authorization header.
"""

import base64
import io
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

RESPONSE = json.loads((Path(__file__).parent / "response.json").read_text(encoding="utf-8"))
LOG = Path(sys.argv[1])
REDIRECT = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:9/"
MARKDOWN = (
    "# The **Daily** Chronicle\n\nThe quick brown fox jumps over\nthe lazy dog &amp; the river."
)
SEEN: dict[int, int] = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body, headers=()):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        for key, value in (("X-Generation-Id", "gen-fake"), *headers):
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # Only a followed redirect would arrive here; log it with its Authorization header.
        with LOG.open("a") as log:
            entry = {"method": "GET", "path": self.path}
            log.write(
                json.dumps(entry | {"authorization": self.headers.get("Authorization")}) + "\n"
            )
        self.reply(404, {"error": {"code": 404, "message": "Not found"}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        parts = body["messages"][0]["content"]
        header, encoded = parts[1]["image_url"]["url"].split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        width = image.width
        SEEN[width] = SEEN.get(width, 0) + 1
        logged = json.loads(json.dumps(body))
        logged["messages"][0]["content"][1]["image_url"]["url"] = header
        logged |= {"path": self.path, "authorization": self.headers.get("Authorization")}
        logged |= {"width": width, "bytes": len(base64.b64decode(encoded))}
        with LOG.open("a") as log:
            log.write(json.dumps(logged) + "\n")
        if self.headers.get("Authorization") != "Bearer test-key" or width == 401:
            return self.reply(401, {"error": {"code": 401, "message": "No auth credentials"}})
        if width == 302:
            return self.reply(302, b"", [("Location", REDIRECT)])
        if width == 402:
            return self.reply(402, {"error": {"code": 402, "message": "Insufficient credits"}})
        if width == 429 and SEEN[width] == 1:
            limited = {"error": {"code": 429, "message": "Rate limited"}}
            return self.reply(429, limited, [("Retry-After", "0")])
        if width == 430 and SEEN[width] == 1:
            return self.reply(200, {"error": {"message": "Provider returned error", "code": "x"}})
        if width == 431 and SEEN[width] == 1:
            failed = json.loads(json.dumps(RESPONSE))
            failed["choices"][0] |= {
                "finish_reason": "error",
                "native_finish_reason": "OTHER",
                "error": {"code": 500, "message": "Internal Server Error"},
            }
            failed["choices"][0]["message"]["content"] = "The image is"
            return self.reply(200, failed)
        if width == 503:
            return self.reply(503, {"error": {"code": 503, "message": "No available provider"}})
        if width == 779:
            return self.reply(200, b"<html>gateway</html>")
        response = json.loads(json.dumps(RESPONSE))
        choice = response["choices"][0]
        if width == 300:
            choice["message"]["content"] = ""
        if width == 910:
            choice["message"]["content"] = MARKDOWN
        if width == 555:
            choice["message"]["content"] = "The river rose. " + "v Brně, " * 20
        if width == 600:
            choice |= {"finish_reason": "length", "native_finish_reason": "MAX_TOKENS"}
        if width == 601:
            choice |= {"finish_reason": "content_filter", "native_finish_reason": "RECITATION"}
            choice["message"]["content"] = None
        if width == 776:
            response["model"] = "google/gemini-3.8-flash-20260902"
        if width == 777:
            response["model"] = "google/gemini-2.5-flash"
        if width == 778:
            response["choices"].append(dict(choice, index=1))
        self.reply(200, response)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_address[1], flush=True)
server.serve_forever()
