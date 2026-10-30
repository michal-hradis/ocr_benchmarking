"""A fake Mistral OCR API on the loopback interface for the offline adapter tests.

It prints its port, then serves POST /v1/ocr. The answer depends on the width of the posted image:
900 (and any other) returns response.json for that image; 300 an empty page; 401 an
authentication error; 429 throttles the first request with Retry-After; 430 answers the first
request with HTTP 520 and a lower-case retry-after date in the past; 503 always fails; 555
returns a text block that ends in a repetition loop; 777 answers from another model; 778 returns
two pages; 779 returns a body that is not JSON. Every request is logged, without the image data,
to the JSON-lines file given as the first argument.
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
SEEN: dict[int, int] = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body, headers=()):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        for key, value in (("x-ratelimit-remaining-req-minute", "59"), *headers):
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        url = body["document"]["image_url"]
        header, encoded = url.split(",", 1)
        image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        width = image.width
        SEEN[width] = SEEN.get(width, 0) + 1
        logged = dict(body, document={"type": body["document"]["type"], "header": header})
        logged |= {"path": self.path, "authorization": self.headers.get("Authorization")}
        logged |= {"width": width, "bytes": len(base64.b64decode(encoded))}
        with LOG.open("a") as log:
            log.write(json.dumps(logged) + "\n")
        if self.headers.get("Authorization") != "Bearer test-key" or width == 401:
            return self.reply(401, {"message": "Unauthorized", "type": "invalid_api_key"})
        if width == 429 and SEEN[width] == 1:
            return self.reply(429, {"message": "Rate limit exceeded"}, [("Retry-After", "0")])
        if width == 430 and SEEN[width] == 1:
            date = [("retry-after", "Wed, 21 Oct 2015 07:28:00 GMT")]
            return self.reply(520, {"message": "Unknown error"}, date)
        if width == 503:
            return self.reply(503, {"message": "Service unavailable"})
        if width == 779:
            return self.reply(200, b"<html>gateway</html>")
        response = json.loads(json.dumps(RESPONSE))
        page = response["pages"][0]
        if width == 300:
            page.update(markdown="", blocks=[], images=[])
        if width == 555:
            loop = "The river rose. " + "v Brně, " * 20
            page["blocks"][2]["content"] = loop
            page["markdown"] = page["markdown"].replace("the lazy dog &amp; the river.", loop)
        if width == 777:
            response["model"] = "mistral-ocr-latest"
        if width == 778:
            response["pages"].append(dict(page, index=1))
        if not body.get("include_blocks"):
            page.pop("blocks")
        self.reply(200, response, [("mistral-correlation-id", f"req-{width}")])


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_address[1], flush=True)
server.serve_forever()
