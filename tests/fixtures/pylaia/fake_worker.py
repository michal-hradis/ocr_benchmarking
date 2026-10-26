"""Fake PyLaia recognizer process for the offline tests in tests/test_pylaia.py.

It speaks the protocol of engines/pylaia/pylaia_worker.py, whose request reader and text joining
it uses, without PyTorch or PyLaia: every line image is "recognized" as ``Řádek <w>x<h>``. The
environment selects failures: ``FAKE_PYLAIA_LOAD_ERROR`` fails loading, ``FAKE_PYLAIA_ERROR_ON``
and ``FAKE_PYLAIA_CRASH_ON`` make the n-th request of this process fail with an error reply, or
exit without a reply.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "engines" / "pylaia"))
from pylaia_worker import read_request, symbols_to_text  # noqa: E402


def symbols(text):
    return ["<space>" if char == " " else char for char in text]


def main():
    out = sys.stdout
    args = json.loads(sys.argv[1])
    if message := os.environ.get("FAKE_PYLAIA_LOAD_ERROR"):
        out.write(json.dumps({"error": message}) + "\n")
        return
    info = {"pylaia": "1.1.2+fake", "pylaia_torch": "1.13.1-fake", "symbols": 160}
    out.write(json.dumps({"ready": True, **info, "args": args}) + "\n")
    out.flush()
    requests = 0
    while (images := read_request(sys.stdin.buffer)) is not None:
        requests += 1
        if str(requests) == os.environ.get("FAKE_PYLAIA_CRASH_ON"):
            sys.exit(3)
        if str(requests) == os.environ.get("FAKE_PYLAIA_ERROR_ON"):
            reply = {"error": "RuntimeError: CUDA out of memory (fake)"}
        else:
            lines = []
            for image in images:
                syms = symbols(f" Řádek {image.width}x{image.height} ")
                confidences = [0.5] * len(syms)
                lines.append(
                    {"symbols": syms, "text": symbols_to_text(syms), "confidences": confidences}
                )
            reply = {"lines": lines}
        out.write(json.dumps(reply, ensure_ascii=False) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
