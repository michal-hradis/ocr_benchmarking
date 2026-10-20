"""Fake OCR engine for runner tests; standard library only, so it also runs in the CPU base.

It follows the engine protocol in docs/RUNNER.md: read $OCRBENCH_INPUT/job.json, process the
samples in order and write $OCRBENCH_OUTPUT/<id>/ with native output and, last, result.json.
``params.behaviors`` maps sample IDs to a behavior (default ``params.default``):

markdown, text, prediction, truncated  valid output (truncated is claimed as such)
failed                                 the engine reports a failure
invalid                                prediction.json with the wrong sample ID
escape                                 primary output is a symlink to a host file
bad-result                             result.json is not JSON
skip                                   no output at all for the sample
crash                                  exit with status 3 before any output
hang                                   sleep until killed
flaky, hang-once                       crash or hang the first time; afterwards markdown
                                       (remembered in $FAKE_ENGINE_FLAKY_DIR/<id>)

Every finished sample also gets seen.json describing what the engine could observe.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def write_result(directory, result):
    tmp = directory / ".result.json.tmp"
    tmp.write_text(result if isinstance(result, str) else json.dumps(result), encoding="utf-8")
    os.replace(tmp, directory / "result.json")  # result.json appears complete, and last


def main():
    source = Path(os.environ["OCRBENCH_INPUT"])
    target = Path(os.environ["OCRBENCH_OUTPUT"])
    job = json.loads((source / "job.json").read_text(encoding="utf-8"))
    params = job["params"]
    for sample in job["samples"]:
        sid = sample["id"]
        behavior = params.get("behaviors", {}).get(sid, params.get("default", "markdown"))
        if behavior in ("flaky", "hang-once"):
            marker = Path(os.environ["FAKE_ENGINE_FLAKY_DIR"]) / sid
            behavior = "markdown" if marker.exists() else behavior.replace("flaky", "crash")
            behavior = behavior.replace("hang-once", "hang")
            marker.touch()
        if behavior == "crash":
            sys.exit(3)
        if behavior == "hang":
            time.sleep(3600)
        if behavior == "skip":
            continue
        out = target / sid
        out.mkdir()
        image = (source / sample["image"]).read_bytes()
        seen = {
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "sample": sample,
            "model": job["model"],
            "input": sorted(str(p.relative_to(source)) for p in source.rglob("*")),
            "env": sorted(os.environ),
            "uid": os.getuid(),
            "nvidia_devices": sorted(p.name for p in Path("/dev").glob("nvidia[0-9]*")),
        }
        (out / "seen.json").write_text(json.dumps(seen), encoding="utf-8")
        result = {"status": "ok", "seconds": 0.01}
        if behavior in ("markdown", "truncated"):
            (out / "page.md").write_text(f"# {sid}\n\nText of *{sid}*.\n", encoding="utf-8")
            result["primary"] = "page.md"
            if behavior == "truncated":
                result.update(status="truncated", message="output token limit reached")
        elif behavior == "text":
            (out / "page.txt").write_text(f"plain {sid}\n", encoding="utf-8")
            result["primary"] = "page.txt"
        elif behavior in ("prediction", "invalid"):
            prediction = {
                "format": "ocrbench.prediction",
                "version": 1,
                "sample_id": sid if behavior == "prediction" else "other",
                "text": f"predicted {sid}",
            }
            (out / "prediction.json").write_text(json.dumps(prediction), encoding="utf-8")
            result["primary"] = "prediction.json"
        elif behavior == "failed":
            result = {"status": "failed", "message": "unsupported page"}
        elif behavior == "escape":
            os.symlink("/etc/hostname", out / "page.txt")
            result["primary"] = "page.txt"
        elif behavior == "bad-result":
            result = "{not json"
        write_result(out, result)
    print(f"fake engine processed {len(job['samples'])} samples", file=sys.stderr)


if __name__ == "__main__":
    main()
