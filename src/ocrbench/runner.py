"""Common runner: execute one OCR engine over a dataset and record every sample.

One container processes all pending samples of a run in job order. The runner watches the output
for finished samples; if none finishes within the timeout, or the container exits early, the
sample in progress is recorded as a failure and a new container continues with the rest. Every
attempt gets its own execution record and is never overwritten. The protocol and the run store
are documented in docs/RUNNER.md; the decisions are in ADR 0007.

Ground truth stays on the host: the container sees only a job file and copies of the sample
images (ADR 0002). The GT hashes are frozen in run.json so that scoring can detect changed GT.
Engine output is untrusted: it is validated and never followed outside its directory.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import io
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Protocol

from PIL import Image

import ocrbench
from ocrbench.convert import SUFFIXES, convert_native
from ocrbench.engine import EngineConfig, parse_engine_config
from ocrbench.issues import Issue, Severity
from ocrbench.manifest import SAMPLE_ID_RE, Sample, load_manifest, sha256_file
from ocrbench.prediction import (
    Prediction,
    load_prediction,
    prediction_to_json,
    validate_prediction,
)

RUN_FORMAT = "ocrbench.run"
RUN_VERSION = 1
RECORD_FORMAT = "ocrbench.execution-record"
RECORD_VERSION = 1
JOB_FORMAT = "ocrbench.job"
JOB_VERSION = 1
RESULT_FILE = "result.json"
STATUSES = ("success", "truncated", "partial", "failure")
REASONS = ("timeout", "crash", "no-output", "invalid-output", "engine-error", "input-error")
# Status claimed by the engine in result.json -> final status when the output is valid.
ENGINE_STATUSES = {"ok": "success", "truncated": "truncated", "partial": "partial"}
ENGINE_STATUSES_ALL = (*ENGINE_STATUSES, "failed")
MAX_RESULT_BYTES = 1 << 20
MAX_PRIMARY_BYTES = 64 << 20
# docker run's own exit codes: the daemon or the container command failed before running.
_START_FAILURES = {125, 126, 127}
_RESULT_KEYS = {"status", "primary", "message", "seconds", "extra"}


class RunnerError(RuntimeError):
    """The run cannot proceed (bad input, locked or changed run, engine does not start)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_json(path: Path, data: Any, *, exclusive: bool = False) -> None:
    """Write JSON atomically; ``exclusive`` refuses to replace an existing file."""
    text = json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    if exclusive:
        with open(path, "x", encoding="utf-8") as stream:
            stream.write(text)
        return
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _finite(value: Any) -> bool:
    """No infinite numbers anywhere, such as the overflow 1e999 that json.loads accepts."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


# Run store -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSample:
    id: str
    image: Path
    image_sha256: str | None  # None when the image could not be read at run creation
    size: tuple[int, int] | None
    job: dict[str, Any]  # what the engine receives: id, unit and enabled metadata only
    gt: Path  # host-side only, like gt_sha256: never in the job or the container input
    gt_sha256: str | None  # None when the GT could not be read at run creation
    error: str | None = None  # input problem found at run creation
    gt_error: str | None = None  # why gt_sha256 is None
    doc_type: str | None = None  # for evaluation strata; in the job only when enabled


@dataclass(frozen=True)
class Run:
    path: Path
    run_id: str
    engine: EngineConfig
    image_id: str | None
    runtime: str
    cache: Path | None
    samples: tuple[RunSample, ...]


def _job_entry(sample: Sample, engine: EngineConfig) -> dict[str, Any]:
    """Allowed inference metadata (ADR 0002): never the GT path, split or checksums."""
    entry: dict[str, Any] = {"id": sample.id, "unit": sample.unit}
    if "language" in engine.metadata and sample.language:
        entry["language"] = list(sample.language)
    if "doc_type" in engine.metadata and sample.doc_type is not None:
        entry["doc_type"] = sample.doc_type
    if "year" in engine.metadata and sample.year is not None:
        entry["year"] = sample.year
    return entry


def _inspect_image(sample: Sample) -> tuple[str | None, tuple[int, int] | None, str | None]:
    try:
        digest = sha256_file(sample.image)
        with Image.open(sample.image) as image:  # reads the header only
            size = image.size
    except (OSError, Image.DecompressionBombError) as error:
        return None, None, f"cannot read image: {error}"
    if sample.image_sha256 is not None and digest != sample.image_sha256:
        return None, None, "image does not match image_sha256 in the manifest"
    return digest, size, None


def _inspect_gt(sample: Sample) -> tuple[str | None, str | None]:
    try:
        digest = sha256_file(sample.gt)
    except OSError as error:
        return None, f"cannot read GT: {error}"
    if sample.gt_sha256 is not None and digest != sample.gt_sha256:
        return None, "GT does not match gt_sha256 in the manifest"
    return digest, None


def _command_output(args: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def host_info(engine: EngineConfig, runtime: str) -> dict[str, Any]:
    memory = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "memory_bytes": memory,
        "docker": None,
        "gpus": None,
    }
    if runtime == "docker":
        info["docker"] = _command_output(["docker", "version", "--format", "{{.Server.Version}}"])
        if engine.gpus:
            query = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
            gpus = _command_output(query)
            info["gpus"] = gpus.splitlines() if gpus else None
    return info


def create_run(
    engine: EngineConfig,
    manifest_path: Path,
    runs_dir: Path,
    *,
    run_id: str | None = None,
    image_id: str | None = None,
    runtime: str = "docker",
    cache: Path | None = None,
) -> Run:
    """Create a run directory with its frozen configuration; nothing is executed yet."""
    manifest, issues = load_manifest(manifest_path)
    if manifest is None:
        raise RunnerError("invalid manifest:\n" + "\n".join(i.format() for i in issues))
    if run_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{engine.name}-{manifest.name}-{stamp}"
    if not SAMPLE_ID_RE.match(run_id):
        raise RunnerError(f"run ID {run_id!r} must match {SAMPLE_ID_RE.pattern}")
    path = Path(runs_dir).resolve() / run_id
    try:
        path.mkdir(parents=True)
    except FileExistsError as error:
        raise RunnerError(f"{path} already exists; continue it with 'run resume'") from error
    samples = []
    for sample in manifest.samples:
        digest, size, error = _inspect_image(sample)
        gt_digest, gt_error = _inspect_gt(sample)
        samples.append(
            {
                "id": sample.id,
                "image": str(sample.image),
                "image_sha256": digest,
                "size": list(size) if size else None,
                "job": _job_entry(sample, engine),
                "error": error,
                "gt": str(sample.gt),
                "gt_sha256": gt_digest,
                "gt_error": gt_error,
                "doc_type": sample.doc_type,
            }
        )
    data = {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "run_id": run_id,
        "created": _now(),
        "ocrbench_version": ocrbench.__version__,
        "runtime": runtime,
        "engine": engine.to_json(),
        "engine_config_sha256": engine.sha256(),
        "image_id": image_id,
        "cache": str(cache) if cache else None,
        "dataset": {
            "name": manifest.name,
            "manifest": str(Path(manifest_path).resolve()),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "host": host_info(engine, runtime),
        "samples": samples,
    }
    _write_json(path / "run.json", data, exclusive=True)
    return load_run(path)


def load_run(path: Path) -> Run:
    path = Path(path).resolve()
    try:
        data = _read_json(path / "run.json")
    except (OSError, ValueError) as error:
        raise RunnerError(f"{path} is not a readable run: {error}") from error
    if data.get("format") != RUN_FORMAT or data.get("version") != RUN_VERSION:
        raise RunnerError(f"{path / 'run.json'} is not an {RUN_FORMAT} v{RUN_VERSION} file")
    engine = parse_engine_config(data["engine"], "run.json engine")
    if engine.sha256() != data["engine_config_sha256"]:
        raise RunnerError(f"{path / 'run.json'}: the frozen engine configuration was modified")
    samples = tuple(
        RunSample(
            id=s["id"],
            image=Path(s["image"]),
            image_sha256=s["image_sha256"],
            size=tuple(s["size"]) if s["size"] else None,
            job=s["job"],
            gt=Path(s["gt"]),
            gt_sha256=s["gt_sha256"],
            error=s["error"],
            gt_error=s["gt_error"],
            doc_type=s.get("doc_type"),  # absent in runs created before #8
        )
        for s in data["samples"]
    )
    cache = Path(data["cache"]) if data["cache"] else None
    return Run(path, data["run_id"], engine, data["image_id"], data["runtime"], cache, samples)


def _attempts(run: Run, sample_id: str) -> list[Path]:
    root = run.path / "samples" / sample_id
    if not root.is_dir():
        return []
    found = [p for p in root.iterdir() if p.name.removeprefix("attempt-").isdigit()]
    return sorted(found, key=lambda p: int(p.name.removeprefix("attempt-")))


def latest_record(run: Run, sample_id: str) -> dict[str, Any] | None:
    """The record of the newest completed attempt, or None while the sample is pending."""
    for attempt in reversed(_attempts(run, sample_id)):
        if (attempt / "record.json").is_file():
            return _read_json(attempt / "record.json")
    return None


def summarize(run: Run) -> dict[str, Any]:
    counts = dict.fromkeys((*STATUSES, "pending"), 0)
    failures = []
    for sample in run.samples:
        record = latest_record(run, sample.id)
        counts[record["status"] if record else "pending"] += 1
        if record and record["status"] == "failure":
            failures.append((sample.id, record["reason"], record.get("message")))
    return {"counts": counts, "failures": failures}


def verify_ground_truth(run: Run) -> list[Issue]:
    """Errors for GT that is not pinned or differs from the GT frozen at run creation.

    Scoring must call this first: stored predictions are only comparable against the same GT.
    """
    issues = []
    for sample in run.samples:
        if sample.gt_sha256 is None:
            message = f"GT was not pinned at run creation: {sample.gt_error}"
            issues.append(Issue(Severity.ERROR, "gt-unpinned", message, sample.id))
            continue
        try:
            changed = sha256_file(sample.gt) != sample.gt_sha256
        except OSError as error:
            issues.append(Issue(Severity.ERROR, "gt-unreadable", str(error), sample.id))
            continue
        if changed:
            message = f"{sample.gt} changed since the run was created"
            issues.append(Issue(Severity.ERROR, "gt-changed", message, sample.id))
    return issues


def verify_image(run: Run, image_id: str) -> None:
    if run.image_id is not None and image_id != run.image_id:
        raise RunnerError(
            f"{run.engine.image} now resolves to {image_id}, but the run used {run.image_id}; "
            "start a new run instead"
        )


# Launchers -----------------------------------------------------------------------------------


class Process(Protocol):
    def poll(self) -> int | None: ...
    def wait(self) -> int: ...
    def kill(self) -> None: ...


class Launcher(Protocol):
    def start(self, name: str, input_dir: Path, output_dir: Path, log: IO[bytes]) -> Process: ...
    def describe(self, name: str, input_dir: Path, output_dir: Path) -> list[str]: ...


def _mount(source: Path, target: str, *, readonly: bool) -> str:
    """A --mount value; CSV quoting keeps commas and quotes in host paths intact."""
    fields = ["type=bind", f"source={source}", f"target={target}"]
    if readonly:
        fields.append("readonly")
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(fields)
    return buffer.getvalue()


def resolve_image(reference: str) -> str:
    """The local image ID of a tag; the runner never builds or pulls images."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", reference],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RunnerError(f"cannot run docker: {error}") from error
    if result.returncode != 0:
        raise RunnerError(
            f"image {reference!r} is not available locally; build it first "
            f"({result.stderr.strip()})"
        )
    return result.stdout.strip()


class _DockerProcess:
    def __init__(self, popen: subprocess.Popen, name: str) -> None:
        self.popen = popen
        self.name = name

    def poll(self) -> int | None:
        return self.popen.poll()

    def wait(self) -> int:
        return self.popen.wait()

    def kill(self) -> None:
        subprocess.run(["docker", "kill", self.name], capture_output=True, timeout=60, check=False)
        try:
            self.popen.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.popen.kill()


class DockerLauncher:
    """Runs the engine image isolated: no network, read-only root and input, host UID/GID.

    An engine configured with ``network = "egress"`` (a hosted API, ADR 0013) runs only when
    ``allow_network`` is set; it gets outbound access on Docker's default bridge network and its
    secrets by name from the host environment, so their values never appear in the command."""

    def __init__(
        self,
        engine: EngineConfig,
        image_id: str,
        cache: Path | None = None,
        allow_network: bool = False,
    ) -> None:
        if engine.network != "none" and not allow_network:
            raise RunnerError(
                f"engine {engine.name!r} needs outbound network access (network = "
                f"{engine.network!r}) to a hosted API; pass --allow-network to permit it"
            )
        missing = [key for key in engine.secrets if not os.environ.get(key)]
        if missing:
            raise RunnerError(f"secrets not set in the environment: {', '.join(missing)}")
        self.engine = engine
        self.image_id = image_id
        self.cache = cache

    def describe(self, name: str, input_dir: Path, output_dir: Path) -> list[str]:
        engine = self.engine
        network = "bridge" if engine.network == "egress" else "none"
        args = ["docker", "run", "--rm", "--name", name, "--network", network, "--read-only"]
        args += ["--tmpfs", "/tmp", "--user", f"{os.getuid()}:{os.getgid()}"]
        env = {"HOME": "/tmp", "OCRBENCH_INPUT": "/input", "OCRBENCH_OUTPUT": "/output"}
        for key, value in {**engine.env, **env}.items():
            args += ["--env", f"{key}={value}"]
        for key in engine.secrets:
            args += ["--env", key]  # Docker copies the value from its own environment
        for flag, value in (
            ("--cpus", engine.cpus),
            ("--memory", engine.memory),
            ("--shm-size", engine.shm_size),
            ("--gpus", engine.gpus),
        ):
            if value is not None:
                args += [flag, str(value)]
        args += ["--mount", _mount(input_dir, "/input", readonly=True)]
        args += ["--mount", _mount(output_dir, "/output", readonly=False)]
        if self.cache is not None:
            args += ["--mount", _mount(self.cache, "/cache", readonly=False)]
        return [*args, self.image_id, *engine.command]

    def start(self, name: str, input_dir: Path, output_dir: Path, log: IO[bytes]) -> Process:
        args = self.describe(name, input_dir, output_dir)
        popen = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
        return _DockerProcess(popen, name)


class _HostProcess:
    def __init__(self, popen: subprocess.Popen) -> None:
        self.popen = popen

    def poll(self) -> int | None:
        return self.popen.poll()

    def wait(self) -> int:
        return self.popen.wait()

    def kill(self) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.popen.pid, signal.SIGKILL)
        self.popen.wait()


class ProcessLauncher:
    """Runs an engine command as a host process, for offline tests. It provides NO isolation."""

    def __init__(self, command: Sequence[str], env: dict[str, str] | None = None) -> None:
        self.command = list(command)
        self.env = env or {}

    def describe(self, name: str, input_dir: Path, output_dir: Path) -> list[str]:
        return self.command

    def start(self, name: str, input_dir: Path, output_dir: Path, log: IO[bytes]) -> Process:
        env = {**os.environ, **self.env}
        env |= {"OCRBENCH_INPUT": str(input_dir), "OCRBENCH_OUTPUT": str(output_dir)}
        popen = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
        )
        return _HostProcess(popen)


# Output evaluation ---------------------------------------------------------------------------


@dataclass
class _Outcome:
    status: str
    reason: str | None = None
    message: str | None = None
    engine_status: str | None = None
    primary: str | None = None
    prediction: Prediction | None = None
    findings: tuple[Issue, ...] = ()
    seconds: float | None = None
    extra: Any = None


def _invalid(message: str, **fields: Any) -> _Outcome:
    return _Outcome("failure", "invalid-output", message, **fields)


def _read_result(directory: Path) -> dict[str, Any] | _Outcome:
    path = directory / RESULT_FILE
    if path.is_symlink() or not path.is_file():
        return _invalid(f"{RESULT_FILE} is not a regular file")
    if path.stat().st_size > MAX_RESULT_BYTES:
        return _invalid(f"{RESULT_FILE} is larger than {MAX_RESULT_BYTES} bytes")
    try:
        result = json.loads(path.read_bytes().decode("utf-8"), parse_constant=_reject_constant)
    except (OSError, ValueError) as error:
        return _invalid(f"{RESULT_FILE} is not valid JSON: {error}")
    if not isinstance(result, dict):
        return _invalid(f"{RESULT_FILE} must be a JSON object")
    problems = [f"unknown field {key!r}" for key in sorted(set(result) - _RESULT_KEYS)]
    if result.get("status") not in ENGINE_STATUSES_ALL:
        problems.append(f"'status' must be one of {ENGINE_STATUSES_ALL}")
    if result.get("status") != "failed" and not isinstance(result.get("primary"), str):
        problems.append("'primary' must name the primary output file")
    if not isinstance(result.get("message", ""), str):
        problems.append("'message' must be a string")
    seconds = result.get("seconds")
    if seconds is not None and (
        isinstance(seconds, bool)
        or not isinstance(seconds, int | float)
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        problems.append("'seconds' must be a finite nonnegative number")
    if not _finite(result.get("extra")):
        problems.append("'extra' must not contain infinite numbers")
    if problems:
        return _invalid(f"{RESULT_FILE}: " + "; ".join(problems))
    return result


def _evaluate(directory: Path, sample: RunSample) -> _Outcome:
    """Check one finished sample's output and build its standardized prediction."""
    if directory.is_symlink() or not directory.is_dir():
        return _invalid("the sample output is not a directory")
    result = _read_result(directory)
    if isinstance(result, _Outcome):
        return result
    fields = {
        "engine_status": result["status"],
        "seconds": result.get("seconds"),
        "extra": result.get("extra"),
    }
    message = result.get("message")
    if result["status"] == "failed":
        return _Outcome(
            "failure", "engine-error", message or "the engine reported a failure", **fields
        )
    primary = result["primary"]
    path = directory / primary
    root = directory.resolve()
    if (
        Path(primary).is_absolute()
        or not path.resolve().is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        return _invalid(f"primary output {primary!r} is not a regular file in the output", **fields)
    if path.stat().st_size > MAX_PRIMARY_BYTES:
        return _invalid(f"primary output is larger than {MAX_PRIMARY_BYTES} bytes", **fields)
    fields["primary"] = primary
    suffix = path.suffix.lower()
    if suffix == ".json":
        prediction, issues = load_prediction(path, sample_id=sample.id, image_size=sample.size)
    elif suffix in SUFFIXES:
        try:
            content = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            return _invalid(f"primary output is not UTF-8: {error}", **fields)
        data = convert_native(content, SUFFIXES[suffix], sample.id)
        prediction, issues = validate_prediction(
            data, sample_id=sample.id, image_size=sample.size, location=primary
        )
    else:
        known = ", ".join([".json", *SUFFIXES])
        return _invalid(f"primary output {primary!r} must end in one of {known}", **fields)
    if prediction is None:
        errors = [i for i in issues if i.severity is Severity.ERROR]
        return _invalid(
            f"invalid prediction: {errors[0].message}", findings=tuple(issues), **fields
        )
    status = ENGINE_STATUSES[result["status"]]
    return _Outcome(status, None, message, prediction=prediction, findings=tuple(issues), **fields)


def _listing(directory: Path) -> list[dict[str, Any]]:
    """Files written by the engine; symbolic links are listed, never followed."""
    if directory.is_symlink() or not directory.is_dir():
        return []
    files = []
    for root, dirs, names in os.walk(directory):
        for name in sorted(names + [d for d in dirs if Path(root, d).is_symlink()]):
            path = Path(root, name)
            entry: dict[str, Any] = {"path": str(path.relative_to(directory))}
            if path.is_symlink():
                entry["symlink"] = True
            else:
                entry["bytes"] = path.stat().st_size
            files.append(entry)
        dirs[:] = sorted(d for d in dirs if not Path(root, d).is_symlink())
    return files


# Execution -----------------------------------------------------------------------------------


class _Executor:
    def __init__(
        self,
        run: Run,
        launcher: Launcher,
        poll_interval: float,
        progress: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self.run = run
        self.launcher = launcher
        self.poll_interval = poll_interval
        self.progress = progress

    def _new_dir(self, parent: Path, prefix: str) -> tuple[int, Path]:
        """Create the next numbered directory; existing ones are never reused."""
        parent.mkdir(parents=True, exist_ok=True)
        names = (p.name.removeprefix(prefix) for p in parent.glob(f"{prefix}*"))
        numbers = [int(name) for name in names if name.isdigit()]
        number = max(numbers, default=0) + 1
        while True:
            path = parent / f"{prefix}{number}"
            try:
                path.mkdir()
                return number, path
            except FileExistsError:
                number += 1

    def record(
        self,
        sample: RunSample,
        outcome: _Outcome | None,
        *,
        output: Path | None,
        container: int | None,
        started_at: str,
        seconds: float | None,
    ) -> dict[str, Any]:
        """Write the next attempt's record; ``outcome`` None evaluates the engine output."""
        attempt, directory = self._new_dir(self.run.path / "samples" / sample.id, "attempt-")
        if output is not None and (output.exists() or output.is_symlink()):
            os.rename(output, directory / "output")  # kept even when the sample failed
        if outcome is None:
            outcome = _evaluate(directory / "output", sample)
        prediction = None
        if outcome.prediction is not None:
            _write_json(directory / "prediction.json", prediction_to_json(outcome.prediction))
            (directory / "prediction.txt").write_text(outcome.prediction.text, encoding="utf-8")
            prediction = "prediction.json"
        record = {
            "format": RECORD_FORMAT,
            "version": RECORD_VERSION,
            "run_id": self.run.run_id,
            "sample_id": sample.id,
            "attempt": attempt,
            "container": container,
            "status": outcome.status,
            "reason": outcome.reason,
            "message": outcome.message,
            "engine_status": outcome.engine_status,
            "started": started_at,
            "finished": _now(),
            "seconds": seconds,
            "engine_seconds": outcome.seconds,
            "image_sha256": sample.image_sha256,
            "primary": outcome.primary,
            "prediction": prediction,
            "findings": [issue.format() for issue in outcome.findings],
            "outputs": _listing(directory / "output"),
            "engine_extra": outcome.extra,
        }
        _write_json(directory / "record.json", record)
        if self.progress is not None:
            self.progress(record)
        return record

    def stage(self, samples: list[RunSample], input_dir: Path) -> list[RunSample]:
        """Copy byte-identical images; record input errors.

        Never a hard link: an in-place edit of the source would change the staged image after
        its hash was checked. A copy (or a copy-on-write clone) keeps the checked bytes.
        """
        images = input_dir / "images"
        images.mkdir(parents=True)
        staged = []
        for sample in samples:
            error = sample.error
            target = images / f"{sample.id}{sample.image.suffix.lower()}"
            if error is None:
                try:
                    shutil.copyfile(sample.image, target)
                    if sha256_file(target) != sample.image_sha256:
                        error = "the image changed since the run was created"
                except OSError as exc:
                    error = f"cannot read image: {exc}"
            if error is not None:
                target.unlink(missing_ok=True)
                outcome = _Outcome("failure", "input-error", error)
                self.record(
                    sample, outcome, output=None, container=None, started_at=_now(), seconds=None
                )
                continue
            staged.append(sample)
        engine = self.run.engine
        job = {
            "format": JOB_FORMAT,
            "version": JOB_VERSION,
            "engine": engine.name,
            "model": engine.model,
            "params": engine.params,
            "samples": [
                {**s.job, "image": f"images/{s.id}{s.image.suffix.lower()}"} for s in staged
            ],
        }
        _write_json(input_dir / "job.json", job)
        return staged

    def container(self, samples: list[RunSample]) -> list[RunSample]:
        """Run one container over ``samples``; return those left for the next container."""
        number, cdir = self._new_dir(self.run.path / "containers", "")
        input_dir, output_dir = cdir / "input", cdir / "output"
        jobs = self.stage(samples, input_dir)
        shutil.copyfile(input_dir / "job.json", cdir / "job.json")
        if not jobs:
            shutil.rmtree(input_dir)
            return []
        output_dir.mkdir()
        name = f"ocrbench-{self.run.run_id}-{number}"[:128]
        engine = self.run.engine
        info: dict[str, Any] = {
            "container": number,
            "samples": [s.id for s in jobs],
            "command": self.launcher.describe(name, input_dir, output_dir),
            "started": _now(),
        }
        done: set[str] = set()
        started = last = time.monotonic()
        last_at = info["started"]
        deadline = started + engine.startup_timeout + engine.sample_timeout
        timed_out = cancelled = False
        code = None
        with open(cdir / "log.txt", "wb") as log:
            process = self.launcher.start(name, input_dir, output_dir, log)

            def harvest() -> None:
                nonlocal last, last_at, deadline
                for sample in jobs:
                    if sample.id in done or not (output_dir / sample.id / RESULT_FILE).exists():
                        continue
                    done.add(sample.id)
                    now = time.monotonic()
                    self.record(
                        sample,
                        None,
                        output=output_dir / sample.id,
                        container=number,
                        started_at=last_at,
                        seconds=round(now - last, 3),
                    )
                    last, last_at = now, _now()
                    deadline = last + engine.sample_timeout

            try:
                while True:
                    code = process.poll()
                    harvest()
                    if code is not None:
                        break
                    if time.monotonic() > deadline:
                        timed_out = True
                        process.kill()
                        code = process.wait()
                        harvest()
                        break
                    time.sleep(self.poll_interval)
            except BaseException:
                cancelled = True
                process.kill()
                code = process.poll()
                raise
            finally:
                info.update(finished=_now(), exit_code=code if not cancelled else None)
                info.update(timed_out=timed_out, cancelled=cancelled)
                _write_json(cdir / "container.json", info)
                shutil.rmtree(input_dir, ignore_errors=True)
        left = [s for s in jobs if s.id not in done]
        if not left:
            _remove_if_empty(output_dir)
            return []
        current = left[0]  # samples are processed in job order
        timing = {"started_at": last_at, "seconds": round(time.monotonic() - last, 3)}
        if timed_out:
            limit = engine.sample_timeout + (0 if done else engine.startup_timeout)
            outcome = _Outcome("failure", "timeout", f"no sample finished within {limit:g} s")
            self.record(
                current, outcome, output=output_dir / current.id, container=number, **timing
            )
            left = left[1:]
        elif code != 0:
            if code in _START_FAILURES and not done:
                raise RunnerError(
                    f"the engine container could not start (exit status {code}); "
                    f"see {cdir / 'log.txt'}"
                )
            message = f"the container exited with status {code} (log: containers/{number}/log.txt)"
            outcome = _Outcome("failure", "crash", message)
            self.record(
                current, outcome, output=output_dir / current.id, container=number, **timing
            )
            left = left[1:]
        else:
            message = "the container finished without a result for this sample"
            for sample in left:
                outcome = _Outcome("failure", "no-output", message)
                self.record(
                    sample, outcome, output=output_dir / sample.id, container=number, **timing
                )
            left = []
        _remove_if_empty(output_dir)
        return left


def _remove_if_empty(directory: Path) -> None:
    with contextlib.suppress(OSError):  # leftovers stay for inspection
        directory.rmdir()


def execute(
    run: Run,
    launcher: Launcher,
    *,
    retry_failed: bool = False,
    poll_interval: float = 0.5,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Process every pending sample (and failed ones with ``retry_failed``); return a summary.

    Finished samples are never re-run. A KeyboardInterrupt stops the container; the sample in
    progress stays pending and a later call continues the run.
    """
    with open(run.path / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RunnerError(f"{run.path} is in use by another runner") from error
        pending = []
        for sample in run.samples:
            record = latest_record(run, sample.id)
            if record is None or (retry_failed and record["status"] == "failure"):
                pending.append(sample)
        executor = _Executor(run, launcher, poll_interval, progress)
        while pending:
            pending = executor.container(pending)
    return summarize(run)
