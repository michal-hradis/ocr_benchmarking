"""Engine configuration files (TOML) for the runner.

An engine configuration names the Docker image and command of one OCR engine adapter, the
parameters passed to it, the optional manifest metadata it may receive, its resource limits and
its timeouts. The format is documented in docs/RUNNER.md. A hosted-API engine also declares
outbound network access and the names of the host secrets it needs (ADR 0013). The parsed
configuration is frozen into every run and identified by a hash, so a run can only be continued
with the same settings.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENGINE_FORMAT = "ocrbench.engine"
ENGINE_VERSION = 1
# Optional manifest metadata an engine may receive when its configuration enables it. Sample ID
# and unit are always passed; GT paths, splits and checksums never are (ADR 0002).
OPTIONAL_METADATA = ("language", "doc_type", "year")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Set by the runner; an engine configuration cannot override them.
RESERVED_ENV = {"OCRBENCH_INPUT", "OCRBENCH_OUTPUT", "HOME"}
# "none": no network (the default); "egress": outbound access for a hosted API (ADR 0013).
NETWORK_MODES = ("none", "egress")

_TOP_KEYS = {"format", "version", "name", "engine_version", "image", "command", "model"}
_TOP_KEYS |= {"metadata", "params", "env", "resources", "timeouts", "network", "secrets"}
_RESOURCE_KEYS = {"cpus", "memory", "gpus", "shm_size"}
_TIMEOUT_KEYS = {"sample", "startup"}


class EngineConfigError(ValueError):
    """The engine configuration is invalid; ``errors`` lists every problem found."""

    def __init__(self, location: str, errors: list[str]) -> None:
        super().__init__(f"{location}: " + "; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class EngineConfig:
    name: str
    engine_version: str  # version of the adapter, recorded with every result
    image: str  # local tag or digest reference; the run records the resolved image ID
    command: tuple[str, ...] = ()  # empty: the image's default command
    model: str | None = None  # model name and revision, passed to the engine
    metadata: tuple[str, ...] = ()  # enabled optional metadata, a subset of OPTIONAL_METADATA
    params: dict[str, Any] = field(default_factory=dict)  # free engine parameters (job.json)
    env: dict[str, str] = field(default_factory=dict)
    cpus: float | None = None
    memory: str | None = None  # Docker size, e.g. "16g"
    gpus: str | None = None  # Docker --gpus value, e.g. "all" or "device=0"; None: CPU only
    shm_size: str | None = None
    sample_timeout: float = 600.0  # seconds without a finished sample before the container stops
    startup_timeout: float = 300.0  # extra allowance for the first sample after each start
    network: str = "none"  # one of NETWORK_MODES
    secrets: tuple[str, ...] = ()  # host environment variables forwarded by name, never by value

    def to_json(self) -> dict[str, Any]:
        data = {
            "format": ENGINE_FORMAT,
            "version": ENGINE_VERSION,
            "name": self.name,
            "engine_version": self.engine_version,
            "image": self.image,
            "command": list(self.command),
            "model": self.model,
            "metadata": list(self.metadata),
            "params": self.params,
            "env": self.env,
            "resources": {
                "cpus": self.cpus,
                "memory": self.memory,
                "gpus": self.gpus,
                "shm_size": self.shm_size,
            },
            "timeouts": {"sample": self.sample_timeout, "startup": self.startup_timeout},
        }
        # Only when set, so the frozen form and hash of offline engines are unchanged.
        if self.network != "none":
            data["network"] = self.network
        if self.secrets:
            data["secrets"] = list(self.secrets)
        return data

    def sha256(self) -> str:
        canonical = json.dumps(self.to_json(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


def _optional_str(table: dict[str, Any], key: str, where: str, errors: list[str]) -> str | None:
    value = table.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        errors.append(f"{where}{key!r} must be a non-empty string")
        return None
    return value


def parse_engine_config(data: Any, location: str = "engine") -> EngineConfig:
    """Check parsed TOML or frozen JSON and return the configuration."""
    if not isinstance(data, dict):
        raise EngineConfigError(location, ["the configuration must be a table"])
    errors = [f"unknown field {key!r}" for key in sorted(set(data) - _TOP_KEYS)]
    if data.get("format", ENGINE_FORMAT) != ENGINE_FORMAT:
        errors.append(f"'format' must be {ENGINE_FORMAT!r}")
    if data.get("version", ENGINE_VERSION) != ENGINE_VERSION:
        errors.append(f"version {data.get('version')!r} is not supported")
    name = data.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name):
        errors.append(f"'name' must match {NAME_RE.pattern}")
    for key in ("engine_version", "image"):
        if not isinstance(data.get(key), str) or not data[key]:
            errors.append(f"{key!r} must be a non-empty string")
    command = data.get("command", [])
    if not (isinstance(command, list) and all(isinstance(c, str) and c for c in command)):
        errors.append("'command' must be a list of non-empty strings")
        command = []
    model = _optional_str(data, "model", "", errors)
    metadata = data.get("metadata", [])
    if not (isinstance(metadata, list) and set(metadata) <= set(OPTIONAL_METADATA)):
        errors.append(f"'metadata' must be a list of {OPTIONAL_METADATA}")
        metadata = []
    elif len(set(metadata)) != len(metadata):
        errors.append("'metadata' lists a field twice")
    params = data.get("params", {})
    if not isinstance(params, dict):
        errors.append("'params' must be a table")
        params = {}
    env = data.get("env", {})
    if not isinstance(env, dict):
        errors.append("'env' must be a table")
        env = {}
    for key, value in env.items():
        if not ENV_NAME_RE.match(key) or key in RESERVED_ENV:
            errors.append(f"environment variable name {key!r} is invalid or reserved")
        if not isinstance(value, str):
            errors.append(f"environment variable {key!r} must be a string")
    network = data.get("network", "none")
    if network not in NETWORK_MODES:
        errors.append(f"'network' must be one of {NETWORK_MODES}")
    secrets = data.get("secrets", [])
    if not (isinstance(secrets, list) and all(isinstance(s, str) for s in secrets)):
        errors.append("'secrets' must be a list of environment variable names")
        secrets = []
    for key in secrets:
        if not ENV_NAME_RE.match(key) or key in RESERVED_ENV or key in env:
            errors.append(f"secret name {key!r} is invalid, reserved or also in 'env'")
    if len(set(secrets)) != len(secrets):
        errors.append("'secrets' lists a name twice")
    if secrets and network == "none":
        errors.append("'secrets' needs network = \"egress\"; offline engines get no secrets")
    resources = data.get("resources", {})
    timeouts = data.get("timeouts", {})
    if not isinstance(resources, dict):
        errors.append("'resources' must be a table")
        resources = {}
    if not isinstance(timeouts, dict):
        errors.append("'timeouts' must be a table")
        timeouts = {}
    errors += [f"unknown field 'resources.{k}'" for k in sorted(set(resources) - _RESOURCE_KEYS)]
    errors += [f"unknown field 'timeouts.{k}'" for k in sorted(set(timeouts) - _TIMEOUT_KEYS)]
    cpus = resources.get("cpus")
    if cpus is not None and not _number(cpus):
        errors.append("'resources.cpus' must be a positive number")
    memory = _optional_str(resources, "memory", "resources.", errors)
    gpus = _optional_str(resources, "gpus", "resources.", errors)
    shm_size = _optional_str(resources, "shm_size", "resources.", errors)
    for key in _TIMEOUT_KEYS:
        if key in timeouts and not _number(timeouts[key]):
            errors.append(f"'timeouts.{key}' must be a positive number of seconds")
    if errors:
        raise EngineConfigError(location, errors)
    return EngineConfig(
        name=name,
        engine_version=data["engine_version"],
        image=data["image"],
        command=tuple(command),
        model=model,
        metadata=tuple(metadata),
        params=params,
        env=dict(env),
        cpus=float(cpus) if cpus is not None else None,
        memory=memory,
        gpus=gpus,
        shm_size=shm_size,
        sample_timeout=float(timeouts.get("sample", 600.0)),
        startup_timeout=float(timeouts.get("startup", 300.0)),
        network=network,
        secrets=tuple(secrets),
    )


def load_engine_config(path: Path) -> EngineConfig:
    path = Path(path)
    try:
        with open(path, "rb") as stream:
            data = tomllib.load(stream)
    except OSError as error:
        raise EngineConfigError(path.name, [f"cannot read: {error}"]) from error
    except tomllib.TOMLDecodeError as error:
        raise EngineConfigError(path.name, [f"invalid TOML: {error}"]) from error
    return parse_engine_config(data, path.name)
