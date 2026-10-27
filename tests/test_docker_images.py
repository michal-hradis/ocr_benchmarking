"""Static conventions for base Dockerfiles, plus opt-in smoke runs of built images."""

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = ROOT / "docker"
DERIVED_CPU_DIR = ROOT / "tests" / "fixtures" / "derived-cpu"
# Every Dockerfile in the repository, so future engine images are checked too.
DOCKERFILES = sorted(
    p for p in ROOT.rglob("Dockerfile") if not {".venv", ".git"} & set(p.relative_to(ROOT).parts)
)
# Locally built shared bases are referenced by an immutable versioned tag (docs/DOCKER.md).
LOCAL_BASE_REF = re.compile(r"^ocrbench/base-[a-z0-9-]+:[A-Za-z0-9._-]+-v[0-9]+$")
DIGEST_REF = re.compile(r"@sha256:[0-9a-f]{64}$")
CPU_IMAGE = os.environ.get("CPU_IMAGE", "ocrbench/base-cpu:py3.12-v1")
CU130_TORCH_IMAGE = os.environ.get(
    "CU130_TORCH_IMAGE", "ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1"
)
CU130_PADDLE_IMAGE = os.environ.get(
    "CU130_PADDLE_IMAGE", "ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1"
)


def test_base_dockerfiles_exist():
    assert {p.parent.name for p in DOCKERFILES} >= {
        "base-cpu",
        "base-cu130-torch",
        "base-cu130-paddle",
    }


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p.parent.relative_to(ROOT)))
def test_parent_images_are_pinned(dockerfile):
    text = dockerfile.read_text()
    refs = re.findall(r"^FROM\s+(\S+)", text, re.M) + re.findall(r"COPY\s+--from=(\S+)", text)
    assert refs
    for ref in refs:
        assert DIGEST_REF.search(ref) or LOCAL_BASE_REF.match(ref), (
            f"{ref}: external images need @sha256 digests; local bases need ocrbench/base-*:...-vN"
        )


@pytest.mark.parametrize(
    ("ref", "ok"),
    [
        ("ocrbench/base-cpu:py3.12-v1", True),
        ("ocrbench/base-cu130-torch:torch2.14.1-py3.12-v12", True),
        ("ocrbench/base-cpu:latest", False),
        ("ocrbench/base-cpu", False),
        ("python:3.12-slim", False),
        ("evil/base-cpu:py3.12-v1", False),
    ],
)
def test_local_base_reference_rule(ref, ok):
    assert bool(LOCAL_BASE_REF.match(ref)) is ok


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: str(p.parent.relative_to(ROOT)))
def test_final_user_is_not_root(dockerfile):
    users = re.findall(r"^USER\s+(\S+)", dockerfile.read_text(), re.M)
    assert users
    assert users[-1] not in {"root", "0"}


@pytest.mark.parametrize(
    ("requirements", "pinned"),
    [
        (DOCKER_DIR / "base-cu130-torch" / "requirements.txt", "torch==2.14.1+cu130"),
        (DOCKER_DIR / "base-cu130-paddle" / "requirements.txt", "paddlepaddle-gpu @ https://"),
        (ROOT / "engines" / "paddleocr" / "requirements.txt", "paddleocr==3.7.0"),
        (ROOT / "engines" / "kraken" / "requirements.txt", "kraken==7.1.1"),
        (ROOT / "engines" / "pylaia" / "pylaia-requirements.txt", "pylaia==1.1.2"),
        (ROOT / "engines" / "lightonocr" / "requirements.txt", "transformers==5.19.0"),
        (ROOT / "engines" / "fireredocr" / "requirements.txt", "transformers==5.19.0"),
        (DOCKER_DIR / "base-cu130-vllm026" / "requirements.txt", "vllm==0.26.0"),
        (ROOT / "engines" / "nemotron-parse" / "requirements.txt", "timm==1.0.22"),
    ],
    ids=lambda v: v.parent.name if isinstance(v, Path) else None,
)
def test_requirements_are_pinned_and_hashed(requirements, pinned):
    lines = requirements.read_text().splitlines()
    pins = [line for line in lines if re.match(r"^[A-Za-z0-9]", line)]
    assert any(line.startswith(pinned) for line in pins)
    assert all("==" in line or " @ https://" in line for line in pins)
    assert all(line.endswith("\\") for line in pins)  # each pin is followed by its hashes
    assert "--hash=sha256:" in "\n".join(lines)


def _run_in(image, code, *docker_args):
    cmd = ["docker", "run", "--rm", "--network", "none", "--read-only", *docker_args]
    cmd += [image, "python", "-c", code]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True).stdout


@pytest.mark.docker
@pytest.mark.parametrize("image", [CPU_IMAGE, CU130_TORCH_IMAGE])
def test_image_runs_offline_as_non_root(image):
    out = _run_in(image, "import os, sys; print(os.getuid(), sys.version_info[:2])")
    uid, version = out.split(" ", 1)
    assert uid != "0"
    assert version.strip() == "(3, 12)"


def _layers(image):
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RootFS.Layers}}", image],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


@pytest.mark.docker
def test_derived_engine_image_inherits_local_base():
    tag = "ocrbench-test/derived-cpu:smoke"
    subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(DERIVED_CPU_DIR)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    try:
        base = re.search(r"^FROM\s+(\S+)", (DERIVED_CPU_DIR / "Dockerfile").read_text(), re.M)[1]
        base_layers = _layers(base)
        assert _layers(tag)[: len(base_layers)] == base_layers
        out = _run_in(tag, "import os; print(os.getuid(), open('/opt/ocrbench-derived').read())")
        uid, marker = out.split(" ", 1)
        assert uid != "0"
        assert marker.strip() == "derived-cpu fixture"
    finally:
        subprocess.run(["docker", "image", "rm", "-f", tag], capture_output=True, check=False)


@pytest.mark.docker
def test_torch_imports_without_gpu():
    out = _run_in(CU130_TORCH_IMAGE, "import torch; print(torch.__version__)")
    assert out.strip() == "2.14.1+cu130"


@pytest.mark.docker
@pytest.mark.gpu
def test_torch_uses_gpu():
    code = (
        "import torch; assert torch.cuda.is_available();"
        "x = torch.ones(1024, 1024, device='cuda'); print(torch.cuda.get_device_name(0), "
        "float((x @ x).sum()))"
    )
    out = _run_in(CU130_TORCH_IMAGE, code, "--gpus", "all")
    assert out.strip().endswith(str(1024.0**3))
