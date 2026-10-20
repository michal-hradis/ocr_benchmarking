"""Static conventions for base Dockerfiles, plus opt-in smoke runs of built images."""

import os
import re
import subprocess
from pathlib import Path

import pytest

DOCKER_DIR = Path(__file__).resolve().parents[1] / "docker"
DOCKERFILES = sorted(DOCKER_DIR.glob("*/Dockerfile"))
CPU_IMAGE = os.environ.get("CPU_IMAGE", "ocrbench/base-cpu:py3.12-v1")
CU130_TORCH_IMAGE = os.environ.get(
    "CU130_TORCH_IMAGE", "ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1"
)


def test_base_dockerfiles_exist():
    assert {p.parent.name for p in DOCKERFILES} >= {"base-cpu", "base-cu130-torch"}


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.parent.name)
def test_external_images_are_pinned_by_digest(dockerfile):
    text = dockerfile.read_text()
    refs = re.findall(r"^FROM\s+(\S+)", text, re.M) + re.findall(r"COPY\s+--from=(\S+)", text)
    assert refs
    for ref in refs:
        assert re.search(r"@sha256:[0-9a-f]{64}$", ref), f"{ref} is not pinned by digest"


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.parent.name)
def test_final_user_is_not_root(dockerfile):
    users = re.findall(r"^USER\s+(\S+)", dockerfile.read_text(), re.M)
    assert users
    assert users[-1] not in {"root", "0"}


def test_torch_requirements_are_hashed():
    lines = (DOCKER_DIR / "base-cu130-torch" / "requirements.txt").read_text().splitlines()
    pins = [line for line in lines if re.match(r"^[A-Za-z0-9]", line)]
    assert any(line.startswith("torch==2.14.1+cu130") for line in pins)
    assert all("==" in line for line in pins)
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
