#!/usr/bin/env bash
# Check that Docker containers can use an NVIDIA GPU. Prints a diagnosis and
# exits non-zero on the first failed step. Does not install anything.
set -uo pipefail

CUDA_PROBE_IMAGE="${CUDA_PROBE_IMAGE:-nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1}"

fail() {
    echo "FAIL: $1" >&2
    echo "  hint: $2" >&2
    echo "  CPU-only work (make check, CPU images) does not need a GPU." >&2
    exit 1
}

echo "1/4 host driver"
command -v nvidia-smi >/dev/null \
    || fail "nvidia-smi not found on the host" "install the NVIDIA driver (no host CUDA toolkit is needed)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    || fail "nvidia-smi cannot talk to the driver" "check 'lsmod | grep nvidia', Secure Boot/MOK, or reboot after a driver update"

echo "2/4 docker daemon"
command -v docker >/dev/null || fail "docker not found" "install Docker Engine for Ubuntu (docs/DOCKER.md)"
docker info >/dev/null 2>&1 \
    || fail "cannot reach the Docker daemon" "start dockerd or add your user to the 'docker' group"

echo "3/4 NVIDIA container runtime"
docker info --format '{{json .Runtimes}}' | grep -q nvidia \
    || fail "no 'nvidia' runtime registered with Docker" \
        "install NVIDIA Container Toolkit, run 'sudo nvidia-ctk runtime configure --runtime=docker' and restart docker"

echo "4/4 GPU inside a container ($CUDA_PROBE_IMAGE)"
docker run --rm --gpus all "$CUDA_PROBE_IMAGE" nvidia-smi -L \
    || fail "container could not access the GPU" \
        "check toolkit/driver versions and 'docker run --gpus all' errors above; the driver must support CUDA 13.0"

echo "OK: Docker containers can use the GPU"
