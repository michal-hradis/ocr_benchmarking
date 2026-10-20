"""Opt-in switches for tests that need Docker or an NVIDIA GPU."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-docker", action="store_true", help="run tests marked 'docker'")
    parser.addoption("--run-gpu", action="store_true", help="run tests marked 'gpu'")


def pytest_collection_modifyitems(config, items):
    for marker, option in (("docker", "--run-docker"), ("gpu", "--run-gpu")):
        if config.getoption(option):
            continue
        skip = pytest.mark.skip(reason=f"opt-in: pass {option}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
