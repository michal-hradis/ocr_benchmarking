"""Runner: per-sample execution records, restarts, continuation and GT isolation.

Offline tests run the fake engine (tests/fixtures/fake-engine) as a host process. The opt-in
Docker tests build the same engine on the CPU base image and run it through the CLI.
"""

import dataclasses
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from ocrbench.cli import main
from ocrbench.engine import EngineConfigError, load_engine_config, parse_engine_config
from ocrbench.runner import (
    DockerLauncher,
    ProcessLauncher,
    RunnerError,
    create_run,
    execute,
    latest_record,
    load_run,
    summarize,
    verify_image,
)

FAKE_DIR = Path(__file__).parent / "fixtures" / "fake-engine"
FAKE = [sys.executable, str(FAKE_DIR / "fake_engine.py")]
GT_MARKER = "SECRET-GT-MARKER"


def make_dataset(root, units):
    """A manifest with synthetic images (page or paragraph units) and marker-only PAGE XML."""
    data = root / "data set"  # a space in every path
    data.mkdir(parents=True)
    samples = []
    for index, (sid, unit) in enumerate(units.items()):
        Image.new("L", (60 + index, 40), 255).save(data / f"{sid}.PNG")
        (data / f"{sid}.xml").write_text(f"<PcGts>{GT_MARKER}</PcGts>", encoding="utf-8")
        samples.append(
            {
                "id": sid,
                "image": f"{sid}.PNG",
                "gt": f"{sid}.xml",
                "unit": unit,
                "language": ["ces"],
                "doc_type": "newspaper",
                "year": 1900,
                "split": "secret-split",
            }
        )
    manifest = data / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "ocrbench.dataset-manifest",
                "version": 1,
                "name": "synthetic",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def engine(behaviors=None, **overrides):
    data = {
        "name": "fake",
        "engine_version": "1",
        "image": "ocrbench-test/fake-engine:test-v1",
        "model": "fake-model@rev1",
        "params": {"default": "markdown", "behaviors": behaviors or {}},
        "timeouts": {"sample": 10, "startup": 10},
    }
    return parse_engine_config(data | overrides)


def run_fake(tmp_path, units, behaviors=None, *, env=None, **overrides):
    manifest = make_dataset(tmp_path, units)
    run = create_run(
        engine(behaviors, **overrides), manifest, tmp_path / "runs dir", runtime="process"
    )
    summary = execute(run, ProcessLauncher(FAKE, env), poll_interval=0.02)
    return run, summary


def records(run):
    return {s.id: latest_record(run, s.id) for s in run.samples}


def attempt_dir(run, sid, attempt=1):
    return run.path / "samples" / sid / f"attempt-{attempt}"


def test_every_sample_gets_a_record_and_valid_outputs_a_prediction(tmp_path):
    behaviors = {
        "md": "markdown",
        "txt": "text",
        "pred": "prediction",
        "trunc": "truncated",
        "failed": "failed",
        "invalid": "invalid",
        "escape": "escape",
        "badres": "bad-result",
        "skipped": "skip",
    }
    units = {sid: "paragraph" if sid == "txt" else "page" for sid in behaviors}
    run, summary = run_fake(tmp_path, units, behaviors)
    got = {sid: (r["status"], r["reason"]) for sid, r in records(run).items()}
    assert got == {
        "md": ("success", None),
        "txt": ("success", None),
        "pred": ("success", None),
        "trunc": ("truncated", None),
        "failed": ("failure", "engine-error"),
        "invalid": ("failure", "invalid-output"),
        "escape": ("failure", "invalid-output"),
        "badres": ("failure", "invalid-output"),
        "skipped": ("failure", "no-output"),
    }
    assert summary["counts"] == {
        "success": 3,
        "truncated": 1,
        "partial": 0,
        "failure": 5,
        "pending": 0,
    }
    md = attempt_dir(run, "md")
    assert (md / "prediction.txt").read_text(encoding="utf-8") == "md\n\nText of md."
    prediction = json.loads((md / "prediction.json").read_text(encoding="utf-8"))
    assert prediction["conversion"] == {"from": "markdown", "version": 1}
    assert (md / "output" / "page.md").read_text(encoding="utf-8").startswith("# md")
    record = records(run)["md"]
    assert record["primary"] == "page.md"
    assert record["engine_seconds"] == 0.01
    assert {f["path"] for f in record["outputs"]} == {"page.md", "seen.json", "result.json"}
    assert records(run)["trunc"]["message"] == "output token limit reached"
    assert records(run)["failed"]["message"] == "unsupported page"
    assert "sample-id-mismatch" in records(run)["invalid"]["findings"][0]
    # a failure keeps no prediction, and the symlink is listed but never followed
    escape = records(run)["escape"]
    assert escape["prediction"] is None
    assert {"path": "page.txt", "symlink": True} in escape["outputs"]
    assert not (attempt_dir(run, "escape") / "prediction.json").exists()
    # one container processed everything; the staged input is removed
    assert sorted(p.name for p in (run.path / "containers").iterdir()) == ["1"]
    assert not (run.path / "containers" / "1" / "input").exists()


def test_crash_and_timeout_fail_one_sample_and_restart(tmp_path):
    units = dict.fromkeys(["a", "crash", "b", "hang", "c"], "page")
    behaviors = {"crash": "crash", "hang": "hang"}
    run, _ = run_fake(tmp_path, units, behaviors, timeouts={"sample": 0.5, "startup": 5})
    got = {sid: (r["status"], r["reason"], r["container"]) for sid, r in records(run).items()}
    assert got == {
        "a": ("success", None, 1),
        "crash": ("failure", "crash", 1),
        "b": ("success", None, 2),
        "hang": ("failure", "timeout", 2),
        "c": ("success", None, 3),
    }
    assert "exited with status 3" in records(run)["crash"]["message"]
    container = json.loads((run.path / "containers" / "2" / "container.json").read_text())
    assert container["timed_out"] is True
    assert container["samples"] == ["b", "hang", "c"]


def test_engine_sees_only_images_and_allowed_metadata(tmp_path):
    units = {"page1": "page", "para1": "paragraph"}
    run, _ = run_fake(tmp_path, units, metadata=["language", "year"])
    job = (run.path / "containers" / "1" / "job.json").read_text(encoding="utf-8")
    for leak in (GT_MARKER, ".xml", "secret-split", "newspaper", "data set"):
        assert leak not in job
    for sample in run.samples:
        seen = json.loads((attempt_dir(run, sample.id) / "output" / "seen.json").read_text())
        # byte-identical image, no common-runner preprocessing
        assert seen["image_sha256"] == hashlib.sha256(sample.image.read_bytes()).hexdigest()
        assert seen["sample"] == {
            "id": sample.id,
            "unit": units[sample.id],
            "language": ["ces"],
            "year": 1900,
            "image": f"images/{sample.id}.png",
        }
        assert seen["input"] == ["images", "images/page1.png", "images/para1.png", "job.json"]
        assert seen["model"] == "fake-model@rev1"
    frozen = json.loads((run.path / "run.json").read_text(encoding="utf-8"))
    assert frozen["engine"]["metadata"] == ["language", "year"]
    assert frozen["samples"][1]["job"] == {
        "id": "para1",
        "unit": "paragraph",
        "language": ["ces"],
        "year": 1900,
    }


def test_docker_command_is_isolated(tmp_path):
    manifest = make_dataset(tmp_path, {"s1": "page"})
    config = engine(resources={"cpus": 2, "memory": "1g", "gpus": "all"}, env={"OMP": "1"})
    run = create_run(config, manifest, tmp_path / "runs", runtime="process")
    launcher = DockerLauncher(run.engine, "sha256:abc", cache=tmp_path / "ca,che")
    args = launcher.describe("name", tmp_path / "in", tmp_path / "out")
    command = " ".join(args)
    assert GT_MARKER not in command
    assert ".xml" not in command
    for flag in ("--network none", "--read-only", "--gpus all", "--cpus 2.0", "--memory 1g"):
        assert flag in command
    assert f"--user {os.getuid()}:{os.getgid()}" in command
    assert f"type=bind,source={tmp_path / 'in'},target=/input,readonly" in args
    assert f'type=bind,"source={tmp_path / "ca,che"}",target=/cache' in args
    assert args[-1] == "sha256:abc"  # the resolved image ID, not the tag
    assert "--env" in args
    assert "OMP=1" in args


def test_resume_keeps_results_and_retries_failures_as_new_attempts(tmp_path):
    flaky = tmp_path / "flaky"
    flaky.mkdir()
    units = dict.fromkeys(["a", "f", "b"], "page")
    env = {"FAKE_ENGINE_FLAKY_DIR": str(flaky)}
    run, _ = run_fake(tmp_path, units, {"f": "flaky"}, env=env)
    assert records(run)["f"]["reason"] == "crash"
    first = (attempt_dir(run, "a") / "record.json").read_bytes()
    launcher = ProcessLauncher(FAKE, env)
    # continuing without --retry-failed has nothing to do
    execute(load_run(run.path), launcher, poll_interval=0.02)
    assert len(list((run.path / "containers").iterdir())) == 2
    summary = execute(load_run(run.path), launcher, retry_failed=True, poll_interval=0.02)
    assert summary["counts"]["success"] == 3
    assert records(run)["f"]["attempt"] == 2
    assert json.loads((attempt_dir(run, "f", 1) / "record.json").read_text())["reason"] == "crash"
    assert (attempt_dir(run, "a") / "record.json").read_bytes() == first
    assert not attempt_dir(run, "a", 2).exists()
    container = json.loads((run.path / "containers" / "3" / "container.json").read_text())
    assert container["samples"] == ["f"]


def test_interrupted_run_continues_with_pending_samples(tmp_path):
    markers = tmp_path / "markers"
    markers.mkdir()
    launcher = ProcessLauncher(FAKE, {"FAKE_ENGINE_FLAKY_DIR": str(markers)})
    manifest = make_dataset(tmp_path, dict.fromkeys(["a", "b", "c"], "page"))
    run = create_run(engine({"b": "hang-once"}), manifest, tmp_path / "runs", runtime="process")

    def interrupt(record):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(run, launcher, poll_interval=0.02, progress=interrupt)
    assert summarize(run)["counts"] == {
        "success": 1,
        "truncated": 0,
        "partial": 0,
        "failure": 0,
        "pending": 2,
    }
    assert json.loads((run.path / "containers" / "1" / "container.json").read_text())["cancelled"]
    summary = execute(load_run(run.path), launcher, poll_interval=0.02)
    assert summary["counts"]["success"] == 3
    assert [r["attempt"] for r in records(run).values()] == [1, 1, 1]
    container = json.loads((run.path / "containers" / "2" / "container.json").read_text())
    assert container["samples"] == ["b", "c"]


def test_input_errors_and_existing_runs(tmp_path):
    manifest = make_dataset(tmp_path, {"ok": "page", "gone": "page"})
    (manifest.parent / "gone.PNG").unlink()
    run = create_run(engine(), manifest, tmp_path / "runs", run_id="r1", runtime="process")
    execute(run, ProcessLauncher(FAKE), poll_interval=0.02)
    assert records(run)["gone"]["reason"] == "input-error"
    assert records(run)["ok"]["status"] == "success"
    with pytest.raises(RunnerError, match="already exists"):
        create_run(engine(), manifest, tmp_path / "runs", run_id="r1", runtime="process")
    with pytest.raises(RunnerError, match="must match"):
        create_run(engine(), manifest, tmp_path / "runs", run_id="../x", runtime="process")
    with pytest.raises(RunnerError, match="now resolves"):
        verify_image(dataclasses.replace(run, image_id="sha256:old"), "sha256:new")


def test_image_changed_after_creation_is_an_input_error(tmp_path):
    manifest = make_dataset(tmp_path, {"s1": "page"})
    run = create_run(engine(), manifest, tmp_path / "runs", runtime="process")
    Image.new("L", (10, 10), 0).save(manifest.parent / "s1.PNG")
    execute(run, ProcessLauncher(FAKE), poll_interval=0.02)
    assert "changed" in records(run)["s1"]["message"]


def test_run_is_locked_against_a_second_runner(tmp_path):
    manifest = make_dataset(tmp_path, {"s1": "page"})
    run = create_run(engine(), manifest, tmp_path / "runs", runtime="process")
    with open(run.path / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RunnerError, match="in use"):
            execute(run, ProcessLauncher(FAKE))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"surprise": 1}, "unknown field 'surprise'"),
        ({"metadata": ["split"]}, "'metadata'"),
        ({"env": {"OCRBENCH_INPUT": "/gt"}}, "reserved"),
        ({"timeouts": {"sample": 0}}, "timeouts.sample"),
        ({"resources": {"gpu": "all"}}, "resources.gpu"),
        ({"name": "Bad Name"}, "'name'"),
    ],
)
def test_engine_config_errors(change, message):
    with pytest.raises(EngineConfigError, match=message):
        engine(**change)


def test_engine_config_file_and_frozen_hash(tmp_path):
    config = load_engine_config(FAKE_DIR / "engine.toml")
    assert config.metadata == ("language",)
    assert config.cpus == 1.0
    assert parse_engine_config(config.to_json()).sha256() == config.sha256()
    manifest = make_dataset(tmp_path, {"s1": "page"})
    run = create_run(config, manifest, tmp_path / "runs", runtime="process")
    frozen = json.loads((run.path / "run.json").read_text(encoding="utf-8"))
    frozen["engine"]["params"]["default"] = "text"
    (run.path / "run.json").write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(RunnerError, match="modified"):
        load_run(run.path)


def test_cli_status_and_errors(tmp_path, capsys):
    run, _ = run_fake(tmp_path, {"a": "page", "b": "page"}, {"b": "failed"})
    assert main(["run", "status", str(run.path)]) == 0
    out = capsys.readouterr().out
    assert "failure: b: engine-error: unsupported page" in out
    assert "1 success" in out
    assert "1 failure" in out
    assert "0 pending" in out
    assert main(["run", "status", str(tmp_path / "missing")]) == 1
    assert main(["run", "resume", str(run.path)]) == 1
    assert "not run with Docker" in capsys.readouterr().err
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "x"\n', encoding="utf-8")
    args = ["run", "start", "--engine", str(bad), "--dataset", "x.json"]
    assert main([*args, "--runs-dir", str(tmp_path)]) == 1
    assert "engine_version" in capsys.readouterr().err


# Opt-in Docker tests -------------------------------------------------------------------------

FAKE_IMAGE = "ocrbench-test/fake-engine:test-v1"


@pytest.fixture(scope="module")
def fake_image():
    subprocess.run(
        ["docker", "build", "-q", "-t", FAKE_IMAGE, str(FAKE_DIR)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    yield FAKE_IMAGE
    subprocess.run(["docker", "image", "rm", "-f", FAKE_IMAGE], capture_output=True, check=False)


def _docker_engine(tmp_path, extra=""):
    config = tmp_path / "engine.toml"
    config.write_text((FAKE_DIR / "engine.toml").read_text() + extra, encoding="utf-8")
    return config


@pytest.mark.docker
def test_docker_run_over_pages_and_paragraphs(tmp_path, fake_image, capsys):
    manifest = make_dataset(tmp_path, {"page1": "page", "para1": "paragraph", "hang": "page"})
    extra = '\n[params.behaviors]\nhang = "hang"\n'
    config = _docker_engine(tmp_path, extra).read_text().replace("sample = 30", "sample = 3")
    (tmp_path / "engine.toml").write_text(config, encoding="utf-8")
    args = ["run", "start", "--engine", str(tmp_path / "engine.toml"), "--dataset", str(manifest)]
    assert main([*args, "--runs-dir", str(tmp_path / "runs"), "--run-id", "docker1"]) == 0
    run = load_run(tmp_path / "runs" / "docker1")
    got = {sid: (r["status"], r["reason"]) for sid, r in records(run).items()}
    assert got == {
        "page1": ("success", None),
        "para1": ("success", None),
        "hang": ("failure", "timeout"),
    }
    assert run.image_id.startswith("sha256:")
    for sid in ("page1", "para1"):
        seen = json.loads((attempt_dir(run, sid) / "output" / "seen.json").read_text())
        sample = next(s for s in run.samples if s.id == sid)
        assert seen["image_sha256"] == sample.image_sha256
        assert seen["uid"] == os.getuid()
        assert seen["sample"]["unit"] == ("paragraph" if sid == "para1" else "page")
        assert "doc_type" not in seen["sample"]
    assert main(["run", "resume", str(run.path)]) == 0
    assert "2 success" in capsys.readouterr().out


@pytest.mark.docker
@pytest.mark.gpu
def test_docker_run_passes_the_gpu(tmp_path, fake_image):
    manifest = make_dataset(tmp_path, {"s1": "page"})
    config = (
        _docker_engine(tmp_path).read_text().replace("[timeouts]", 'gpus = "all"\n\n[timeouts]')
    )
    (tmp_path / "engine.toml").write_text(config, encoding="utf-8")
    args = ["run", "start", "--engine", str(tmp_path / "engine.toml"), "--dataset", str(manifest)]
    assert main([*args, "--runs-dir", str(tmp_path / "runs"), "--run-id", "gpu"]) == 0
    run = load_run(tmp_path / "runs" / "gpu")
    seen = json.loads((attempt_dir(run, "s1") / "output" / "seen.json").read_text())
    assert seen["nvidia_devices"]
    assert json.loads((run.path / "run.json").read_text())["host"]["gpus"]
