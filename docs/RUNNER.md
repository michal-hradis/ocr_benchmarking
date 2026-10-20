# Runner and engine protocol

The runner executes one OCR engine, packaged as a Docker image, over a dataset manifest. It records an execution record for every sample, including samples that failed, and keeps the engine's native output together with the standardized prediction. The decisions behind it are recorded in [#7](https://github.com/michal-hradis/ocr_benchmarking/issues/7#issuecomment-6300080767) and [ADR 0007](adr/0007-runner-and-engine-protocol.md).

## Commands

```sh
uv run ocrbench run start --engine engines/<name>/engine.toml --dataset datasets/dev-01.json \
    [--runs-dir runs] [--run-id <id>] [--cache <dir>]
uv run ocrbench run resume runs/<run-id> [--retry-failed]
uv run ocrbench run status runs/<run-id>
```

- **`start`** creates a new run directory with a frozen configuration and processes every sample. The engine image must already exist locally: the runner never builds or pulls images. `--cache` mounts a host directory at `/cache` for model weights ([DOCKER.md](DOCKER.md)). An existing run ID is refused.
- **`resume`** continues an interrupted run. Samples that already have a record are kept and not run again. `--retry-failed` also retries failed samples; each retry is a new attempt and earlier attempts are kept. Continuing is refused if the image tag now resolves to a different image ID.
- **`status`** counts samples by status and lists failures.

`start` and `resume` print one line per finished sample and a summary. They exit with status 0 when the run completes, even if samples failed (failures are results); 1 on an error that stops the run; and 130 when interrupted with Ctrl-C, after which the run can be continued.

## Engine configuration

An engine configuration is a TOML file, usually next to the engine's Dockerfile.

```toml
name = "tesseract"                    # [a-z0-9][a-z0-9._-]*
engine_version = "1"                  # version of the adapter
image = "ocrbench/engine-tesseract:5.5.1-v1"
command = []                          # optional; empty uses the image's CMD
model = "ces+deu@4.1.0"               # optional model name and revision, passed to the engine
metadata = ["language"]               # optional metadata to pass: language, doc_type, year
env = { OMP_THREAD_LIMIT = "4" }      # optional container environment

[params]                              # free parameters passed to the engine (prompts, limits, ...)
max_new_tokens = 8192

[resources]                           # all optional
cpus = 4
memory = "16g"
gpus = "all"                          # Docker --gpus value; omit for CPU-only engines
shm_size = "2g"

[timeouts]                            # seconds
sample = 600                          # without a finished sample before the container is stopped
startup = 300                         # extra allowance for the first sample after each start
```

Unknown fields are errors. The parsed configuration is frozen into `run.json` with its SHA-256 hash, together with the resolved image ID. A changed configuration needs a new run.

## Engine protocol

**One container per run.** The runner starts one container with all pending samples. It processes them in the order of the job file and stops. The model is loaded once per container.

**Input** (`$OCRBENCH_INPUT`, mounted read-only at `/input`):

```json
{
  "format": "ocrbench.job",
  "version": 1,
  "engine": "tesseract",
  "model": "ces+deu@4.1.0",
  "params": {"max_new_tokens": 8192},
  "samples": [
    {"id": "page-0001", "unit": "page", "language": ["ces"], "image": "images/page-0001.jpg"}
  ]
}
```

- `job.json` and `images/<id><suffix>` are the only files in the input. Each image is byte-identical to the dataset image. The runner never resizes, crops or tiles it ([ADR 0001](adr/0001-image-only-samples-engine-owned-preprocessing.md)).
- **Allowed metadata.** Every sample has `id`, `unit` (`page` or `paragraph`) and `image`. `language`, `doc_type` and `year` are included only when listed in the configuration's `metadata` and present in the manifest. GT paths, GT content, statistics derived from GT, splits and checksums are never passed ([ADR 0002](adr/0002-ground-truth-isolation.md)).

**Output** (`$OCRBENCH_OUTPUT`, mounted at `/output`): for each sample the engine writes a directory `<id>/` with its native outputs and, **last**, `result.json`:

```json
{"status": "ok", "primary": "page.md", "message": "optional", "seconds": 1.7, "extra": {}}
```

| Field | Meaning |
| --- | --- |
| `status` | `ok`, `truncated` (for example the output token limit was reached), `partial` (part of the page was not processed) or `failed` |
| `primary` | The file to evaluate, relative to the sample directory. It is required unless `status` is `failed`. `.txt`, `.md`/`.markdown` and `.html`/`.htm` are converted on the host ([PREDICTIONS.md](PREDICTIONS.md#host-side-conversion-of-native-output)); `.json` must be a prediction JSON |
| `message`, `seconds`, `extra` | Optional: a short explanation, the engine's own processing time and any engine-specific data |

Write `result.json` atomically (write a temporary file in the same directory, then rename it). Its appearance tells the runner that the sample is finished. An engine that cannot process a sample writes `status: failed` and continues with the next one.

**Container environment.** Containers run with `--network none`, a read-only root filesystem with a writable `/tmp`, the host user's UID/GID (so output files belong to the host user) and `HOME=/tmp`. The configured CPU, memory, shared-memory and GPU limits are applied, and `/cache` is mounted only with `--cache`. The image's own user is therefore not used. Engines must not rely on writing anywhere except `/output`, `/tmp` and `/cache`.

## Statuses and failures

The runner decides the final status of every sample. The engine's claim is accepted only when the output is valid.

| Status | Meaning |
| --- | --- |
| `success` | Valid output, claimed `ok` |
| `truncated` | Valid output, claimed `truncated` |
| `partial` | Valid output, claimed `partial` |
| `failure` | No valid prediction; `reason` says why |

| Failure reason | When |
| --- | --- |
| `timeout` | No sample finished within `timeouts.sample` (plus `timeouts.startup` for the first sample after a start). The container is stopped |
| `crash` | The container exited with a non-zero status, including the out-of-memory killer, before writing the sample's result |
| `no-output` | The container exited normally without a result for the sample |
| `invalid-output` | `result.json` or the primary output is missing, malformed, too large, a symbolic link, outside the sample directory, or fails prediction validation (for example a wrong sample ID or image size) |
| `engine-error` | The engine reported `status: failed` |
| `input-error` | The image cannot be read on the host, or it changed after the run was created |

**Restarts.** After a timeout or a crash, only the sample in progress fails: the first unfinished sample in job order. A new container then continues with the remaining samples. If the container cannot start at all (Docker exit status 125–127 before any sample finishes), the run stops with an error and the samples stay pending.

A sample without any record is **pending**, for example after Ctrl-C. Pending samples are processed by `resume`. A failure is never turned into an empty successful transcript; how the evaluator scores failures is defined in [EVALUATION.md](EVALUATION.md#failed-partial-and-truncated-samples).

## Run directory

```text
runs/<run-id>/
  run.json                      frozen configuration, image ID, dataset, host, samples
  containers/<n>/
    container.json              command, samples, start/end, exit status, timeout/cancel
    job.json                    the job given to the container
    log.txt                     container stdout and stderr
    output/                     only leftovers the runner did not collect, if any
  samples/<id>/attempt-<k>/
    record.json                 execution record (below)
    output/                     the engine's sample directory, unchanged
    prediction.json             standardized prediction (valid outputs only)
    prediction.txt              its page text, UTF-8
```

- `run.json` records the engine configuration and its hash, the resolved image ID, the manifest path and hash, and the ocrbench version. It also records the host (platform, CPU count, memory, Docker version and, for GPU engines, the GPU names) and, for every sample, the image path, SHA-256 and size and the exact job entry passed to the engine.
- Directories are created with exclusive numbering and never reused. `run.json` is written once. A run directory is locked while a runner uses it.
- `record.json` (`ocrbench.execution-record`, version 1) holds `status`, `reason`, `message`, `engine_status`, `attempt`, `container`, `started`, `finished`, `seconds`, `engine_seconds`, `image_sha256`, `primary`, `prediction`, `findings` (validation findings), `outputs` (files with sizes; symbolic links are listed, never followed) and `engine_extra`. `seconds` is measured by the runner from the previous sample's end, or from the container start for the first sample, so it includes model loading.
- The newest attempt with a record is the sample's current state.

## Tests and the fake engine

[`tests/fixtures/fake-engine`](../tests/fixtures/fake-engine) contains a fake engine (standard library only) and its Dockerfile on the CPU base. It simulates valid output, failures, crashes, hangs and malicious output. `make test` runs it as a host process. This has no isolation and is meant only for tests. `make docker-smoke` builds the image and runs it through `ocrbench run start` over page and paragraph samples, and `make gpu-smoke` checks that `resources.gpus` reaches the container.
