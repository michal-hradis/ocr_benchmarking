# Indra OCR engine (CPU image pipeline)

Modern CPU baseline ([#100](https://github.com/michal-hradis/ocr_benchmarking/issues/100)). [Indra OCR](https://github.com/datalab-to/indra) by Datalab is a small ONNX Runtime pipeline with no PyTorch and no GPU. For one page image it:

1. estimates the small skew and the quarter turn of the page (`orient.onnx`) and turns and straightens the page;
2. detects text lines (`det.onnx`);
3. routes each line to a script recognizer (`script.onnx`: Latin/Cyrillic/Greek `rec.onnx`, CJK, right-to-left, other scripts) and reads it with CTC; unsure lines are re-read with n-gram language models;
4. groups the lines into labeled layout blocks and orders them (`order.onnx`).

The adapter calls Indra's **image** OCR on the decoded page (`Converter.ocr`) with `mode="ocr"` and math off. PDF input and the PDF text layer are never used. The pipeline has no other settings ([Parameters](#parameters)). Indra is independent of Docling and other orchestration backends here.

It runs on the shared CPU base and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-indra-ocr:1.1.2-v1`, with the models |
| `requirements.in`, `requirements.txt` | Indra OCR and its dependencies, hash-locked for Python 3.12 (`make docker-lock`) |
| `models.sha256` | Checksums of the model files and their license; checked by the build and by the adapter before it loads the models |
| `indra_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions and license

| Component | Version |
| --- | --- |
| `indra-ocr` | 1.1.2 (PyPI, tag `v1.1.2`) |
| ONNX Runtime (CPU) | 1.31.1 |
| `opencv-python-headless` | 5.0.0.93 |
| numpy | 2.5.4 |
| Models | Hugging Face [`datalab-to/indra-ocr`](https://huggingface.co/datalab-to/indra-ocr), tag `v1.1.0` (the revision Indra 1.1.2 uses) at commit `a1283c4b42a273c09480ad50e71461daff50fcc7` |

The code is Apache-2.0. The **model weights use a modified OpenRAIL-M license** with use restrictions ([LICENSE](https://huggingface.co/datalab-to/indra-ocr/blob/main/LICENSE)); check it before you use the engine. A copy is in the image next to the models.

**Models.** The default model set of Indra 1.1.2 (12 files, 73 MB: the networks, `meta.json` and the language models, with the license) is downloaded **when the image is built** and checked against `models.sha256`, so inference needs no `--cache` and no network. The optional math models and the recognizer's unused `model.safetensors` copy are not downloaded. Before loading, the adapter checks every file against `models.sha256` again. A missing or modified file makes every sample fail with `engine-error`. The job's `model` (from `engine.toml`) must be exactly `datalab-to/indra-ocr@v1.1.0:a1283c4b42a273c09480ad50e71461daff50fcc7`, the models in the image. Any other value, or none, makes every sample fail with `engine-error` before Indra is loaded, so a result is never attributed to another model. `INDRA_OCR_MODEL_DIR` points Indra at the image's folder and `HF_HUB_OFFLINE=1` rules out downloads. ONNX Runtime's telemetry is off (`ORT_DISABLE_TELEMETRY=1`).

## Build, run and evaluate

```sh
make docker-cpu docker-indra-ocr    # the CPU base and this image (with the models)
uv run ocrbench run start --engine engines/indra-ocr/engine.toml --dataset datasets/dev-01.json
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/indra-ocr --inspect
make indra-ocr-smoke                # opt-in: build the image, run it on synthetic samples on the CPU
```

The container gets 4 CPUs (`--cpus 4`) and 4 GB of memory. The runner sends one page at a time.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `threads` | `4` | Threads for one page: ONNX Runtime's intra-op threads, Indra's page thread pool and OpenCV's threads. `engine.toml` sets `OMP_NUM_THREADS` and `cpus` to the same number |

These are fixed in the adapter. They are Indra's defaults for an image, except that the PDF text layer is ruled out:
- `mode="ocr"`: OCR only, never a text layer;
- `math=False`: no math models, so formulas come out as plain text;
- `furniture=False`: page headers and footers are left out of Indra's Markdown only; they stay in the lines and blocks, so they are in the prediction;
- CPU execution provider, orientation detection on, language models on.

A math preset (`math=True`, 20 MB more models, LaTeX inside lines) is not included. Its line text would need host-side markup conversion.

## Outputs

Per sample, in the sample's output directory:

| File | Content |
| --- | --- |
| `indra.json` | Native page, with the fields of `indra-ocr --format json`: `source` (`ocr`), `rotation`, `skew`, `vertical`, `note`, `lines` (`box`, `text`, `conf`), `markdown`, `blocks` (`label`, `bbox`, `text`) and `timings` |
| `prediction.json` | Standardized prediction (primary output) |

**Mapping.** Indra's blocks split its reading-ordered lines into consecutive runs: a block's `text` is its lines joined by line breaks. Each block therefore takes the next lines in reading order. The adapter checks that the texts agree and that every line is used. Otherwise the sample fails rather than guessing.

- Every block becomes a region in Indra's block order (`reading_order`), with its box as polygon and its label as `native_type`. A `Picture` block is a `figure` region without text.
- Each line keeps its box as a 4-point polygon, its text unchanged and its confidence (`conf`). Lines are numbered `l1`, `l2`, … in reading order. A blank line is left out but keeps its place in the numbering.
- A page that Indra reads without layout blocks (fewer than two lines) keeps its lines in one region without a position (`extra.grouping: "no-layout"`).
- The page text is the lines of each region, with a blank line between regions.
- Line text is plain text, so no host-side markup conversion is needed.

| Indra label | Region type |
| --- | --- |
| `Text`, `List-Group` | `text` |
| `Section-Header` | `heading` |
| `Page-Header`, `Page-Footer` | `page-header`, `page-footer` |
| `Table`, `Caption`, `Footnote` | `table`, `caption`, `footnote` |
| `Picture` | `figure` |
| `Table-Of-Contents`, `Form`, other | `other` |

**Coordinates.** The image is decoded as stored, in grayscale as Indra reads it, without EXIF orientation (`IMREAD_IGNORE_ORIENTATION`). Indra maps boxes back from its skew correction itself. When it turns a page by a quarter or half turn (`rotation` 90, 180 or 270), its boxes are on the turned page, and the adapter turns them back exactly onto the original image. A vertical CJK page already has its boxes on the page as it is. The synthetic tests and a real run on `PAGE` turned 90° and 180° check this.

`result.json` `extra` records the line, block and region counts, `rotation`, Indra's own per-stage `timings` and its time for the page (`indra_seconds`), the package versions, `threads`, the CPUs the process may run on (`affinity_cpus`), the container's CPU quota (`cpu_max`, from Docker's `--cpus`), the model load time, the model revision and the verified model checksums.

## Observed behaviour and restrictions

- **Czech.** Upstream lists Czech among the Latin languages read less well: the recognizer sometimes writes a close but wrong diacritic. This is visible on the dev-01 newspapers (`tëchto`, `nendležejí`, `Novdk`).
- **Dense newspapers.** On the two large Czech newspaper pages, Indra found 339 and 247 lines where the GT has 419 and 484.
- **Scripts.** Glagolitic is not among Indra's scripts; `CkXII61-0017` comes out as Latin-like text.
- **Orientation.** A synthetic page turned 90° clockwise was not detected as turned (orient.onnx under its 0.95 threshold) and was read as garbage. The page turned counter-clockwise and the upside-down page were turned back and read correctly.

## Measured on dev-01

Run `dev01-indra-ocr-1.1.2-v1` (not in Git) on the seven pages of dev-01, image `sha256:303b11fe…` built from the committed files, AMD Ryzen 5 2600X (6 cores, 12 threads), `--cpus 4`, `threads = 4`, one page at a time. The default evaluation configuration was used. All 7 samples were `success`. Rates are micro averages; fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)). Engine time is the sum of the adapter's per-sample times, without the 1.1 s model load. Tesseract (`tesseract-fast-psm3-dev01`, 1 CPU) and PP-OCRv6 (`paddleocr-dev01`, RTX 4090) are earlier runs, shown for comparison.

| Engine | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time (7 pages) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `indra-ocr` | 65.35 % | 81.71 % | 50.12 % | 65.09 % | 60.09 | 42.91 | 7.5 s |
| `tesseract` (`fast-psm3`) | 48.15 % | 62.04 % | 27.05 % | 40.88 % | 81.68 | 69.08 | 123 s |
| `paddleocr` (PP-OCRv6, GPU) | 71.97 % | 89.18 % | 12.08 % | 25.06 % | 85.20 | 65.92 | 12 s |

CER per page in % (fCER in parentheses where it differs), with Indra's line count and the time of Indra's pipeline for the page (`indra_seconds`).

| Page | Content | Lines (GT) | CER (fCER) | Indra time |
| --- | --- | ---: | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 339 (419) | 75.22 (45.68) | 2.99 s |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 247 (484) | 74.93 (65.17) | 2.26 s |
| `CkXII61-0017` | Glagolitic print | 21 (19) | 79.96 | 0.48 s |
| `dta-346672…` | Latin, Antiqua and italic | 36 (34) | 8.89 | 0.36 s |
| `dta-laube…` | German Fraktur, 1833 | 25 (25) | 11.42 | 0.30 s |
| `dta-martzi…` | German Fraktur, 1692 | 38 (29) | 16.12 | 0.47 s |
| `dta-steinen…` | German Antiqua, 1894 | 43 (44) | 1.38 | 0.45 s |

- **Fast, but weak on the newspapers.** Indra takes 0.3–0.5 s for a DTA page and 2–3 s for a newspaper page on 4 CPUs. The newspapers dominate the micro averages: their recognition errors on Czech letters and their missed lines ([above](#observed-behaviour-and-restrictions)) keep the flexible rates high. On `3e6309d0…` the ordered CER (75.2 %) is much higher than the flexible one (45.7 %), so Indra's block order there differs from the GT reading order.
- **Single and parallel pages.** The benchmark runs one page at a time with 4 threads. Separately, Indra's own `convert_many` with 4 single-threaded worker processes in the same image, `--cpus 4` and `--network none`, read the seven pages in 8.3–8.4 s wall time (two runs), including worker start-up and model loading. With one page per task, the two newspaper pages dominate, so parallel pages gave no speed-up here. Parallel throughput is not part of the benchmark's engine time.
- The development set is small and not representative; these numbers are not a general ranking.

## Tests

`tests/test_indra_ocr.py` checks the mapping on committed native pages. The files in `tests/fixtures/indra-ocr/` are real `indra.json` output of this image for synthetic pages rendered by `tests/synthetic.py`: the single-column, two-column and newspaper pages, a blank page, the single-column page turned 90° counter-clockwise, and a one-line image without layout blocks. The tests check reading order, line and block geometry in input-image pixels, the quarter-turn mapping for all four rotations on a rendered image, types, pictures, blank lines, and the rejection of blocks and lines that do not agree. The protocol test runs the adapter as a host process with fake `indra_ocr` and `cv2` modules (`tests/fixtures/indra-ocr/fake`) through the runner. It covers success, an empty page, a failing page, the recorded environment and model checksums, the Indra arguments (OCR mode, no math, CPU, threads) and setup errors, including a job that requests another model. The tests use placeholder model files with their own checksums, and check that a missing or modified file is rejected.

The opt-in `make indra-ocr-smoke` builds the image and runs real Indra OCR on the synthetic pages, including the two-column and newspaper pages, through `run`, `evaluate` and `report`. It checks the recorded versions, CPU quota and model checksums, the geometry and reading order, and a character error rate below 5 % on the text pages.
