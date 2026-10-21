# Tesseract engine (CPU baseline)

Reference engine adapter ([#11](https://github.com/michal-hradis/ocr_benchmarking/issues/11)): classic Tesseract OCR in a CPU Docker image built on the shared CPU base ([DOCKER.md](../../docs/DOCKER.md)), following the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-tesseract:5.3.0-v1` from `ocrbench/base-cpu:py3.12-v1` |
| `tessdata.sha256` | Checksums of the installed models; the build fails if they differ |
| `tesseract_engine.py` | The adapter (Python standard library only) |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions

- Tesseract **5.3.0** (Leptonica 1.82.0) from the Debian bookworm package `tesseract-ocr=5.3.0-2`.
- Models: Debian `tesseract-ocr-{eng,ces,deu,osd}=1:4.1.0-2`, which are the **tessdata_fast 4.1.0** LSTM models (integer, no legacy engine), checked against `tessdata.sha256`.

The packages are pinned by version, not by a snapshot of the Debian archive. If Debian ever drops these versions, the build fails rather than silently changing them; adding another language or version means editing the `Dockerfile` and `tessdata.sha256` and bumping the image tag (`-v2`) and `engine_version`.

## Build, run and evaluate

```sh
make docker-cpu                     # shared CPU base, once
make docker-tesseract               # this image (Debian packages, tens of MB downloaded)
uv run ocrbench run start --engine engines/tesseract/engine.toml --dataset datasets/dev-01.json
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/tesseract --inspect
make tesseract-smoke                # opt-in: build the image, run it on synthetic samples
```

Inference runs offline (`--network none`); the models are in the image, so no `--cache` is needed.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `lang` | `eng` | Tesseract `-l`: installed traineddata joined by `+` (`ces`, `deu`, `eng`). The shipped configuration uses `ces+deu+eng` |
| `psm` | `3` | Page segmentation mode (`--psm`, 0–13); 3 is fully automatic, 6 a single uniform block of text |
| `oem` | `1` | OCR engine mode (`--oem`); only 1 (LSTM) works with the fast models |
| `dpi` | none | `--dpi` when the image has no reliable resolution; otherwise Tesseract estimates it |

The sample `language` from the manifest is **not** used; the language is part of the frozen engine configuration. Thread count is set by `OMP_THREAD_LIMIT` in `[env]` (1 by default, matching `cpus = 1`).

## Outputs

For every sample the adapter runs `tesseract <image> page -l … --psm … --oem … txt hocr tsv` and writes into the sample directory:

| File | Content |
| --- | --- |
| `page.txt`, `page.hocr`, `page.tsv` | Tesseract's native outputs, unchanged |
| `stderr.txt` | Tesseract's diagnostics |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output |
| `result.json` | Status, the adapter's processing time and, in `extra`, the Tesseract version, parameters and command arguments |

**Mapping from hOCR.** Every Tesseract paragraph (`ocr_par`) with text becomes a region of type `text` (`native_type: paragraph`, its block ID in `extra.block`) with the paragraph's bounding box. Every text line (`ocr_line`, or `ocr_header`/`ocr_caption`/`ocr_textfloat`, whose class without `ocr_` becomes the line `type`) becomes a line with its words joined by single spaces, its bounding box as polygon, the hOCR baseline as a two-point polyline and the mean word confidence (0–100). Coordinates are pixels of the input image, as Tesseract reports them. Lines and paragraphs without text are dropped. There is no explicit reading order: regions are in Tesseract's output order. The page text is the line text, lines joined by a newline and paragraphs by a blank line, so it agrees with the structure; `page.txt` may differ in whitespace.

**Status.** A Tesseract error (exit status other than 0, for example an unreadable image or a missing language) or missing or unreadable hOCR is reported as `status: failed` with the end of Tesseract's error output as message; the runner records it as `failure` / `engine-error`. A page on which Tesseract finds no text is a **genuine empty result**: `success` with empty text and no regions.

## Tests

`tests/test_tesseract.py` checks the hOCR mapping on a committed hOCR file (`tests/fixtures/tesseract/page.hocr`, Tesseract 5.3.0 output for a synthetic page), parameter validation, and the engine protocol through the runner with a fake `tesseract` executable, all offline in `make test`. `make tesseract-smoke` builds the image and runs real Tesseract through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page (an empty `success`), plus a run with a missing language that must fail explicitly.
