# Tesseract engine (CPU baseline)

Reference engine adapter ([#11](https://github.com/michal-hradis/ocr_benchmarking/issues/11)): classic Tesseract OCR in a CPU Docker image built on the shared CPU base ([DOCKER.md](../../docs/DOCKER.md)), following the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol). Named presets with the `tessdata_best` models, German Fraktur and other page segmentation modes ([#41](https://github.com/michal-hradis/ocr_benchmarking/issues/41)) are described in [Presets](#presets).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-tesseract:5.3.0-v2` from `ocrbench/base-cpu:py3.12-v1` |
| `tessdata.sha256` | Checksums of the installed models; the build fails if they differ |
| `tessdata_best.sha256` | Checksums of the opt-in `tessdata_best` models in `/cache`, checked by the adapter (also in the image) |
| `fetch_tessdata.py` | Downloads the pinned `tessdata_best` models into the cache (`make tesseract-best`) |
| `tesseract_engine.py` | The adapter (Python standard library only) |
| `engine.toml` | Engine configuration of the default preset `fast-psm3` for `ocrbench run start` |
| `presets/*.toml` | Engine configurations of the `tessdata_best` presets ([Presets](#presets)) |

## Versions

- Tesseract **5.3.0** (Leptonica 1.82.0) from the Debian bookworm package `tesseract-ocr=5.3.0-2`.
- Models in the image: Debian `tesseract-ocr-{eng,ces,deu,osd}=1:4.1.0-2`, which are the **tessdata_fast 4.1.0** LSTM models (integer, no legacy engine), checked against `tessdata.sha256`.
- Models in the cache (opt-in, best presets only): **tessdata_best** `ces`, `deu`, `deu_latf` and `eng` from [tesseract-ocr/tessdata_best](https://github.com/tesseract-ocr/tessdata_best) at commit `e12c65a915945e4c28e237a9b52bc4a8f39a0cec` (2024-03-09), 48 MB in total, checked against `tessdata_best.sha256`. They are float LSTM models (OEM 1; no legacy engine). License: Apache 2.0, like Tesseract and the Debian models. `deu_latf` (German in Latin script, Fraktur) is the former `frk` model, renamed in that commit. It is not in Debian bookworm's packages.

The packages are pinned by version, not by a snapshot of the Debian archive. If Debian ever drops these versions, the build fails rather than silently changing them; adding another language or version means editing the `Dockerfile` and `tessdata.sha256` (or `tessdata_best.sha256` and `BEST_REVISION` in the adapter) and bumping the image tag (`-v3`) and `engine_version`. Image `-v2` adds the `tessdata` parameter and `tessdata_best.sha256`; with the default `fast-psm3` preset its predictions on dev-01 are byte-identical to `-v1`.

## Build, run and evaluate

```sh
make docker-cpu                     # shared CPU base, once
make docker-tesseract               # this image (Debian packages, tens of MB downloaded)
uv run ocrbench run start --engine engines/tesseract/engine.toml --dataset datasets/dev-01.json
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/tesseract --inspect
make tesseract-best                 # opt-in, best presets: download tessdata_best (48 MB) into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/tesseract/presets/best-psm3.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
make tesseract-smoke                # opt-in: build the image, run all presets on synthetic samples
```

Inference runs offline (`--network none`). The default preset's models are in the image, so it needs no `--cache`. The best presets read their models from `/cache` and need `--cache` with the directory `make tesseract-best` filled (`OCRBENCH_CACHE`, default `~/.cache/ocrbench`). The download is never automatic.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `lang` | `eng` | Tesseract `-l`: installed traineddata joined by `+` (`ces`, `deu`, `eng`). The shipped configuration uses `ces+deu+eng` |
| `psm` | `3` | Page segmentation mode (`--psm`, 0–13); 3 is fully automatic, 6 a single uniform block of text |
| `oem` | `1` | OCR engine mode (`--oem`); only 1 (LSTM) works with the fast and best models |
| `tessdata` | `fast` | Model variant: `fast` uses the image's tessdata_fast; `best` uses tessdata_best from `/cache/models/tesseract/tessdata_best/<commit>/` (`--tessdata-dir`), and `lang` must name models in `tessdata_best.sha256` |
| `dpi` | none | `--dpi` when the image has no reliable resolution; otherwise Tesseract estimates it |

With `tessdata = "best"`, the adapter checks the SHA-256 of every model named in `lang` before the first sample. An unregistered, missing or modified model makes every sample fail with `engine-error`. Tesseract's output configurations (`txt`, `hocr`, `tsv`) are then passed by their path in the image's tessdata, because the cache directory holds only models.

The sample `language` from the manifest is **not** used; the language is part of the frozen engine configuration. Thread count is set by `OMP_THREAD_LIMIT` in `[env]` (1 by default, matching `cpus = 1`).

## Presets

A preset is a frozen engine configuration; its `name` identifies it as a separate system in reports. All presets use the same image, OEM 1 (LSTM), one CPU with `OMP_THREAD_LIMIT=1`, 2 GB of memory and the same outputs. The checkpoint variant (`tessdata`) and the page segmentation mode (`psm`) are separate parameters, and the presets are chosen so that each comparison changes only one of them, instead of covering every combination:

| Preset (file) | `name` | `tessdata` | `lang` | `psm` | Purpose |
| --- | --- | --- | --- | ---: | --- |
| `fast-psm3` (`engine.toml`) | `tesseract` | fast | `ces+deu+eng` | 3 | Default page baseline (#11), unchanged |
| `best-psm3` (`presets/best-psm3.toml`) | `tesseract-best-psm3` | best | `ces+deu+eng` | 3 | Accuracy models; differs from `fast-psm3` only in the models |
| `best-psm11` (`presets/best-psm11.toml`) | `tesseract-best-psm11` | best | `ces+deu+eng` | 11 | Sparse text (as much text as possible, no layout order); differs from `best-psm3` only in the segmentation |
| `best-deu-latf-psm3` (`presets/best-deu-latf-psm3.toml`) | `tesseract-best-deu-latf-psm3` | best | `deu_latf` | 3 | Historical German Fraktur; differs from `best-psm3` only in the language |
| `best-psm6` (`presets/best-psm6.toml`) | `tesseract-best-psm6` | best | `ces+deu+eng` | 6 | Paragraph crops (`unit: paragraph`): one uniform block of text. Not meant for full pages |

The engine name `tesseract` of the default preset is kept for continuity with earlier runs. A new preset is a new TOML file with its own `name`; a changed setting is a new preset, not an edit of an existing one.

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

## Measured on dev-01

The four full-page presets on the seven pages of dev-01 (runs `tesseract-{fast-psm3,best-psm3,best-psm11,best-deu-latf-psm3}-dev01`, not in Git; image `sha256:c65d9732…`, AMD Ryzen 5 2600X, the four runs in parallel with one CPU each). `best-psm6` was not run, because dev-01 has no paragraph samples; it is covered by the synthetic paragraph in `make tesseract-smoke`. All 28 samples were `success`. Rates are micro averages over the pages; fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)).

| Preset | CER | WER | fCER | fWER | BoW F1 | Engine time (7 pages) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fast-psm3` | 48.2 % | 62.0 % | 27.1 % | 40.9 % | 81.7 | 123 s |
| `best-psm3` | 48.2 % | 62.3 % | 27.4 % | 41.1 % | 81.8 | 239 s |
| `best-psm11` | 72.1 % | 91.3 % | 23.5 % | 37.7 % | 81.9 | 252 s |
| `best-deu-latf-psm3` | 54.6 % | 84.0 % | 35.9 % | 74.7 % | 39.6 | 141 s |

CER per page (fCER where it differs):

| Page | Content | `fast-psm3` | `best-psm3` | `best-psm11` | `best-deu-latf-psm3` |
| --- | --- | ---: | ---: | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 83.1 (61.5) | 83.6 (62.8) | 83.0 (30.3) | 85.1 (67.3) |
| `CkXII61-0017` | Glagolitic print | 83.1 | 81.0 | 81.6 | 85.2 |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 33.0 (6.0) | 32.9 (5.9) | 81.3 (20.5) | 44.9 (20.1) |
| `dta-346672…` | Latin, Antiqua and italic | 23.1 (17.4) | 21.6 (16.0) | 22.6 | 23.1 (17.3) |
| `dta-laube…` | German Fraktur, 1833 | 12.5 | 11.3 | 11.6 | **7.2** |
| `dta-martzi…` | German Fraktur, 1692 | 29.3 | 30.0 | 33.7 | **23.6** |
| `dta-steinen…` | German Antiqua, 1894 | 1.1 | 1.3 | 2.1 | 2.1 |

Observations:

- **fast vs. best:** on dev-01 the best models are not more accurate overall (48.2 % CER either way; small gains on some pages, losses on others) and are about twice as slow.
- **psm 3 vs. psm 11:** psm 11 drops the layout order, so the strict CER rises sharply (72 %) on the newspapers. Its order-independent fCER is lower overall (23.5 %) because it finds more of the text on the first newspaper (fCER 30 % vs. 63 %), but it is worse on the second newspaper and on the historical pages.
- **deu_latf:** better on both Fraktur pages (CER 12.5 → 7.2 % and 29.3 → 23.6 % against `fast-psm3`). It has no Czech diacritics, so it is much worse on the Czech newspapers; it is a preset for Fraktur collections, not for mixed data.
- The Glagolitic page has no matching model in any preset (CER over 80 %).
- **Speed:** `deu_latf` alone is faster than three languages; `fast-psm3` took the same time as the earlier `-v1` run (123 s vs. 119 s).

**Reproducibility.** A second `best-psm3` run gave byte-identical `prediction.json` files on all seven pages, and the `fast-psm3` predictions with image `-v2` are byte-identical to those of the earlier `-v1` run.

## Tests

`tests/test_tesseract.py` checks the hOCR mapping on a committed hOCR file (`tests/fixtures/tesseract/page.hocr`, Tesseract 5.3.0 output for a synthetic page), parameter validation, and the engine protocol through the runner with a fake `tesseract` executable, all offline in `make test`. They also check the presets (valid configurations; each differs from its pair only in the models, the segmentation or the language), the `best` command line, the checksum checks of the cached models (unregistered, missing, modified) and `fetch_tessdata.py` with a fake download, and a best preset through the runner with fake models in a temporary cache. `make tesseract-smoke` builds the image and runs real Tesseract through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page (an empty `success`), plus a run with a missing language that must fail explicitly. It then runs every best preset on the same samples with the models from `make tesseract-best` (it fails if they are missing) and requires a CER below 10 % for `best-psm3` on the page and `best-psm6` on the paragraph.
