# PERO OCR engine (offline page pipeline, GPU)

This adapter runs [PERO OCR](https://github.com/DCGM/pero-ocr) as a complete offline page pipeline ([#34](https://github.com/michal-hradis/ocr_benchmarking/issues/34): adapter [#39](https://github.com/michal-hradis/ocr_benchmarking/issues/39), presets [#40](https://github.com/michal-hradis/ocr_benchmarking/issues/40)). The pipeline runs layout analysis (text regions and text lines with baselines), region reading order, line cropping and transformer line OCR, all inside a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)). It follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol). The models are **user-supplied PERO model bundles**. They are not downloaded or redistributed: they are copied into the `/cache` volume and checked against `bundles.sha256`.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-pero:0.7.0-81b496e-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | The runtime dependencies of pero-ocr's inference path, hash-locked against the base (`make docker-lock`) |
| `bundles.sha256` | The registered bundles: every file of every bundle with its SHA-256 |
| `install_bundles.py` | Copies the registered bundles from a host directory into a cache directory (`make pero-bundles`) |
| `pero_engine.py` | The adapter |
| `engine.toml` | Engine configuration of the universal preset for `ocrbench run start` |

## Versions

- **pero-ocr** (BSD-3-Clause) at commit [`81b496e`](https://github.com/DCGM/pero-ocr/tree/81b496e3084e294d2945e139fe60cf291010a83f) of the `develop` branch (2026-05-15, package version 0.7.0). It is installed from GitHub's source archive of that commit, and the build fails unless the archive matches its SHA-256.
  - **Why `develop`.** The supplied OCR bundle is an *exported* (TorchScript) transformer (`"exported": true` in `ocr_engine.json`). The PyPI release 0.7.0 (2024) and the `master` branch cannot load it: master added support for exported transformers and reverted it a day later. `develop` has that support. It also has beam search, and its PAGE XML/ALTO writers record confidences.
  - **Dependencies.** pero-ocr is installed with `--no-deps`. Its declared dependencies pin `numpy~=1.24.2`, which has no Python 3.12 wheels, and they include packages that the inference path never imports: `brnolm` (language-model decoding), `safe-gpu`, `lmdb`, `imgaug`, `pyamg`, `ultralytics` (YOLO layout, imported only when configured), `music21` and PDF export. `requirements.txt` locks what inference imports: numpy **1.26.4** (the closest 1.x release with Python 3.12 wheels), OpenCV 4.11 headless (the base has no libGL), SciPy, Shapely, lxml, scikit-image, scikit-learn, Numba and arabic-reshaper. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580). torch 2.14 still loads TorchScript models but warns that `torch.jit.load` is deprecated.
- **Bundles** are described in the next section. They are not in the image.

## Model bundles and presets

A PERO **bundle** is a directory with a `config.ini` entry point and the files it references. The INI is read unmodified with pero-ocr's own `PageParser`, and relative paths resolve against the bundle directory, as in pero-ocr's `parse_folder.py`. A **preset** is a frozen `engine.toml` that names one layout bundle and one OCR bundle. The adapter builds one `PageParser` from each bundle and runs them on the same page: layout first, then OCR. Each preset is a separate system in reports: its `name` and `model` identify it.

### The universal preset (`engine.toml`, `pero-universal-2025-03-19`)

The maintainer supplied this preset's two bundles. They were not downloaded from a public source, and **no public identifier, download URL or license is stated** in the bundles. Their files therefore stay out of Git and out of the image. The table below is inferred from the INI files and `ocr_engine.json` as supplied:

| | `layout_model` | `universal.transformer.2025-03-19` |
| --- | --- | --- |
| Purpose | Layout analysis: text regions, text lines (baseline, heights, polygon), region reading order | Line cropping and line OCR of the regions and lines found by the layout bundle |
| Entry point | `config.ini`: `RUN_LAYOUT_PARSER = yes`, every other stage `no` | `config.ini`: `RUN_LINE_CROPPER = yes`, `RUN_OCR = yes`, layout and decoder `no` |
| Model | `ParseNet_069500.pt` (127 MB, TorchScript ParseNet, iteration 69,500) | `checkpoint_1420000.pt` (223 MB, TorchScript transformer `michal_universal`, iteration 1,420,000), defined by `ocr_engine.json` |
| Settings | `[LAYOUT_PARSER_1]` `LAYOUT_CNN`: lines and regions, no line merging or height adjustment, `MAX_MEGAPIXELS = 5` with adaptive downsampling (`DOWNSAMPLE = 3`), `DETECTION_THRESHOLD = 0.4`, `LINE_END_WEIGHT = 0.6`, `VERTICAL_LINE_CONNECTION_RANGE = 3`, no smoothing. `[LAYOUT_PARSER_2]` `REGION_SORTER_SMART` (reading order of regions) | `[LINE_CROPPER]` 48-pixel lines, `LINE_SCALE = 1.15`, `INTERP = 2`. `[OCR]` `pytorch_ocr-transformer`, greedy decoding (no `BEAM_SIZE`), line crops wider than 1,800 pixels are recognized in overlapping parts (`max_line_width`) |
| Transcription | — | 511 output symbols: Latin with Czech, German, Polish, French and other diacritics, Cyrillic including Church Slavonic letters, Greek, digits and punctuation, 5 combining marks, `ſ` and some medieval letters. There are also 24 private-use code points (U+101010–U+101027) whose meaning the bundle does not document; none appeared on dev-01. On the four DTA Fraktur pages of dev-01, the long s comes out as `s` |
| Device | GPU (`USE_CPU = no`; `GPU_FRACTION` is a TensorFlow-era setting that the torch code ignores) | GPU only: on the CPU pero-ocr loads `<checkpoint>.cpu`, which the bundle does not contain |
| Version | `ParseNet_069500`, file dated 2024-12-07 | `2025-03-19` (directory name), checkpoint dated 2025-05-14 |

The SHA-256 of all five files is in `bundles.sha256`. It also serves as the bundle fingerprint, and every sample records it in `result.json`. **Overrides:** none of the INI settings is changed. The adapter supplies only what `parse_folder.py` would: the device (`device` parameter, `cuda:0`) and the bundle directory as the base for relative paths. Neither bundle configures language-model decoding (`RUN_DECODER = no`), so a "native decode" variant does not exist and is not benchmarked.

### Supplying bundles

```sh
make pero-bundles                                   # from dev_data/pero_ocr_models into ~/.cache/ocrbench
make pero-bundles PERO_BUNDLES=/path/to/bundles OCRBENCH_CACHE=/path/to/cache
```

This copies `<PERO_BUNDLES>/<bundle>/<file>` to `<OCRBENCH_CACHE>/models/pero/<bundle>/<file>` for every file listed in `bundles.sha256`, and checks each copy. Files that are already present and correct are kept, and nothing is downloaded. Runs need `--cache <OCRBENCH_CACHE>`. At start-up the adapter checks again that every registered file of both bundles matches, and it never writes into the bundle directories. The runner mounts `/cache` read-write for all engines; a read-only bundle mount would need a runner change and is not part of this adapter.

**Adding a bundle or preset.** Add the bundle's files to `bundles.sha256` as `<sha256>  <bundle>/<file>` and rebuild the image (the file is in the image). Then copy `engine.toml` to a new file with its own `name`, a `model` naming the bundles, and `layout_bundle`/`ocr_bundle`. Record its purpose and provenance in the table above. Keep the supplied INI unmodified; a changed setting is a new bundle with its own checksums.

**Checks.** Each of the following makes every sample fail with `engine-error`, before any image is processed:

- a bundle that is not in `bundles.sha256`;
- a missing file (`... missing file ... (run make pero-bundles)`), or a file that differs from its checksum;
- an INI that cannot be parsed or lacks `[PAGE_PARSER]`;
- a layout bundle that does not run layout analysis or that runs OCR or decoding; an OCR bundle that does not run cropping and OCR or that runs layout analysis or decoding;
- a `MODEL_PATH`, `OCR_JSON` or OCR checkpoint that lies outside the bundle or is not a registered file (for example `device = "cpu"` with this bundle, which would need a `.cpu` checkpoint).

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-pero                    # this image (no models)
make pero-bundles                   # the supplied bundles (350 MB) into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/pero/engine.toml \
    --dataset datasets/dev-01.json --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/pero --inspect
make pero-smoke                     # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` (only for bundles with `.cpu` checkpoints) |
| `layout_bundle` | `layout_model` | Bundle directory name under `/cache/models/pero/`, registered in `bundles.sha256` |
| `ocr_bundle` | `universal.transformer.2025-03-19` | The same, for the OCR bundle |

The sample `language` and other metadata are not used; the universal model has no language switch.

## Pipeline and outputs

For every sample the adapter reads the image as `parse_folder.py` does, with OpenCV as 8-bit BGR. It does not apply EXIF orientation (`IMREAD_IGNORE_ORIENTATION`), so coordinates refer to the image as stored ([PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)). It then creates an empty `PageLayout` of the image size and runs the layout and OCR `PageParser`s on it. All resizing is internal to PERO, and every coordinate it writes is in original-image pixels. The layout CNN runs at an adaptively chosen downsampling of at most 5 megapixels, and line crops are taken from the full-resolution image. This is how 4k × 6k newspaper pages are processed without tiling. PERO orders lines of the same height with random jitter, so the adapter seeds Python's `random` with 0 before every page. The sample directory then gets:

| File | Content |
| --- | --- |
| `page.xml` | PERO's own PAGE XML (2019-07-15 schema): regions in reading order with polygons and the `category` in `custom`, lines with polygons, baselines, heights and their `TextEquiv` with confidence |
| `alto.xml` | PERO's own ALTO (v2) with words and their positions, as `parse_folder.py --output-alto-path` writes it. If the ALTO export fails, the sample still succeeds, and the error is recorded as `alto_error` in `result.json` |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output |
| `result.json` | Status, processing time and, in `extra`: the numbers of PERO regions and lines and of predicted regions and lines, `gpu_peak_memory_mb` (`torch.cuda.max_memory_allocated`), the pero-ocr version and commit, the torch/CUDA versions, the GPU name, the model loading time, both bundles with their path and the SHA-256 of every file, and the parameters |

**Mapping.** The prediction is built from `page.xml`, PERO's own serialization, so a mapping fix can be re-applied to stored output:

- **Regions:** every PAGE `TextRegion` with at least one non-blank line becomes a region with its polygon. `extra.pero_id` is the PAGE ID. The universal layout CNN gives regions no PAGE `type`, so they are `text`. A PAGE type, if one appears, is kept as `native_type` and mapped to a [recommended type](../../docs/PREDICTIONS.md#region-types): `paragraph` → `text`, `heading`, `caption`, `header` → `page-header`, `footer` → `page-footer`, `page-number`, `footnote`/`footnote-continued` → `footnote`, `list-label` → `list-item`, anything else → `other`. Columns are not merged: each PERO region stays a region of its own.
- **Lines:** every `TextLine` with non-blank text keeps its text, polygon, baseline, its `TextEquiv` confidence and `extra.pero_id`. A polygon with fewer than 3 points, or a baseline with fewer than 2, is omitted rather than invented.
- **Reading order:** PERO's order. If the PAGE file has a `ReadingOrder`, that order is used, read as the benchmark's PAGE reader reads it: members of an ordered group by their integer `@index` (document order if any index is missing), nested groups in place, an unordered group in document order, and unreferenced regions after the referenced ones. Otherwise the file's region order is used, which is the order of PERO's region sorter. The universal preset writes no `ReadingOrder`. `reading_order` lists the regions in this order, and the lines of a region keep PERO's order (top to bottom).
- **Not in the prediction:** regions without text (none occur with this preset) and lines with blank text stay only in the native outputs.
- **Page text:** the lines of a region joined by newlines, regions joined by a blank line.

**Status.** An error during layout analysis or OCR (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`), and the container continues. An unusable GPU, invalid parameters, or a missing, modified or incompatible bundle fail every sample. A page on which PERO finds no lines is a **genuine empty result**: `success` with empty text.

## Measured on dev-01 (RTX 4090)

These are the results with the universal preset on the seven pages of dev-01 (`runs/pero-dev01`, not in Git). Engine time excludes model loading (7.8 s, once per container), and the GPU peak is `torch.cuda.max_memory_allocated`.

| Page | Size | Engine time | GPU peak | PERO regions / lines | CER | fCER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `3e6309d0…` (newspaper) | 4194 × 6192 | 13.9 s | 1,299 MB | 26 / 428 | 63.8 % | 0.5 % |
| `CkXII61-0017` | 1334 × 1878 | 0.8 s | 691 MB | 3 / 19 | 3.8 % | 3.8 % |
| `bd497a62…` (newspaper) | 3509 × 5239 | 20.0 s | 1,673 MB | 33 / 495 | 13.4 % | 0.6 % |
| `dta-346672…` | 1600 × 2182 | 1.0 s | 666 MB | 6 / 33 | 4.3 % | 4.3 % |
| `dta-laube…` | 1600 × 2382 | 0.8 s | 563 MB | 5 / 22 | 7.4 % | 7.4 % |
| `dta-martzi…` | 1600 × 2823 | 0.8 s | 666 MB | 1 / 30 | 5.9 % | 5.9 % |
| `dta-steinen…` | 1600 × 2325 | 3.3 s | 538 MB | 2 / 43 | 1.0 % | 1.0 % |

Over all pages, CER is 29.6 % and fCER 0.99 %. The text of the two newspaper pages is almost exact, but their strict CER is high because PERO's region order differs from the order of the GT regions. The flexible metrics ([EVALUATION.md](../../docs/EVALUATION.md)) do not depend on that order. Every line was recognized and no ALTO export failed.

**Reproducibility.** Three runs with the same image gave identical predictions (text, regions, coordinates and confidences) except for `extra.pero_id`: on the two newspaper pages, PERO numbered some native region IDs in `page.xml` differently between runs (for example `r029` and `r030` swapped).

**Blank page.** On the pure-white synthetic blank page, the universal layout model finds one faint line, and the OCR reads it as `1918` with confidence 0.17. This is the model's genuine output. It is a `success`, scored as produced, and is not filtered.

## Tests

`tests/test_pero.py` runs offline in `make test`. It checks:

- the PAGE XML mapping on a committed native result (`tests/fixtures/pero/page.xml` and `alto.xml`, real pero-ocr output with this preset for the synthetic page). The PAGE XML is read with the benchmark's own PAGE reader and must contain the same lines;
- reading order, region types and omitted elements on hand-written PAGE XML;
- parameter validation, the registered checksums and `install_bundles.py` (copying only registered files, keeping correct ones, rejecting a modified source);
- bundle verification (unregistered, missing and modified files) and the INI checks (invalid INI, wrong or forbidden stages, decoding, assets outside the bundle or unregistered, the missing CPU checkpoint), on synthetic bundles in a temporary cache;
- the engine protocol through the runner, with fake `torch`, `cv2` and `pero_ocr` modules (`tests/fixtures/pero/fake`).

`make pero-smoke` (after `make pero-bundles`) builds the image and runs real PERO OCR on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and the blank page. Two further runs, one without `--cache` and one with a GPU that is not visible, must fail explicitly.
