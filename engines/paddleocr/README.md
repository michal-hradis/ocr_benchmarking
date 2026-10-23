# PaddleOCR engine (PP-OCRv6, GPU)

Modular detector/recognizer baseline ([#12](https://github.com/michal-hradis/ocr_benchmarking/issues/12), detector input size presets [#42](https://github.com/michal-hradis/ocr_benchmarking/issues/42)): the PaddleOCR general OCR pipeline (text line detection followed by line recognition) with the PP-OCRv6 medium models, in a GPU Docker image on the shared Paddle base ([DOCKER.md](../../docs/DOCKER.md)), following the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-paddleocr:3.7.0-v2` from `ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1` |
| `requirements.in`, `requirements.txt` | PaddleOCR and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files; the build fails if they differ |
| `paddleocr_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`, preset `det4000` |
| `presets/det8000.toml` | Preset `det8000`, full-resolution detection of pages up to 8000 pixels ([Presets](#presets)) |

## Versions

- PaddleOCR **3.7.0** (PaddleX 3.7.2) on PaddlePaddle **3.4.0** for CUDA 13.0 (the `base-cu130-paddle` family; at run time Paddle reports cuDNN 9.13). It needs an NVIDIA driver for CUDA 13.0 (≥ 580) and has been run on an RTX 4090.
- Models: [`PaddlePaddle/PP-OCRv6_medium_det`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det) at commit `8e0f56fb2ef86b461d99cfc7ac5c137738985f61` and [`PaddlePaddle/PP-OCRv6_medium_rec`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec) at commit `e5a92bcbc5cc1b494628e458d267778f0704fd7c`, Apache-2.0. PP-OCRv6 medium is PaddleOCR 3.7's default; one recognition model covers Chinese, English, Japanese and 46 Latin-script languages, including Czech and German.
- The models (133 MB) are downloaded **when the image is built** and checked against `models.sha256`, so inference needs no `--cache` and no network. This is the "small weights" case of the image conventions: they are a few percent of the image.

The Ubuntu packages (`libgomp1` in the base, `libgl1` and `libglib2.0-0t64` for OpenCV here) are pinned by version, not by an archive snapshot. If Ubuntu removes a version, the build fails rather than silently changing it; then bump the pin and the image tag (`-v2`) and `engine_version`.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-paddle            # shared Paddle base, once (about 6 GB)
make docker-paddleocr               # this image (PaddleOCR and the models)
uv run ocrbench run start --engine engines/paddleocr/engine.toml --dataset datasets/dev-01.json
uv run ocrbench run start --engine engines/paddleocr/presets/det8000.toml --dataset datasets/dev-01.json
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/paddleocr --inspect
make paddleocr-smoke                # opt-in: build the image, run both presets on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. The image is a CUDA build: without a visible GPU every sample fails with `engine-error` and the reason (for example `libcuda.so.1` missing); there is no CPU fallback (`PADDLE_PDX_DISABLE_DEVICE_FALLBACK`).

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the pipeline defaults, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `gpu:0` | Paddle device, `gpu:<n>` |
| `text_det_limit_side_len`, `text_det_limit_type` | 64, `min` | Detection input scaling: the shorter (`min`) or longer (`max`) side is scaled to at least / at most this many pixels |
| `text_det_max_side_limit` | 4000 | Then the detection input is scaled down so that its longer side is at most this many pixels, an integer `>= 32` (the input is rounded to multiples of 32). PaddleOCR() has no argument for it, so the adapter sets `SubModules.TextDetection.max_side_limit` in the PaddleX pipeline configuration and fails the sample if the native result reports a different value |
| `text_det_thresh`, `text_det_box_thresh` | 0.3, 0.6 | DB detector pixel threshold and box score threshold, `0 < value <= 1` |
| `text_det_unclip_ratio` | 1.5 | DB box expansion ratio, a finite number `> 0` |
| `text_rec_score_thresh` | 0.0 | Lines with a lower recognition score are dropped, `0 <= value <= 1` (a higher value would silently drop every line, so it is rejected) |
| `text_recognition_batch_size` | 6 | Lines recognized per batch |

Fixed: document orientation classification and unwarping are **off**, because they would transform the image so that positions no longer refer to the input; the text line orientation classifier is off because its model is not in the image. The sample `language` is not used; the recognition model is multilingual.

## Presets

A preset is a frozen engine configuration; its `name` identifies it as a separate system in reports. Both presets use the same image, models, recognition and resources, and differ **only in the detector input size** (`text_det_max_side_limit`). Recognition always runs on line crops of the original image, so the preset changes which lines are detected and how they are cut, not the recognizer's input resolution.

| Preset (file) | `name` | `text_det_max_side_limit` | Detector input for a 4194 × 6192 page | Purpose |
| --- | --- | ---: | --- | --- |
| `det4000` (`engine.toml`) | `paddleocr` | 4000 | 2720 × 4000 (scaled 0.65) | PaddleOCR's default, unchanged from #12 |
| `det8000` (`presets/det8000.toml`) | `paddleocr-det8000` | 8000 | 4192 × 6208 (full resolution, rounded to multiples of 32) | Full-resolution detection of pages up to 8000 pixels, such as ~6000-pixel newspaper scans |

Pages whose longer side is at most 4000 pixels give identical results with both presets. The engine name `paddleocr` of the default preset is kept for continuity with earlier runs; its predictions on dev-01 are byte-identical to those of image `-v1`. A new preset is a new TOML file with its own `name`; a changed setting is a new preset, not an edit of an existing one.

**GPU memory.** Detection memory grows with the detector input area. Measured peak Paddle allocation on an RTX 4090 for the 4194 × 6192 page: 2.9 GB allocated (5.4 GB reserved) with `det4000`, 6.7 GB (9.9 GB reserved) with `det8000`. An 8000 × 8000 page would need roughly 2.5 times as much as the 6192-pixel page. Running out of GPU memory fails only that sample (see Status); the next sample runs normally.

**Reading order and layout.** Neither preset changes the grouping: the basic PP-OCR pipeline has no layout analysis, and the adapter does not invent one (see Mapping). Layout analysis is the separate [PP-StructureV3 engine](../ppstructurev3/README.md), not a preset of this adapter. The evaluator's ordering is unchanged; compare the strict (order-dependent) CER/WER with the flexible fCER/fWER to separate reading order from recognition ([Measured on dev-01](#measured-on-dev-01)).

## Outputs

For every sample the adapter decodes the image as stored (EXIF orientation not applied, as in [PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)), runs the pipeline and writes into the sample directory:

| File | Content |
| --- | --- |
| `paddleocr.json` | PaddleOCR's native result (`res.json`: detected and recognized polygons, texts, scores, detection settings), unchanged |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output |
| `result.json` | Status, processing time and, in `extra`, the effective detection settings, the PaddleOCR/PaddleX/Paddle/CUDA/cuDNN versions, the GPU name, the model loading time, the models and the parameters |

**Mapping.** Every recognized line (`rec_texts`, `rec_polys`, `rec_scores`) with non-blank text becomes a line with its detected quadrilateral as polygon (pixels of the input image: the detector maps its resized input back) and its recognition score (0–1) as confidence. Lines have no baseline. **Grouping:** the basic PP-OCR pipeline detects lines only, with no blocks, paragraphs or reading order, so all lines go into a single region `lines` of type `text` without a polygon (`extra.grouping: "all-lines"`); this is not a claim of layout structure. Lines are in PaddleOCR's order: top to bottom, left to right within about 10 pixels, so lines of side-by-side columns interleave. Strict CER/WER therefore suffer on multi-column pages; the flexible metrics are order-independent ([EVALUATION.md](../../docs/EVALUATION.md)). Layout analysis is the separate [PP-StructureV3 engine](../ppstructurev3/README.md). The page text is the lines joined by newlines.

**Status.** An image that cannot be decoded or an error in the pipeline fails only that sample (`status: failed`, runner `engine-error`) and the container continues. CUDA out of memory is such an error: Paddle raises `MemoryError` with a long C++ traceback, of which the message keeps only Paddle's summary (`MemoryError: ResourceExhaustedError: Out of memory error on GPU 0. Cannot allocate …`); the pipeline recovers for the next sample (checked with a 19000 × 19000 image and `text_det_max_side_limit` 20000 between two pages). A result whose `text_det_params.max_side_limit` differs from the requested `text_det_max_side_limit` fails the sample. An unusable GPU or invalid parameters fail every sample. A page on which nothing is detected is a **genuine empty result**: `success` with empty text and no regions.

## Measured on dev-01

Both presets on the seven pages of dev-01 (runs `paddleocr-det4000-dev01` and `paddleocr-det8000-dev01`, not in Git; image `sha256:80c7e0d0…`, RTX 4090). All 14 samples were `success`. Rates are micro averages over the pages; fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)). Engine time is the sum of the adapter's per-sample times.

| Preset | CER | WER | fCER | fWER | BoW F1 | Engine time (7 pages) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `det4000` | 72.0 % | 89.2 % | 12.1 % | 25.1 % | 85.2 | 11 s |
| `det8000` | 72.0 % | 89.2 % | 14.0 % | 27.1 % | 84.9 | 12 s |

CER per page (fCER in parentheses where it differs). Only the two newspapers exceed 4000 pixels; the other five pages are byte-identical between the presets.

| Page | Content | Lines `det4000` / `det8000` | `det4000` | `det8000` |
| --- | --- | --- | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 401 / 402 | 82.2 (12.4) | 82.2 (13.2) |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 468 / 458 | 81.3 (9.8) | 81.4 (13.2) |
| `CkXII61-0017` | Glagolitic print | 19 | 82.3 | 82.3 |
| `dta-346672…` | Latin, Antiqua and italic | 31 | 12.0 | 12.0 |
| `dta-laube…` | German Fraktur, 1833 | 25 | 4.0 | 4.0 |
| `dta-martzi…` | German Fraktur, 1692 | 31 | 13.1 | 13.1 |
| `dta-steinen…` | German Antiqua, 1894 | 39 | 17.5 | 17.5 |

Observations:

- **Reading order, not recognition, dominates the strict rates on the newspapers:** CER above 80 % against fCER of 10–13 %, with both presets. The lines of side-by-side columns interleave (see Mapping), and the detector size does not change that.
- **Full-resolution detection does not help on dev-01.** Its text lines are about 36 pixels high, so the 0.65–0.76 downscaling of `det4000` still detects them well. With `det8000` the detector merged some lines of neighbouring columns across narrow gutters, split large headline letters and cut other lines slightly differently (56 of 468 lines differ in text on `bd497a62…`, for example `ROCNIK1945` / `ROCNIK 1945` and `MORAVÉ` / `MORAVE`). The recognizer is the same; the differences come from the line boxes it is given. The fCER rises from 9.8 to 13.2 % on `bd497a62…` (substitutions 136 → 849 in the flexible alignment) and from 12.4 to 13.2 % on `3e6309d0…`.
- `det8000` is a candidate for scans with small text, where the 4000-pixel cap shrinks lines below what the detector resolves; dev-01 has no such page, so this is not measured here.

## Tests

`tests/test_paddleocr.py` checks the mapping on committed native results (`tests/fixtures/paddleocr/`: `page.json`, PaddleOCR 3.7.0 output for a synthetic page, and `page-x6-limit4000.json`/`page-x6-limit8000.json` for the same page scaled 6× to 5400 × 2520 and detected downscaled or at full resolution: the polygons of both must cover the rendered text lines in input-image pixels), parameter validation, the pipeline arguments and PaddleX configuration, the presets (they differ from `engine.toml` only in name and `text_det_max_side_limit`), and the engine protocol through the runner with fake `paddle`, `paddlex`, `paddleocr` and `cv2` modules (`tests/fixtures/paddleocr/fake`), including a simulated CUDA out-of-memory error between two successful samples and a pipeline that ignores the limit, all offline in `make test`. `make paddleocr-smoke` builds the image and runs real PP-OCRv6 with both presets on the GPU through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, the same page scaled to 5400 × 2520 (its polygons must cover the text in input-image pixels, CER < 10 %), a paragraph crop and a blank page (an empty `success`), plus a run with a GPU that is not visible, which must fail explicitly. The out-of-memory behaviour on a real GPU was checked once by hand (Status) and is not part of the smoke test, because it depends on the GPU's memory.
