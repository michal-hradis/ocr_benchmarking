# PaddleOCR engine (PP-OCRv6, GPU)

Modular detector/recognizer baseline ([#12](https://github.com/michal-hradis/ocr_benchmarking/issues/12)): the PaddleOCR general OCR pipeline (text line detection followed by line recognition) with the PP-OCRv6 medium models, in a GPU Docker image on the shared Paddle base ([DOCKER.md](../../docs/DOCKER.md)), following the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-paddleocr:3.7.0-v1` from `ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1` |
| `requirements.in`, `requirements.txt` | PaddleOCR and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files; the build fails if they differ |
| `paddleocr_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

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
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/paddleocr --inspect
make paddleocr-smoke                # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. The image is a CUDA build: without a visible GPU every sample fails with `engine-error` and the reason (for example `libcuda.so.1` missing); there is no CPU fallback (`PADDLE_PDX_DISABLE_DEVICE_FALLBACK`).

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the pipeline defaults, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `gpu:0` | Paddle device, `gpu:<n>` |
| `text_det_limit_side_len`, `text_det_limit_type` | 64, `min` | Detection input scaling: the shorter (`min`) or longer (`max`) side is scaled to at least / at most this many pixels; the pipeline also caps the longer side at 4000 |
| `text_det_thresh`, `text_det_box_thresh`, `text_det_unclip_ratio` | 0.3, 0.6, 1.5 | DB detector pixel threshold, box score threshold and box expansion |
| `text_rec_score_thresh` | 0.0 | Lines with a lower recognition score are dropped |
| `text_recognition_batch_size` | 6 | Lines recognized per batch |

Fixed: document orientation classification and unwarping are **off**, because they would transform the image so that positions no longer refer to the input; the text line orientation classifier is off because its model is not in the image. The sample `language` is not used; the recognition model is multilingual.

## Outputs

For every sample the adapter decodes the image as stored (EXIF orientation not applied, as in [PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)), runs the pipeline and writes into the sample directory:

| File | Content |
| --- | --- |
| `paddleocr.json` | PaddleOCR's native result (`res.json`: detected and recognized polygons, texts, scores, detection settings), unchanged |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output |
| `result.json` | Status, processing time and, in `extra`, the effective detection settings, the PaddleOCR/PaddleX/Paddle/CUDA/cuDNN versions, the GPU name, the model loading time, the models and the parameters |

**Mapping.** Every recognized line (`rec_texts`, `rec_polys`, `rec_scores`) with non-blank text becomes a line with its detected quadrilateral as polygon (pixels of the input image: the detector maps its resized input back) and its recognition score (0–1) as confidence. Lines have no baseline. **Grouping:** the basic PP-OCR pipeline detects lines only, with no blocks, paragraphs or reading order, so all lines go into a single region `lines` of type `text` without a polygon (`extra.grouping: "all-lines"`); this is not a claim of layout structure. Lines are in PaddleOCR's order: top to bottom, left to right within about 10 pixels, so lines of side-by-side columns interleave. Strict CER/WER therefore suffer on multi-column pages; the flexible metrics are order-independent ([EVALUATION.md](../../docs/EVALUATION.md)). Layout analysis (PP-StructureV3) is a possible follow-up. The page text is the lines joined by newlines.

**Status.** An image that cannot be decoded or an error in the pipeline (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. An unusable GPU or invalid parameters fail every sample. A page on which nothing is detected is a **genuine empty result**: `success` with empty text and no regions.

## Tests

`tests/test_paddleocr.py` checks the mapping on a committed native result (`tests/fixtures/paddleocr/page.json`, PaddleOCR 3.7.0 output for a synthetic page), parameter validation, the pipeline arguments, and the engine protocol through the runner with fake `paddle`, `paddleocr` and `cv2` modules (`tests/fixtures/paddleocr/fake`), all offline in `make test`. `make paddleocr-smoke` builds the image and runs real PP-OCRv6 on the GPU through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page (an empty `success`), plus a run with a GPU that is not visible, which must fail explicitly.
