# PyLaia engine (CNN-RNN-CTC line recognition, GPU)

[PyLaia](https://doc.teklia.com/pylaia/) baseline ([#113](https://github.com/michal-hradis/ocr_benchmarking/issues/113)): a convolutional-recurrent line recognizer trained with CTC, which is a different recognition architecture from Kraken's or Calamari's. It recognizes lines found by a frozen page segmenter, **Kraken's default baseline segmentation, exactly as in the [Kraken engine](../kraken/README.md)**. With the same segmentation, the two engines differ only in how lines are cropped and recognized. The engine runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-pylaia:1.1.2-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1`; build context `engines/` (`make docker-pylaia`) |
| `pylaia-requirements.in`, `pylaia-requirements.txt` | PyLaia and its dependencies for the recognizer venv, hash-locked for Python 3.10 (`make docker-lock`) |
| `models.sha256` | Checksums of the PyLaia model files; the build fails if they differ |
| `pylaia_engine.py` | The adapter: segmentation, line crops, outputs (runs in the base venv with Kraken) |
| `pylaia_worker.py` | The PyLaia recognizer process (runs in the PyLaia venv) |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions and licenses

- **Two venvs in one image.** PyLaia **1.1.2** (MIT, the latest release) pins Python < 3.11, torch 1.13 and PyTorch Lightning 1.4.2. No Kraken release installs alongside those pins. The image therefore contains:
  - the base venv `/opt/venv` (Python 3.12, torch 2.14.1+cu130) with **Kraken 7.1.1**, installed from the Kraken engine's lock (`engines/kraken/requirements.txt`) for segmentation;
  - a separate venv `/opt/pylaia` with a uv-managed **Python 3.10.22**, **torch 1.13.1** (CUDA 11.7 libraries from PyPI; they run on an RTX 4090 with the CUDA 13 driver ≥ 580), torchvision 0.14.1 and torchaudio 0.13.1.

  The PyLaia venv is locked with `--exclude-newer 2024-10-17`, so its unpinned dependencies are the versions available on PyLaia 1.1.2's release date. Because `import laia` fails without a `git` executable, the image also installs a pinned Ubuntu `git` package.
- **Segmentation:** Kraken's default `blla.mlmodel` (Apache-2.0), shipped in the hash-locked `kraken` wheel. It finds regions of class `text` and line baselines with boundary polygons, and orders lines with Kraken's polygonal heuristic. It has no reading order model.
- **Recognition model in the image:** [`Teklia/pylaia-newseye-austrian`](https://huggingface.co/Teklia/pylaia-newseye-austrian) at revision `5361aa77db34d23cb23b27194c7c095f01f5ea54` (MIT; the files `model`, `weights.ckpt` (43 MB) and `syms.txt`, checked against `models.sha256` when the image is built). Teklia trained it on the [NewsEye / READ OCR dataset of 19th-century Austrian newspapers](https://zenodo.org/record/3387369): **printed German, mostly Fraktur**, 52,834 training lines with a fixed height of 128 pixels. The model card reports 1.82 % CER on its validation set without a language model.

### Checkpoint audit and its limits

Teklia publishes all of its public PyLaia checkpoints on Hugging Face under MIT: NewsEye Austrian, IAM (English), RIMES, POPP, Belfort and PELLET (French), Himanis and HOME-Alcar (medieval Latin/French), NorHand v1–v3 (Norwegian), Esposalles (Catalan) and CASIA-HWDB2 (Chinese). **None covers Czech.** NewsEye Austrian is the only printed-text model and the closest to the development set. That is why it is the default and only preset. Its 157 symbols, besides `<space>`, `<unk>` and the CTC blank, include German umlauts and `ß`. Of the Czech letters with diacritics it has only `á é ó ř ž É Ó Š Ž`. It has **no** `č ď ě í ň š ť ú ů ý` (nor their capitals except `Š Ž`), no long s `ſ` and no Glagolitic letters. On Czech pages, every word with a missing letter is necessarily wrong, so the CER/WER there measures charset coverage as much as recognition. No model was trained or tuned on the development pages.

**Handwriting.** PyLaia is a handwriting recognizer, but the development pages are all printed (Czech newspapers, Glagolitic, Latin and German prints). They are **out of domain for handwriting**, and scores on them say nothing about PyLaia's handwriting accuracy.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-pylaia                  # this image (Kraken, PyLaia and the NewsEye model)
uv run ocrbench run start --engine engines/pylaia/engine.toml --dataset datasets/dev-01.json
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/pylaia --inspect
make pylaia-smoke                   # opt-in: build the image, run it on synthetic samples on the GPU
```

The model is in the image, so inference needs no `--cache` and no network. The container runs with `--gpus all` (`[resources]` in `engine.toml`). Segmentation and recognition both use `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` both run on the CPU, which is slower and not the configuration recorded here.

## Pipeline

1. **Image.** Decoded as stored with Pillow (EXIF orientation not applied, [PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)); modes other than grayscale and RGB are converted to RGB.
2. **Segmentation.** This is the Kraken engine's code and model, including its [segmentation canvas](../kraken/README.md#parameters): an image less than `segmentation_min_height` pixels high is segmented on a taller canvas and the result moved back into image pixels, so paragraph crops (`unit: paragraph`) are segmented too. The adapter never reads PAGE XML or any other GT geometry, and it has no separate line-crop input mode.
3. **Line crops.** For each Kraken line, the adapter crops the axis-aligned bounding box of its boundary polygon (rounded outwards and clipped to the image) from the original image converted to grayscale. Pixels outside the polygon are not masked, and the line is not dewarped. A line whose polygon has fewer than 3 points or an empty box is not recognized (`crop: null`).
4. **Recognition.** PyLaia's `pylaia-htr-decode-ctc` path, called through PyLaia's own classes:
   - `ToImageTensor` inverts the crop and resizes it to the model's **fixed height of 128 pixels** with Lanczos, keeping the aspect ratio, then pads it to the model's minimum valid width.
   - Batches of `batch_size` crops are sorted by descending width and zero-padded (`PaddingCollater`), and run through the model.
   - Decoding is **greedy CTC** (`CTCGreedyDecoder`, temperature 1). There is no beam search and **no language model**: the NewsEye model ships none, and none is added.
   - The symbols are joined as `--decode.use_symbols --decode.convert_spaces --decode.join_string ""` prints them: `<space>` becomes a space, other symbols are kept (including a literal `<unk>`), and the line is stripped.
5. **Lines and page.** Each line keeps Kraken's boundary polygon and baseline, and its confidence is the mean of PyLaia's per-character probabilities (PyLaia's line confidence). Lines are grouped into regions and ordered exactly as in the [Kraken engine's mapping](../kraken/README.md#outputs): Kraken's reading order, one region per run of consecutive lines in the same Kraken region, page text with lines joined by newlines and regions by a blank line. Lines with blank text and regions without text stay only in the native outputs.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `recognition_model` | `newseye-austrian` | The PyLaia model in the image (the only one) |
| `batch_size` | 8 | Line crops per recognition batch (PyLaia's default) |
| `segmentation_min_height` | 1200 | As in the Kraken engine: shorter images are segmented on a taller canvas; `0` turns this off |

The sample `language` is not used.

## Outputs

For every sample the adapter writes into the sample directory:

| File | Content |
| --- | --- |
| `segmentation.json` | Kraken's segmentation container as found (moved back from the canvas), before recognition: regions, lines with boundary polygons and baselines, `text: null`. It loads with `kraken.containers.Segmentation(**data)` |
| `pylaia.json` | PyLaia's native per-line output, in Kraken's line order and aligned with `segmentation.json`: `kraken_id`, the `crop` box `[x0, y0, x1, y1]` in image pixels (or `null`), the decoded `symbols`, the `text` and per-character `confidences` |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output; lines carry `extra.kraken_id` |
| `result.json` | Status, processing time and, in `extra`: `canvas_offset`; the numbers of Kraken lines and regions, uncropped lines and predicted lines and regions; the Kraken, both torch, CUDA and PyLaia versions; the GPU name; the model loading time; the model files with SHA-256; and the parameters |

**Status.** An error in segmentation or recognition, for example CUDA out of memory, fails only that sample (`status: failed`, runner `engine-error`), and the container continues. If the recognizer process exits, the sample fails and the process is restarted for the next sample. An unusable GPU, invalid parameters, or a missing or unloadable model fail every sample. A page on which Kraken finds no lines, or only lines with blank text, is a **genuine empty result**: `success` with empty text and no regions.

## Tests

`tests/test_pylaia.py` runs offline in `make test`. It checks:

- line crop boxes, joining of PyLaia symbols and the recognizer's request format;
- the mapping to the standardized prediction with Czech and German text;
- parameter validation;
- the engine protocol through the runner, using the Kraken engine's fake `torch` and `kraken` modules and a fake recognizer process (`tests/fixtures/pylaia/fake_worker.py`). This covers recognizer errors, a recognizer that exits and is restarted, and setup errors.

`make pylaia-smoke` builds the image and runs real Kraken and PyLaia on the GPU on a synthetic full page, a paragraph crop and a blank page (an empty `success`), through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report`. It also runs with a GPU that is not visible, which must fail explicitly.
