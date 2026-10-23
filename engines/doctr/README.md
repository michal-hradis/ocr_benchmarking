# docTR engine (text detector and recognizer, GPU)

[docTR](https://github.com/mindee/doctr) as a modular detector/recognizer baseline ([#48](https://github.com/michal-hradis/ocr_benchmarking/issues/48)): one pinned pretrained pair, a word detector followed by a word recognizer, with docTR's own grouping of words into lines and blocks and its reading order. It runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-doctr:1.1.0-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1`, without weights |
| `requirements.in`, `overrides.txt`, `requirements.txt` | docTR and its dependencies, hash-locked against the base (`make docker-lock`), with headless OpenCV |
| `models.sha256` | Checksums of both checkpoints |
| `fetch_model.py` | Downloads the checkpoints into the cache on the host (`make doctr-model`) |
| `doctr_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`, preset `det1024` |
| `presets/det2048.toml` | Preset `det2048`, detection at 2048 pixels ([Presets](#presets)) |

## Versions and licenses

- **docTR 1.1.0** (2026-08-21), PyTorch backend (the only one since docTR 1.0), on the base's torch **2.14.1** for CUDA 13.0, with torchvision 0.29.1. docTR declares `opencv-python`, which loads libGL, which the base does not have. The lock replaces it with `opencv-python-headless` 5.0 (`overrides.txt`), which provides the same `cv2` module.
- **License:** docTR is **Apache-2.0**. The image installs it unmodified from PyPI. The checkpoints are release assets of the same repository, and docTR does not state a separate model license. They are not redistributed: `make doctr-model` downloads them from docTR's GitHub releases.
- **Checkpoints:** docTR's default pair (`doctr.models.ocr_predictor()`), the pretrained PyTorch weights docTR 1.1.0 itself points to. docTR checks the first eight hex digits of the SHA-256 in the file name; `models.sha256` pins the full SHA-256.

  | Role | Architecture | Checkpoint (release) | Size | SHA-256 |
  | --- | --- | --- | --- | --- |
  | Word detection | FAST base | `fast_base-688a8b34.pt` (v0.8.1) | 66 MB | `688a8b34…9c62ace` |
  | Word recognition | CRNN with VGG16-BN | `crnn_vgg16_bn-0417f351.pt` (v0.12.0) | 63 MB | `0417f351…ddac087` |

  The recognizer's vocabulary is docTR's `french`: digits, ASCII letters, punctuation, `°`, currency signs and the French accented letters. It has no Czech letters with háček or kroužek (č, ě, ř, š, ž, ů …), no long s or other historical characters and no non-Latin scripts. docTR publishes other recognizers (PARSeq, ViTSTR, multilingual models on Hugging Face). They are not integrated: this engine is one pinned pair, not a matrix of detectors and recognizers.
- **Hardware:** an NVIDIA GPU with CUDA 13.0 (driver ≥ 580). On an RTX 4090 the models load in about 11 s. A synthetic page takes under 0.4 s, and the 4194 × 6192 newspaper page about 1 s with `det1024` and under 2 s with `det2048`. Peak allocated GPU memory for the two dev-01 newspaper pages was 0.5 GB with `det1024` and 1.4 GB with `det2048` (3.1 GB at 3072, 5.4 GB at 4096). docTR's batch sizes stay at their defaults (2 pages for detection, 128 words for recognition), in float32.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-doctr                   # this image (no weights)
make doctr-model                    # opt-in: the pinned checkpoints (129 MB) into OCRBENCH_CACHE
uv run ocrbench run start --engine engines/doctr/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench run start --engine engines/doctr/presets/det2048.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id>... -o reports/doctr --inspect
make doctr-smoke                    # opt-in: build the image, run both presets on synthetic samples on the GPU
```

`make doctr-model` writes the checkpoints to `$(OCRBENCH_CACHE)/models/doctr/models/` (default `~/.cache/ocrbench`) and checks both against `models.sha256`. Files already present and correct are kept. The container reads them as `/cache/models/doctr/models/...` and checks them again before loading. A missing or changed file makes every sample fail with `engine-error`. The adapter sets `DOCTR_CACHE_DIR=/cache/models/doctr`, where docTR finds the verified checkpoints and downloads nothing. The runner gives the container no network.

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. `device = "cpu"` runs on the CPU, which is not the configuration recorded here.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample. The checkpoints are fixed.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `detection_size` | `1024` | Side of the square the detector resizes the page into, a multiple of 32. 1024 is docTR's own input size for `fast_base` and keeps docTR's pre-processor unchanged. Any other value replaces it with one built as docTR builds its own, for that size (aspect ratio kept, symmetric padding). `result.json` records the size docTR used (`detection_input`) |

All other settings keep docTR 1.1.0's `ocr_predictor` defaults: straight pages (`assume_straight_pages`), no page or crop orientation models, no language detection, no layout or table models, and the default detection thresholds. The sample `language` is not used: docTR's recognizer takes no language hint.

## Presets

A preset is a frozen engine configuration; its `name` identifies it as a separate system in reports. Both presets use the same image, checkpoints and resources and differ **only in `detection_size`**. Recognition always runs on word crops of the original image, so the preset changes which words are found and how they are cut, not the recognizer's input resolution.

| Preset | `name` | `detection_size` | Purpose |
| --- | --- | --- | --- |
| `det1024` (`engine.toml`) | `doctr` | 1024 | docTR's default |
| `det2048` (`presets/det2048.toml`) | `doctr-det2048` | 2048 | Large pages with small text, such as ~6000-pixel newspaper scans, which docTR otherwise shrinks about six times |

A new preset is a new TOML file with its own `name`; a changed setting is a new preset, not an edit of an existing one.

## Pipeline and outputs

For every sample the adapter decodes the image as stored with Pillow (EXIF orientation not applied, as in [PREDICTIONS.md](../../docs/PREDICTIONS.md#positions); other modes are converted to RGB) and passes it to docTR's `OCRPredictor` as one page:

1. **Detection:** FAST finds word boxes on the page resized into `detection_size` × `detection_size` (aspect ratio kept, symmetric padding). docTR removes the padding and returns the boxes relative to the page, as fractions of its width and height.
2. **Recognition:** CRNN recognizes each word crop, cut from the original image.
3. **Grouping and reading order:** docTR groups words into lines by their vertical position, splitting a line at large horizontal gaps. Its page export (`Page.export()`, docTR 1.1.0's default with `reading_order=True`) orders the lines with docTR's reading order (a column-following topological sort, `doctr.models.reading_order`) and groups consecutive lines of one column into blocks. This is deterministic for given boxes.

It writes into the sample directory:

| File | Content |
| --- | --- |
| `doctr.json` | docTR's native page export: the page dimensions, and the blocks in reading order with their lines and words (text, recognition confidence, box, detection score, crop orientation), in docTR's relative coordinates. It is the source of the prediction |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output |
| `result.json` | Status, processing time and, in `extra`: the numbers of blocks, words, lines and regions; the time docTR took (`doctr_seconds`); the docTR/torch/CUDA versions, GPU and detection input size; the model loading time; the architectures, checkpoints, their sources and SHA-256; and the parameters |

**Mapping.**

- **Coordinates:** docTR's relative boxes `((x0, y0), (x1, y1))` are multiplied by the input image's width and height and written as 4-point polygons in input-image pixels, rounded to 0.01 pixel. `doctr.json` keeps the relative values. The adapter checks that docTR's page dimensions are the input image's.
- **Regions:** every block becomes a region of type `text` with the block's box, in docTR's reading order, and `reading_order` lists them. docTR's blocks here carry no layout label (no layout model runs), so there is no `native_type`.
- **Lines:** every line with text keeps its box. Its text is its words joined by single spaces, in docTR's word order (left to right). `id` `l<n>` numbers the lines in reading order. Word boxes and confidences stay in `doctr.json`; the prediction format has no word level, and the adapter does not derive a line confidence from them.
- Words with blank text are left out of a line's text, lines without text are dropped, and blocks left without lines are not regions.
- **Page text:** the lines of a region joined by newlines, regions joined by a blank line.

**Status.** An error in one sample (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. An unusable GPU, invalid parameters, or missing or changed checkpoint files fail every sample. A page on which docTR detects no words is a **genuine empty result**: `success` with empty text.

## Observed behaviour

**Synthetic pages.** On the smoke pages (`make doctr-smoke`: a page, a paragraph crop, a blank page, a two-column page and a three-column newspaper page) both presets transcribe every line exactly (CER 0). docTR's reading order reads the columns one after another, with each paragraph as its own block.

**Measured on dev-01.** Both presets on the seven pages of dev-01 (runs `doctr-det1024-dev01` and `doctr-det2048-dev01`, not in Git; image `sha256:85d308c6…`, RTX 4090). All 14 samples were `success`. Rates are micro averages over the pages; fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)). Engine time is the sum of docTR's per-sample times, without model loading.

| Preset | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `det1024` | 76.6 % | 89.4 % | 76.6 % | 88.7 % | 22.9 | 12.1 | 2.3 s |
| `det2048` | 71.9 % | 91.4 % | 70.0 % | 91.4 % | 46.2 | 20.1 | 5.2 s |

CER per page, in % (fCER in parentheses where it differs), and the number of predicted lines:

| Page | Content | Lines `det1024` / `det2048` | `det1024` | `det2048` |
| --- | --- | --- | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 375 / 259 | 83.5 | 82.0 (76.5) |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 181 / 351 | 90.1 | 81.1 |
| `CkXII61-0017` | Glagolitic print | 20 / 29 | 91.6 | 95.2 |
| `dta-346672…` | Latin, Antiqua and italic | 34 / 36 | 15.4 | 16.4 |
| `dta-laube…` | German Fraktur, 1833 | 29 / 34 | 18.7 | 20.0 |
| `dta-martzi…` | German Fraktur, 1692 | 41 / 47 | 30.9 | 38.2 |
| `dta-steinen…` | German Antiqua, 1894 | 47 / 49 | 1.9 | 1.9 |

Observations:

- **Most of the newspapers is not detected at 1024 pixels.** The 6192-pixel page is shrunk six times, and its 30–40-pixel text lines become too small for the detector. The two newspapers yield 4,059 and 2,641 characters against 15,454 and 21,648 in the GT; 30,550 of the 33,571 character errors on dev-01 are deletions. With `det2048` the output length matches the GT (15,470 and 21,684 characters), and BoW F1 doubles.
- **At 2048 pixels, docTR merges side-by-side columns into one line.** docTR forms lines from words at the same height and splits them only at gaps larger than its paragraph break, which the narrow newspaper gutters are not. Lines such as `V râmci vojska armady generala Jeremenka krâtce rektor Masarykovy university prof. dr. V.` join two columns. The flexible alignment cannot separate such lines, so fCER stays close to CER. That is why `3e6309d0…` has fewer lines with `det2048` than with `det1024` although it has four times the text. With `det1024` the detected words are mostly headlines and other large type, which merge less.
- **The vocabulary has no Czech letters.** The recognizer writes `dékuje`, `Ceskoslovensko`, `câst` for `děkuje`, `Československo`, `část`. Every háček is a substitution, on every Czech page and with both presets. The Glagolitic page is outside the vocabulary, and the Fraktur pages are recognized as Antiqua shapes (`9as foûl das 3agen?` for `Was ſoll das Zagen?`).
- **On the smaller pages, upscaling to 2048 does not help.** The five pages 1334–1600 pixels wide are upscaled for detection; `det2048` splits more lines and blocks there and is 0–7 points worse in CER. fCER equals CER on these pages with both presets, so their errors are recognition, not reading order.
- `det2048` is a candidate for large scans with small text. A larger size recovers no more characters on dev-01: at 3072 and 4096 pixels the two newspapers give 15,729/21,870 and 15,761/21,896 characters (measured once, by hand, not presets).

## Tests

`tests/test_doctr.py` checks the mapping on committed native results, all offline in `make test`. These are `tests/fixtures/doctr/page.json`, `columns.json`, `newspaper.json` and `blank.json`: real docTR 1.1.0 page exports (`det1024`) for the synthetic pages of `tests/synthetic.py`. The tests cover docTR's reading order across columns, relative coordinates converted to input-image pixels, line text from words, blank lines and blocks, empty and inconsistent results, parameter validation and the presets. They also check that the checksums cover the pinned checkpoints and match docTR's hash prefixes, that `fetch_model.py` keeps verified files and rejects wrong ones, and the engine protocol through the runner with fake `torch`, `numpy` and `doctr` modules (`tests/fixtures/doctr/fake`), including the detection size reaching docTR, evaluation and a report. `make doctr-smoke` builds the image and runs real docTR with both presets on the GPU through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on the five synthetic pages, plus a run without the cache mount, which must fail explicitly.
