# Surya OCR engine (detector, recognizer and layout/reading order, GPU)

[Surya](https://github.com/datalab-to/surya) as a distinct pipeline ([#47](https://github.com/michal-hradis/ocr_benchmarking/issues/47)): a text line detector, a line recognizer on the detected lines, and a layout model that gives the blocks and their reading order. It runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-surya:0.17.1-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1`, without weights |
| `requirements.in`, `requirements.txt` | Surya and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of every checkpoint file the adapter loads |
| `fetch_model.py` | Downloads the checkpoints into the cache on the host (`make surya-model`) |
| `surya_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions and licenses

- **Surya 0.17.1** (2026-01-30), the last release of the first generation, on the base's torch **2.14.1** for CUDA 13.0, with **transformers 4.57.6** (the last 4.x release; Surya 0.17.1 only requires ≥ 4.56.1) and torchvision 0.29.1. Later releases (0.20 and newer, "Surya 2") replace the line recognizer with a VLM served by vLLM or llama.cpp that returns block-level HTML without text lines. Surya 2 is a different engine and is not integrated here.
- **Code license:** Surya 0.17.1 is **GPL-3.0-or-later**. The image installs it unmodified from PyPI; no Surya code is copied into this repository.
- **Weights license:** the checkpoints are under Datalab's **modified AI Pubs Open RAIL-M** license ([`MODEL_LICENSE`](https://github.com/datalab-to/surya/blob/v0.17.1/MODEL_LICENSE)). Research and personal use are free. Commercial use needs a license from Datalab if you or your organization had more than US $2M of gross revenue in the prior year, or raised more than US $2M in funding. The license also lists use restrictions. Check it before using the weights. The weights are not redistributed: `make surya-model` downloads them from Datalab.
- **Checkpoints:** the three Surya 0.17.1 defaults, downloaded from `https://models.datalab.to/<checkpoint>/`. `models.sha256` pins all 21 files the adapter loads. Each checkpoint's `README.md` and `.gitattributes`, and the pickled `training_args.bin`, are not downloaded.

  | Role | Checkpoint | Size |
  | --- | --- | --- |
  | Text line detection (EfficientViT segmentation) | `text_detection/2025_05_07` | 77 MB |
  | Text recognition (Surya foundation model, OCR task) | `text_recognition/2025_09_23` | 1.4 GB |
  | Layout and reading order (foundation model, layout task) | `layout/2025_09_23` | 1.4 GB |

- **Hardware:** an NVIDIA GPU with CUDA 13.0 (driver ≥ 580). On an RTX 4090 (24 GB) the models load in about 12 s, and a synthetic page takes under 1 s once the models are warm. Surya's GPU batch defaults are 256 lines for recognition (about 40 MB of VRAM per line, by Surya's README) and 36 image chunks for detection. With less memory, set `recognition_batch_size`. The recognizer runs in bfloat16 and the detector in float16. Neither flash-attention nor `torch.compile` is used (`sdpa` attention).
- **torch native ops:** torch 2.14 sends some operations (here the rotary embedding's batched outer product) to Triton kernels compiled at first use, which needs a C compiler the image does not have. The image sets `TORCH_DISABLE_NATIVE_JIT=1` so these run on the standard ATen kernels. Without it every sample fails with `Failed to find C compiler`.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-surya                   # this image (no weights)
make surya-model                    # opt-in: the pinned checkpoints (3 GB) into OCRBENCH_CACHE
uv run ocrbench run start --engine engines/surya/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/surya --inspect
make surya-smoke                    # opt-in: build the image, run it on synthetic samples on the GPU
```

`make surya-model` writes the checkpoints to `$(OCRBENCH_CACHE)/models/surya/<checkpoint>/` (default `~/.cache/ocrbench`) and checks every file against `models.sha256`. Files already present and correct are kept. The container reads them as `/cache/models/surya/...` and checks them again before loading. A missing or changed file makes every sample fail with `engine-error`. Surya never downloads anything in the container: the adapter loads the checkpoints from local directories, the image sets `HF_HUB_OFFLINE`, and the runner gives it no network.

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. `device = "cpu"` runs on the CPU (float32), which is much slower and not the configuration recorded here.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample. The checkpoints are fixed: this engine is one pinned model combination.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `math_mode` | `false` | Surya's math recognition. When `true`, the recognizer may write mathematics as LaTeX in `<math>` tags. Surya's own CLI turns it on by default (`--disable_math` turns it off). It is off here because the benchmark scores plain text of prints and manuscripts |
| `recognition_batch_size` | `null` | Lines recognized per batch; `null` uses Surya's default for the device (256 on CUDA, 32 on the CPU) |

Surya's other settings keep their 0.17.1 defaults, among them the detection thresholds (text 0.6, blank 0.35), the 1400-pixel detection chunk height and the recognizer's per-line token limit. The sample `language` is not used: Surya's recognizer takes no language hint.

## Pipeline and outputs

For every sample the adapter decodes the image as stored with Pillow (EXIF orientation not applied, as in [PREDICTIONS.md](../../docs/PREDICTIONS.md#positions); other modes are converted to RGB). It then runs, on that one image:

1. **Detection:** `DetectionPredictor` finds text lines as 4-point polygons with confidences. Surya resizes internally and splits tall pages into chunks, and returns the polygons in input-image pixels.
2. **Recognition:** `RecognitionPredictor` (task `ocr_with_boxes`) recognizes each detected polygon. This is what Surya's OCR does when it is given a detector; the adapter passes the polygons explicitly so the detection result is kept unchanged.
3. **Layout:** `LayoutPredictor` finds blocks with a label and a reading order `position`. Like the detector, it works on the whole image and returns input-image pixels.

It writes into the sample directory:

| File | Content |
| --- | --- |
| `surya.json` | Surya's native results, as `model_dump` of its result objects: `detection` (line polygons and confidences, without the heatmaps), `recognition` (lines with text, polygon, confidence and per-character boxes) and `layout` (blocks with polygon, label, `position`, confidence and top-5 labels). It is the source of the prediction |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output |
| `result.json` | Status, processing time and, in `extra`: the numbers of detected and predicted lines, of lines outside every block, of layout blocks and of regions; the time Surya took (`surya_seconds`); the Surya/transformers/torch/CUDA versions and GPU; the model loading time; the checkpoints with the SHA-256 of every file; and the parameters |

**Mapping.**

- **Regions:** every layout block becomes a region, in Surya's reading order (`position`; `extra.position` keeps it). Each region has the block polygon, the Surya label as `native_type` and a [recommended type](../../docs/PREDICTIONS.md#region-types): `Text` → `text`, `SectionHeader` → `heading`, `ListItem` → `list-item`, `Caption` → `caption`, `Footnote` → `footnote`, `PageHeader` → `page-header`, `PageFooter` → `page-footer`, `Table` → `table`, `Equation` → `formula`, `Picture` and `Figure` → `figure`. Anything else (`Form`, `Code`, `TableOfContents`) becomes `other`. Blocks without lines (for example pictures) stay as regions without text. `reading_order` lists all regions.
- **Lines:** every recognized line with non-blank text keeps its detected polygon and Surya's mean character confidence (0–1). It goes into the block that contains the largest part of its box, provided that part is at least half of the box; ties go to the block earlier in reading order. Within a block, lines are read top to bottom, and boxes on one visual line (vertical overlap ≥ 80 % of the smaller height) are read left to right. Surya itself does not assign lines to blocks. Its detection order is row by row across the page, which interleaves side-by-side columns, so the adapter uses the layout order instead. `id` `l<n>` is the line's position in Surya's `text_lines`.
- **Lines outside every block** follow after the blocks, top to bottom, in one region of type `text` without a polygon (`extra.grouping: "unassigned"`). This keeps all recognized text but does not claim a layout structure; `unassigned_lines` in `result.json` counts them.
- **Text:** the recognizer writes inline formatting as HTML. The prediction drops the tags `b`, `i`, `u`, `s`, `del`, `mark`, `sup`, `sub` and `small` (keeping their text), replaces `<br>` by a space and decodes HTML entities. Other tags, such as `<math>` with `math_mode`, stay as text. `surya.json` keeps Surya's text unchanged, for example `<b>Evening Gazette</b>`.
- **Page text:** the lines of a region joined by newlines, regions joined by a blank line.

**Status.** An error in one sample (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. An unusable GPU, invalid parameters, or missing or changed checkpoint files fail every sample. A page on which Surya detects no lines is a **genuine empty result**: `success` with empty text. On the blank synthetic page the layout model reports one `Picture` block, which stays a region without text.

## Observed behaviour

On the synthetic smoke pages (`make surya-smoke`: a page, a paragraph crop, a blank page, a two-column page and a three-column newspaper page) Surya transcribes every line exactly (CER 0). The layout order reads the columns one after another, while the detection order alternates between them. It labels the two-line paragraph crop as a `SectionHeader` (type `heading`); this changes the region type but not the text. The adapter has not been run on the development set or on real historical scans; results there are not documented.

## Tests

`tests/test_surya.py` checks the mapping on committed native results, all offline in `make test`. These are `tests/fixtures/surya/page.json`, `columns.json`, `newspaper.json` and `blank.json`: real Surya 0.17.1 output for the synthetic pages of `tests/synthetic.py`, with the per-character boxes removed to keep them small. The tests cover reading order across columns, polygons in input-image pixels, removal of formatting tags, region types, line assignment and unassigned lines, line order within a block, empty and inconsistent results, and parameter validation. They also check that the checksums cover the pinned checkpoints, that `fetch_model.py` keeps verified files and rejects wrong ones, and the engine protocol through the runner with fake `torch`, `transformers` and `surya` modules (`tests/fixtures/surya/fake`), including evaluation and a report. `make surya-smoke` builds the image and runs real Surya on the GPU through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on the five synthetic pages, plus a run without the cache mount, which must fail explicitly.
