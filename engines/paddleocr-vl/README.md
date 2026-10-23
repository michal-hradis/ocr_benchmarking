# PaddleOCR-VL-1.6 engine (layout detection + compact document VLM, GPU)

Compact document VLM ([#49](https://github.com/michal-hradis/ocr_benchmarking/issues/49)): [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) is a 0.9B vision-language model that reads one page element at a time. The official page-level pipeline (`PaddleOCRVL` in PaddleOCR, pipeline version `v1.6`) first detects layout blocks and their reading order with **PP-DocLayoutV3**. Then it runs the VL model on every block with a task prompt chosen by the block label. This adapter runs that pipeline unchanged in a GPU Docker image on the shared Paddle base ([DOCKER.md](../../docs/DOCKER.md)). It follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol). The input is always the full page image; PDFs are not used.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-paddleocr-vl:1.6-v1` from `ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1` |
| `requirements.in`, `requirements.txt` | PaddleOCR with the PaddleX `ocr` extra, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revisions |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make paddleocr-vl-model`) |
| `paddleocr_vl_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions

- **Inference:** PaddleOCR **3.7.0** and PaddleX **3.7.2** with the `ocr` extra (the same packages as [PP-StructureV3](../ppstructurev3/README.md); only huggingface-hub was locked at a newer version). They run on PaddlePaddle **3.4.0** for CUDA 13.0 (the `base-cu130-paddle` family), which needs an NVIDIA driver ≥ 580. The image has been run on an RTX 4090.
  - The pipeline configuration is PaddleX's `PaddleOCR-VL-1.6.yaml` of that version. Only the arguments under [Parameters](#parameters) change it.
  - The VL model runs with PaddleX's **native** backend: the Paddle implementation of the model, loaded from the Hugging Face safetensors, in bfloat16, one block per call.
  - The model card also describes a vLLM or SGLang server backend (faster) and a transformers example that reads single elements only. Neither is used: a server inside the container would need its own CUDA stack and process ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)).
- **Models** from Hugging Face at pinned commits, both Apache-2.0:

| Role | Model | Commit | Size |
| --- | --- | --- | --- |
| Layout detection and reading order | [`PaddlePaddle/PP-DocLayoutV3`](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) | `241f8bdfc77a7c7bee915a5057aaee58c235a8d3` | 131 MB |
| Element recognition (`PaddleOCR-VL-1.6-0.9B`) | [`PaddlePaddle/PaddleOCR-VL-1.6`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | `c5630abae1d940eafe0697512a0325494b02ab42` | 1.9 GB |

  - PaddleOCR-VL-1.6 was released on 2026-05-28 ([technical report](https://arxiv.org/abs/2606.03264)). It has the architecture of PaddleOCR-VL-1.5: a NaViT-style vision encoder (SigLIP, 27 layers, 14-pixel patches) and the ERNIE-4.5-0.3B language model (18 layers).
  - The model card lists English, Chinese and "multilingual". The benchmarks are OmniDocBench and Real5-OmniDocBench.
  - Only the files that PaddleX loads are downloaded (`models.sha256`): weights, configurations and tokenizer of the VL model, and the Paddle inference model of PP-DocLayoutV3. The repository's custom transformers code and README are left out.
- **Models are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)), downloaded by an explicit step:

  ```sh
  make paddleocr-vl-model                        # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/paddleocr-vl/<repository>/<revision>/` for both models under the cache directory and checks every file against `models.sha256`.
  - Runs need `--cache <dir>`.
  - At start-up the adapter checks both directories again. They must hold exactly the files of `models.sha256`, all matching, because PaddleX may load other files it finds there. Otherwise every sample fails.
  - The SHA-256 of every file is recorded with each sample.
  - PaddleX never downloads anything: both model directories are given explicitly, the model-host check is off (`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`) and the container has no network.

The Ubuntu packages (`libgomp1` in the base, `libgl1` and `libglib2.0-0t64` for OpenCV here) are pinned by version, as for PP-StructureV3.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-paddle            # shared Paddle base, once (about 6 GB)
make docker-paddleocr-vl            # this image (no models)
make paddleocr-vl-model             # the pinned models, 2 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/paddleocr-vl/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/paddleocr-vl --inspect
make paddleocr-vl-smoke             # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. The image is a CUDA build: without a visible GPU every sample fails with `engine-error` and the reason; there is no CPU fallback (`PADDLE_PDX_DISABLE_DEVICE_FALLBACK`).

## Pipeline, prompts and parameters

For every page the pipeline:

1. **Detects layout blocks** with PP-DocLayoutV3: 25 classes (text, paragraph and document titles, headers, footers, page numbers, footnotes, tables, formulas, images, charts, seals, vertical text and others). Each block gets a box, a polygon, and a place in the reading order the model predicts. The score threshold is 0.3, and overlapping boxes are suppressed or merged per class as in the pipeline configuration.
2. **Merges adjacent text blocks** that are aligned with each other (`merge_layout_blocks`, on in v1.6). Tables and images are never merged.
3. **Crops every block** from the page and runs the VL model on it. The prompt depends on the label:
   - `Table Recognition:` for tables
   - `Formula Recognition:` for formulas (the crop is trimmed to the ink)
   - `OCR:` for every other text block
   - Images, charts and seals are not read: chart and seal recognition are off in v1.6.
4. **Post-processes the output.** A repeated tail is cut down to one copy (at least 50 repeats, 5,000 for tables). Tables are converted from the model's OTSL format to HTML, and `\(…\)`/`\[…\]` formulas to `$…$`/`$$…$$`.

The VL model scales every block crop to between 112,896 and 1,003,520 pixels (about 336 × 336 to 1002 × 1002), keeping the aspect ratio, in units of 28 × 28 pixels (one image token each).
- A full newspaper column is therefore read at a reduced resolution when it is large.
- A small block, such as a page number, is enlarged.

Decoding is greedy. The native backend supports no sampling, temperature or repetition penalty.

The document preprocessing steps (orientation classification and unwarping) are off in the v1.6 defaults, and the adapter keeps them off, because they would change the image the positions refer to. The sample `language` and other metadata are not used; the prompts are the pipeline's, recorded in this README rather than per sample.

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the pipeline defaults, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `gpu:0` | Paddle device, `gpu:<n>` |
| `max_new_tokens` | 4096 | Generated tokens per block before the block counts as cut off (the pipeline's default, made explicit) |
| `min_pixels`, `max_pixels` | 112,896, 1,003,520 | Block crops are scaled to at least / at most this many pixels before the VL model |
| `layout_threshold` | 0.3 | Layout detection score threshold for all classes, `0 < value <= 1` |

## Truncation, time and memory

- **Truncation.** The pipeline returns only the decoded text of each block, so the adapter wraps the VL model's `generate` to record how many tokens each block generated and whether it ended with the end token.
  - If any block stops at `max_new_tokens` without the end token, the sample is `truncated` (runner status `truncated`), with `message` naming the number of cut-off blocks. The text produced so far is kept and scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)).
  - The pipeline's repetition clean-up may already have shortened such a block. A block that loops below the limit and is cleaned up is not reported.
  - The counts are recorded per sample: `vl_calls`, `generated_tokens`, `max_generated_tokens`, `truncated_blocks` and the effective `max_new_tokens`.
- **Dense and multi-column pages.** Each block is read separately, so a newspaper page needs many short generations instead of one long one, and no block comes near 4,096 tokens in normal text.
  - Text that the layout model misses (no block) is not read at all; it counts as deletions.
  - On the synthetic three-column newspaper page of the tests, PP-DocLayoutV3 orders the paragraphs row by row across the columns instead of column by column. The strict metrics score that order; the flexible metrics are independent of it.
- **Memory.** The bfloat16 VL weights take 1.9 GB of GPU memory, and one block is generated at a time. A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues. Lowering `max_pixels` reduces the memory needed. Host memory is limited to 16 GB by `[resources]`.
- **Time.** Loading both models takes about 12 s on the RTX 4090. Time per page grows with the number of blocks and the amount of text. On the dev-01 pages it was 9–26 s for 1600-pixel book pages (6–10 blocks) and 158–206 s for the two large Czech newspaper pages (28 and 46 blocks read, 6,000–8,000 generated tokens). `timeouts.sample = 900` seconds allows for slow pages.
- **Repetition.** On the dev-01 Glagolitic page, which the model cannot read, every block produced unrelated Latin, Greek or symbol text, and one block repeated a phrase until `max_new_tokens`. The page is reported as `truncated` and scored as produced. A cut-off block can end in U+FFFD where the limit split a multi-token character.

## Outputs

| File | Content |
| --- | --- |
| `paddleocr_vl.json` | The native result (`PaddleOCRVLResult.json`): the parsed blocks (`parsing_res_list`: label, content, box, polygon, order), the raw layout detections with scores (`layout_det_res`) and the model settings |
| `prediction.json` | The standardized prediction, the primary output |
| `result.json` | Status, processing time and, in `extra`: the generation counts above, `blocks`, `layout_boxes`, `model_settings`, the PaddleOCR/PaddleX/Paddle/CUDA versions, the GPU name, the model loading time, the models (`pipeline_version`, the name, repository and revision of both models, file checksums, the configured `model`) and the parameters |

**Prediction.** Every block of `parsing_res_list` becomes a region, in that order: the reading order of PP-DocLayoutV3, after the pipeline's merging.

- Each region has the block's polygon from the layout model (its box when there is no polygon), in input-image pixels. The block label is `native_type`, and the region type is mapped from it (`doc_title` and `paragraph_title` → `heading`, `header`/`footer`/`number` → `page-header`/`page-footer`/`page-number`, `figure_title` → `caption`, `table`, formulas → `formula`, images, charts and seals → `figure`, other text labels → `text`).
- Its `text` is the VL model's output for the block, converted to plain text:
  - An HTML table becomes one line per row with tab-separated cells, as in the host converter ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)).
  - A formula keeps its TeX source without the `$$` or `$` delimiters.
  - Other text is kept as generated, including inline `$…$` formulas. The model usually writes a paragraph as one line, so the line breaks of the page are not kept.
- Blocks without text (figures, empty output) stay regions without text, with their polygon.
- The page text is the region texts joined by blank lines, in reading order.
- Headers, footers, page numbers, footnotes and side notes are kept as regions with their text, although the pipeline's own Markdown leaves them out (`markdown_ignore_labels`): the reference transcriptions contain them.
- There are **no line positions**: the model reads a whole block, and its output has no geometry. No line geometry is invented. The block polygons are real layout-model geometry.

**Status.** An error during the pipeline (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- a model directory is missing (`... does not exist; download the models with make paddleocr-vl-model and run with --cache`);
- the model files do not match `models.sha256`, or a model directory holds other files.

A page without detected layout blocks is a **genuine empty result**: `success` with empty text.

## Not included

- **Text spotting** (`Spotting:` prompt, line-level text with positions from the model) is not part of the page pipeline. It would need a separate preset with layout detection off; a possible follow-up.
- Chart and seal recognition (off in v1.6), document unwarping and orientation correction, and the vLLM/SGLang server backends.

## Tests

`tests/test_paddleocr_vl.py` runs offline in `make test`. It checks:

- the result mapping, parameter validation, the model checksums and the download script (without the network);
- the engine protocol through the runner, with a fake `paddleocr` (`tests/fixtures/paddleocr-vl/fake`) and the fake `paddle`, `paddlex` and `cv2` of the PaddleOCR tests.

The fake returns real PaddleOCR-VL-1.6 results for synthetic pages: a page, two columns, a three-column newspaper and a blank page. For every block, it calls the VL model's `generate` the way the pipeline does, so that truncation at `max_new_tokens` is exercised. It also simulates a pipeline error and an out-of-memory error.

`make paddleocr-vl-smoke` (after `make paddleocr-vl-model`) builds the image and runs the real pipeline on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on synthetic full pages, a paragraph crop, a two-column page, a newspaper page and a blank page. A second run with `max_new_tokens = 8` must mark the page `truncated` and keep and score its partial text. It also runs without `--cache`, which must fail explicitly.
