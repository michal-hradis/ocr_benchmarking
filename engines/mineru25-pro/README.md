# MinerU2.5-Pro engine (coarse-to-fine page parsing, GPU)

Additional system ([#51](https://github.com/michal-hradis/ocr_benchmarking/issues/51)): [MinerU2.5-Pro](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) is a 1.2B document parsing vision-language model from OpenDataLab. One model performs both steps of its **coarse-to-fine** parsing:

1. **Layout detection** on the whole page, scaled down.
2. **Content recognition** of every detected block, cropped from the original image at full resolution.

This adapter runs the official two-step extraction of [mineru-vl-utils](https://github.com/opendatalab/mineru-vl-utils), MinerU's client for the model (`MinerUClient.two_step_extract`, as in the model card), unchanged. It uses the transformers backend in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol). The input is always the page image, and every crop is made inside the container. PDFs and PDF text layers are not used.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-mineru25-pro:2605-1.2B-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | mineru-vl-utils with its `transformers` extra, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make mineru25-pro-model`) |
| `mineru25_pro_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions and license

- **Inference:**
  - mineru-vl-utils **2.0.7** (Apache-2.0, released 2026-10-10) with its `transformers` extra: transformers **5.19.0**, accelerate **1.15.0** and torchvision **0.29.1+cu130**.
  - These run on PyTorch **2.14.1** for CUDA 13.0 (the `base-cu130-torch` family), which needs an NVIDIA driver ≥ 580. The image has been run on an RTX 4090.
- **Model:** [`opendatalab/MinerU2.5-Pro-2605-1.2B`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) at commit `08aaea840498d49ce16247b6263196cf02814885`, **Apache-2.0**, 2.3 GB (bfloat16 safetensors).
  - This is the 2605 update (2026-05-21) of MinerU2.5-Pro ([technical report](https://arxiv.org/abs/2604.04771)). It keeps the architecture of MinerU2.5: Qwen2-VL (`Qwen2VLForConditionalGeneration`) with a 675M NaViT vision encoder and a 0.5B Qwen2 language model. The model card lists Chinese and English.
  - All eleven model, tokenizer and processor files of the revision are downloaded; the README and `.gitattributes` are left out.
  - The earlier 2604 revision (`opendatalab/MinerU2.5-Pro-2604-1.2B`) is not used.
- The full **MinerU application** (the `mineru` package with PDF handling, its other backends and middle-JSON output) is not used. Its VLM backend calls the same mineru-vl-utils extraction.
- The **vLLM backends** recommended by the model card for throughput are not used. vLLM would need its own CUDA build and engine process in the container, as for the other VLM engines here.
- **Models are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)), downloaded by an explicit step:

  ```sh
  make mineru25-pro-model                        # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/mineru25-pro/MinerU2.5-Pro-2605-1.2B/<revision>/` under the cache directory and checks every file against `models.sha256`.
  - Runs need `--cache <dir>`.
  - At start-up the adapter checks the directory again. It must hold exactly the files of `models.sha256`, all matching, because transformers may load other files it finds there. Otherwise every sample fails.
  - The SHA-256 of every file is recorded with each sample.
  - transformers never downloads anything: the model is loaded from the local directory, `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` are set and the container has no network.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-mineru25-pro            # this image (no weights)
make mineru25-pro-model             # the pinned weights, 2.3 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/mineru25-pro/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/mineru25-pro --inspect
make mineru25-pro-smoke             # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. Without a visible GPU every sample fails with `engine-error` and the reason; there is no CPU fallback.

## Pipeline, prompts and parameters

The model and processor are loaded with mineru-vl-utils' own loaders: the checkpoint's bfloat16, the tied output embeddings that transformers 5 would otherwise initialize randomly, and the torchvision image processor. Prompts, sampling and post-processing are the client's defaults. For every page, `two_step_extract`:

1. **Detects the layout.** The whole page is resized to **1036 × 1036 pixels** (bicubic, ignoring the aspect ratio; 1,369 image tokens) and read with the `Layout Detection:` prompt.
   - The model writes one line per block: a box on a 0–1000 grid of the page, a type and the text rotation.
   - The order of the lines is the model's reading order.
   - Text blocks can be marked as continuing the previous block (`merge_prev`, for paragraphs that run across columns or pages).
   - Duplicate boxes (IoU > 0.9, same type) are dropped, and so are text and equation boxes inside a table.
2. **Crops every block from the original image** at its own resolution.
   - The crop is rotated upright when the layout gave a rotation.
   - The Qwen2-VL image processor then scales it to between 50,176 and 1,605,632 pixels (`preprocessor_config.json`, in 28 × 28 units, one image token each, at most 2,048 tokens).
   - A narrow newspaper column is therefore read at close to its scanned resolution, while a large block, such as a full book page of text, is scaled down to about 1.6 million pixels.
3. **Recognizes every block** with the prompt for its type: `Table Recognition:`, `Formula Recognition:` for equations, and `Text Recognition:` for all other text blocks, including titles, headers, footers, page numbers, footnotes, captions and code.
   - Images and charts are not read: image analysis is off by default (`image_analysis=False`).
   - List and equation-block containers are not read; their items are separate blocks.
4. **Post-processes** the output.
   - Tables are converted from OTSL to HTML.
   - Equations are cleaned up and wrapped in `\[ … \]`.
   - `[Non-Text]` placeholder blocks are dropped (headers and footers are kept, empty).
   - List items become text blocks. Headers, footers and other paratext are kept (`abandon_paratext=False`).

Decoding is greedy (temperature 0) with `no_repeat_ngram_size = 100` and no repetition penalty. The presence and frequency penalties of MinerU's sampling parameters are ignored by the transformers backend. Blocks are generated one at a time (batch size 1).

The sample `language` and other metadata are not used.

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | PyTorch device, `cuda:<n>` |
| `max_new_tokens` | not set | Generated tokens per call (the layout and every block). Not set keeps mineru-vl-utils' default: a generation may run to the end of the model's **8,192-token context** (prompt included) |

## Truncation, time and memory

- **Truncation.** The transformers client removes the end tokens before decoding, so the adapter wraps the model's `generate` to record every call.
  - For each call it records the image tokens, the generated tokens and whether the call ended with `<|im_end|>` or `<|endoftext|>`.
  - If the layout call or any block stops without an end token (at `max_new_tokens` or at the end of the context), the sample is `truncated`, with `message` naming what was cut off. The text produced so far is kept and scored as produced ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)).
  - A cut-off layout loses the blocks it did not list; they are not read.
- **Recorded per sample** (`result.json` `extra`):
  - `layout`: image tokens, generated tokens, stop reason and seconds of the layout step.
  - `vl_calls`, `generated_tokens`, `max_generated_tokens`, `image_tokens`, `max_image_tokens` and `truncated_blocks` of the block recognition, and its `recognition_seconds`.
  - `other_seconds`: the rest of the page time, which is cropping, image preprocessing, tokenization and post-processing.
  - `gpu_peak_memory_mb`: the peak GPU memory PyTorch allocated during the page.
  - The effective `max_new_tokens`.
- **Memory.**
  - The bfloat16 weights take 2.3 GB of GPU memory, and one generation runs at a time.
  - The peak was 2.4–2.5 GB on the dev-01 pages, newspapers included, because every block is a separate generation of at most 2,048 image tokens. The largest block crop on dev-01 used 1,980 image tokens.
  - A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues. Host memory is limited to 16 GB by `[resources]`.
- **Time** on the RTX 4090:
  - Loading the model takes about 13 s, once per container.
  - The layout step reads the page once at the fixed 1036 × 1036 size. Its cost grows with the number of blocks it lists: 3–5 s for the 1600-pixel dev-01 book pages (6–11 blocks, 130–240 tokens) and 19–26 s for the two large Czech newspaper pages (41 and 56 blocks, 900–1,230 tokens).
  - Block recognition dominates: 7–20 s for the book pages and 152–207 s for the newspapers (6,900–9,600 generated tokens, 43–47 tokens/s).
  - Cropping, image preprocessing and post-processing (`other_seconds`) took under 1 s per page.
  - In total, 12–24 s per book page and 172–234 s per newspaper page. `timeouts.sample = 900` seconds allows for slower pages.
- **Glagolitic.** On the dev-01 Glagolitic page, which the model cannot read, the blocks produced unrelated digits, `###` sequences and Devanagari-like text. Every generation still ended with an end token below the context limit, so the page is `success` and scored as produced.

## Outputs

| File | Content |
| --- | --- |
| `mineru.json` | The native result of `two_step_extract`: the post-processed blocks in reading order, each with `type`, `bbox` (0–1, relative to the page), `angle`, `content` and, for text, `merge_prev` |
| `page.md` | The official Markdown (`json2md` of mineru-vl-utils): the block contents joined by blank lines, continued paragraphs merged |
| `prediction.json` | The standardized prediction, the primary output |
| `result.json` | Status, processing time and, in `extra`: the counts above, `blocks`, the mineru-vl-utils, transformers, PyTorch and CUDA versions, the GPU name and memory, the model loading time, the model (repository, revision, file checksums, the configured `model`) and the parameters |

**Prediction.** Every block of `mineru.json` becomes a region, in that order, which is the model's reading order.

- Each region has the block's box as a four-point polygon. The box is relative to the page (the layout step sees the whole page), so it is mapped to input-image pixels by multiplying with the image width and height. The box comes from the layout model's 0–1000 grid, so it is accurate to about 0.1 % of the page size.
- The block type is `native_type`, and the region type is mapped from it:
  - `title`, `doc_title` and `paragraph_title` → `heading`
  - `header`, `footer` and `page_number` → `page-header`, `page-footer` and `page-number`
  - captions → `caption`, footnotes (`page_footnote` included) → `footnote`
  - `table` → `table`, `equation` and `formula_number` → `formula`
  - images and charts → `figure`
  - text, code, references, aside text and index → `text`, list items → `list-item`
  - list containers and unknown types → `other`
- Its `text` is the block's content, converted to plain text:
  - An HTML table becomes one line per row with tab-separated cells, as in the host converter ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)).
  - An equation keeps its TeX source without the `\[ … \]` delimiters.
  - Other text is kept as generated, including inline formulas and any line breaks the model wrote inside a paragraph.
- Blocks without text (images, containers) stay regions without text, with their box.
- A text block that continues the previous one (`merge_prev`) stays its own region, with its own box; the flag and the block's `angle` are kept in the region's `extra`. `page.md` joins such paragraphs; the prediction does not, because the two parts have different positions.
- The page text is the region texts joined by blank lines, in reading order.
- There are **no line positions**: each block is read as a whole, and the output has no geometry. No line geometry is invented. The block boxes are real layout-model geometry.

**Status.** An error during the extraction (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make mineru25-pro-model and run with --cache`);
- the model files do not match `models.sha256`, or the directory holds other files.

A page without detected blocks is a **genuine empty result**: `success` with empty text.

## Not included

- Image and chart analysis (`image_analysis=True`), which would add descriptions and chart data that are not page text.
- Cross-page table merging and the full MinerU application.
- The vLLM, LMDeploy, MLX and llama.cpp backends of mineru-vl-utils.

## Tests

`tests/test_mineru25_pro.py` runs offline in `make test`. It checks:

- the block mapping, parameter validation, the model checksums and the download script (without the network);
- the engine protocol through the runner, with fake `mineru_vl_utils`, `torch` and `transformers` modules (`tests/fixtures/mineru25-pro/fake`).

The fake returns real MinerU2.5-Pro-2605 results recorded with this image for synthetic pages: a page, a paragraph crop, two columns and a three-column newspaper. For the layout and for every block, it calls the model's `generate` the way the transformers backend does, so that truncation at `max_new_tokens` and at the end of the context is exercised. It also simulates an out-of-memory error and a blank page.

`make mineru25-pro-smoke` (after `make mineru25-pro-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on synthetic full pages, a paragraph crop, a two-column page, a newspaper page and a blank page. A second run with `max_new_tokens = 8` must mark the page `truncated` and keep and score its partial text. It also runs without `--cache`, which must fail explicitly.
