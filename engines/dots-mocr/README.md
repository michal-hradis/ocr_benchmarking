# dots.mocr engine (page-image to layout with text, GPU)

[dots.mocr](https://huggingface.co/dots-studio/dots.mocr) by rednote hilab ([#52](https://github.com/michal-hradis/ocr_benchmarking/issues/52)) is a 3-billion-parameter multilingual document parsing vision-language model, the successor of dots.ocr. With its document parsing prompt it reads a whole page image and generates a JSON list of layout elements in reading order. Each element has a box, a category, and its text: Markdown for text, HTML for tables, and LaTeX for formulas. There is no separate detection or line recognition. The engine runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-dots-mocr:3B-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision, including the model's own Python code |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make dots-mocr-model`) |
| `dots_mocr_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions and license

- **Model:** [`dots-studio/dots.mocr`](https://huggingface.co/dots-studio/dots.mocr) at commit `e539fbb52280393adc081b289ec597430a0f9031` (2026-07-04). The model card links to `rednote-hilab/dots.mocr`, but the published checkpoint is in the `dots-studio` organization. Paper: [Multimodal OCR: Parse Anything from Documents](https://arxiv.org/abs/2603.13032).
  - The architecture is `DotsOCRForCausalLM`, defined by Python code in the checkpoint (`configuration_dots.py`, `modeling_dots_ocr.py`, `modeling_dots_vision.py`). It combines a 42-layer vision transformer with 14-pixel patches (merged 2 × 2 into one token) and a 28-layer, 1536-wide Qwen2 language model with a 131,072-token context.
  - The weights are 6.08 GB of bfloat16 in two safetensors files. Every file, including the model code, is checked against `models.sha256`.
  - The model card lists multilingual, English and Chinese. Historical European prints, Fraktur, Glagolitic and Czech are not specifically claimed.
  - The SVG variant `dots.mocr-svg` is a separate checkpoint and is not used.
- **License:** the model card states the MIT license. The checkpoint and the [official code](https://github.com/rednote-hilab/dots.mocr) also include the [dots.mocr License Agreement](https://huggingface.co/dots-studio/dots.mocr/blob/e539fbb52280393adc081b289ec597430a0f9031/dots.mocr%20LICENSE%20AGREEMENT), which supplements the MIT license. Among other prohibited uses, it forbids "unauthorized digitization of publications/document scanning" and requires permission for copyright-protected material. Check that your use of the benchmark data is permitted. The official code repository (commit `23f3e5612fb8066d4034d5ecfc8f33a9243533eb`) was checked for the prompt, decoding and post-processing below; no code from it is copied here. The license agreement is not downloaded with the weights.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **4.57.6**, the version of the official `requirements.txt`, with the checkpoint's own code (`trust_remote_code`) and torchvision 0.29.1 for the image processor. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580).
  - The official parser uses vLLM, or transformers with `--use_hf`. vLLM is not used, because it would need a server inside the container and its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)).
  - **flash-attn is not installed.** The model's vision code imports `flash_attn_varlen_func` from flash-attn at the top of `modeling_dots_vision.py`, and its default vision attention calls it. Building flash-attn for this torch and CUDA version would add a large compiled dependency. Instead, before loading the model, the adapter registers a module `flash_attn` that contains only this function, computed with PyTorch's scaled-dot-product attention (SDPA): each image's patches attend only to each other, with the default scale. The function call is unchanged; the result is the same up to floating-point differences. Package metadata is not registered, so transformers does not treat flash-attn as available. The language model uses SDPA attention.
  - The model code's own SDPA vision attention (`attn_implementation = "sdpa"` in the vision config) is not used. It builds a dense boolean mask over all patches, which for a page at the pixel limit (57,600 patches) would take 3.3 GB on its own.
- **Weights are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)) in an explicit download step:

  ```sh
  make dots-mocr-model                           # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/dots-mocr/dots-mocr/<revision>/` under the cache directory and checks every file against `models.sha256`. The directory name has no periods, as the model card asks for the model code. Runs need `--cache <dir>`. At start-up the adapter checks that the directory holds exactly these files, all matching: transformers imports the Python files it finds there. It records the SHA-256 of every file with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only, and transformers copies the model code to `/tmp` inside the container to import it.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-dots-mocr               # this image (no weights)
make dots-mocr-model                # the pinned weights and code, 6.1 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/dots-mocr/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id>... -o reports/dots-mocr --inspect
make dots-mocr-smoke                # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no CPU fallback, since the model code runs its vision encoder in bfloat16.

## Prompt and parameters

There is one configuration (`engine.toml`, `name = "dots-mocr"`). It uses the official `prompt_layout_all_en` prompt, the default of the official parser for documents, byte for byte; a test checks its SHA-256. The prompt asks for the box, the category and the text of every layout element, as one JSON list in reading order:

- **Categories:** `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text` and `Title`.
- **Boxes:** `[x1, y1, x2, y2]` in pixels of the image the model sees (below).
- **Text:** none for `Picture`, LaTeX for `Formula`, HTML for `Table`, and Markdown for the others. The text must be the original text, without translation.

Page headers and footers are included as their own elements; the official `_nohf` Markdown, which leaves them out, is not used. The prompt does not ask the model to normalize or modernize spelling, and the adapter does not change characters such as long s (`ſ`).

- The request is the official parser's: one user turn with the image before the prompt and no system message. The checkpoint's chat template turns it into `<|user|><|img|><|imgpad|><|endofimg|>`, the prompt, `<|endofuser|><|assistant|>`.
- The same prompt is used for full pages and for paragraph crops. The sample `language` and other metadata are not used.
- The conversation and the SHA-256 of the prompt are recorded with every sample (`extra.models.prompt`, `extra.models.prompt_sha256`).

Other official prompts are not used: `prompt_ocr` gives text without boxes and leaves out page headers and footers, and `prompt_layout_only_en` gives boxes without text. A new configuration would be a new TOML file with its own `name`.

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters that are left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>` |
| `max_new_tokens` | 24000 | Generated tokens before the output counts as truncated (the official transformers inference's limit) |

**Decoding.** As in the official transformers inference (`demo/demo_hf.py`, `parser.py --use_hf`): greedy decoding without a repetition penalty, up to 24,000 new tokens. Generation stops at any end token of the checkpoint's `generation_config.json` (`<|endoftext|>`, `<|assistant|>` and `<|endofassistant|>`), at `max_new_tokens`, or at the end of the 131,072-token context. The official vLLM parser instead samples with temperature 0.1; greedy decoding is used here so that runs can be repeated. There is no official repetition guard, and the adapter adds none.

**Image size.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, other modes converted to RGB). The checkpoint's Qwen2-VL image processor then scales it:

1. Both sides are rounded to whole 28-pixel units (14-pixel patches, merged 2 × 2), keeping the aspect ratio.
2. The image is kept between 3,136 and 11,289,600 pixels: a smaller image is enlarged and a larger one reduced, with bicubic resampling.
3. The model sees one token per 28 × 28 pixels. A 4194 × 6192-pixel newspaper scan is reduced to 2744 × 4060 pixels and 14,210 image tokens; an A4 page at 150 dpi (1240 × 1754) becomes 1232 × 1764 pixels and 2,772 image tokens.

This matches the official parser with `--no_fitz_preprocess`. Its default for image files first renders the image through PyMuPDF at 200 dpi, which can enlarge small images. That step is not reproduced: it would add PyMuPDF, and for the large scans of the benchmark the pixel limit above applies either way. The original size and the size the model saw are recorded as `source_image` and `model_image`.

## Conversion to the prediction

The adapter converts the output into the standardized prediction itself (`prediction.json`, [PREDICTIONS.md](../../docs/PREDICTIONS.md)). Every layout element becomes one region, in output order, which is the model's reading order. The prediction lists the regions in that order in `reading_order`.

- **Positions.** The box is mapped from the image the model saw to the input image, as float pixel coordinates (the official parser truncates them to integers). A box that is not four numbers with `x1 < x2` and `y1 < y2` is not used: the region has no polygon, the box is kept in the region's `extra.bbox`, and the element is counted in `extra.invalid_boxes`. There are no line positions: an element is a block of text.
- **Types.** `Title` and `Section-header` become `heading`, `Picture` becomes `figure`, `List-item`, `Page-header` and `Page-footer` become `list-item`, `page-header` and `page-footer`, and the others their lower-case names. The category is kept as `native_type`. An unknown category becomes `other`.
- **Text.** The element's text, in plain text:
  - `Table`: the HTML table becomes one line per row, with cells separated by tabs, as the host converter does.
  - `Formula`: the TeX source without `$$`, `\[ … \]` or `$` delimiters.
  - `Picture`: no text. The prompt asks for none; anything the model writes is not used.
  - Other categories: leading heading marks (`#`) of a `Title` or `Section-header` and the bullet (`-`, `*`, `+`, `•`) of a `List-item` are removed; numbers of ordered list items are kept. Inline Markdown is then converted by the [host converter's inline rules](../../docs/PREDICTIONS.md), with the same parser ([markdown-it-py](https://github.com/executablebooks/markdown-it-py) 4.2.0 and its dollar-math plugin, as pinned for the host): emphasis, links, strikethrough and code keep only their text, inline math keeps its TeX source, escapes and HTML entities are decoded, images and HTML tags add nothing (`<br>` is a line break), runs of spaces become one space and blank lines are dropped. A test compares the result with the host converter. The native `output.json` is unchanged.
- **Page text.** The region texts joined by blank lines.

**Incomplete output.** An output that is not one complete JSON list keeps the elements of its list that are complete: everything up to the last complete element. The official parser instead falls back to the raw text. `extra.json_complete` tells whether the output was a complete list, and `extra.elements` gives the number of elements used.

## Truncation, memory and time

Generation stops at the first end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- Without an end token the sample is `truncated` (runner status `truncated`). Its `message` names the reason and the number of complete elements used, and `extra.stop_reason` is `max_new_tokens` or `context` (`end-token` otherwise).
- An output that ends with an end token but is not a complete JSON list is `partial`, with the complete elements used.
- An output with no list at all (for example plain text) fails the sample (`engine-error`, `the output is not a JSON list of layout elements`); `output.json` keeps it.
- The text of the complete elements is scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)).
- **Repetition.** Greedy decoding can fall into a loop that repeats an element until the token limit. Such a sample is `truncated`, and `extra.repeated_elements` counts the elements whose category and text equal the previous element's. The repeated elements stay in the prediction, scored as produced. A loop inside the text of one element is not counted; it leaves that element incomplete, so it is not in the prediction.

**Repetition on dev-01.** Only the Glagolitic page `CkXII61-0017` looped: after one element of Georgian-script text, the second element repeated `მიმართება და` until the 24,000-token limit, so the page is `truncated` with one complete element (`repeated_elements` 0). All other pages ended with an end token.

**Memory.** Measured on an RTX 4090 (24 GB) for the dev-01 pages:

- The bfloat16 weights take about 6.1 GB of GPU memory.
- The key/value cache takes about 28 KB per token (28 layers, 2 key/value heads of 128 dimensions), so about 1.1 GB for 14,400 image tokens and 24,000 generated ones.
- The vision encoder attends over up to 57,600 patches at once.

The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`. It was 6.6–6.8 GB for the 1600-pixel book pages and the Glagolitic page, and 8.3 GB for the newspaper pages, which are reduced to the 11,289,600-pixel limit (about 14,300 image tokens). A CUDA out-of-memory error fails only that sample (`engine-error`), and the container continues.

**Time.** Generation is token by token with batch size 1. On the RTX 4090, generation ran at about 40–43 tokens/s, so a book page took 15–28 s and a newspaper page 196–270 s (8,136 and 11,516 tokens). The Glagolitic page ran to `max_new_tokens` in 553 s, within `timeouts.sample = 1800`. Checking and loading the weights took 15 s. `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample.

## Outputs

| File | Content |
| --- | --- |
| `output.json` | The generated text, decoded without special tokens and otherwise unchanged: the native output, normally a JSON list (not when it is cut off) |
| `prediction.json` | The standardized prediction with positioned regions (above): the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `source_image`, `model_image`, `json_complete`, `elements`, `skipped_elements` (list items that are not JSON objects), `invalid_boxes`, `repeated_elements`, the timings, `gpu_peak_memory_mb`, the transformers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (repository, revision, file checksums, the configured `model`, the prompt and its SHA-256) and the parameters |

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`), and the container continues. An empty list is a genuine empty result: `success` with empty text. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make dots-mocr-model and run with --cache`);
- the model files do not match `models.sha256`, or the directory holds other files.

## Tests

`tests/test_dots_mocr.py` runs offline in `make test`. It checks the following without the network:

- parameter validation and the prompt against the official SHA-256;
- the `flash_attn` import the model code needs, without the flash-attn package;
- reading complete elements from cut-off JSON;
- the plain text of every category;
- the mapping of boxes to the input image and the handling of invalid boxes;
- the model checksums and the download script;
- the engine protocol through the runner.

The engine runs with fake `torch` and `transformers` modules (`tests/fixtures/dots-mocr/fake`).

- The fake processor checks the official request (the image before the prompt), resizes like the real one with the checkpoint's limits, and counts image tokens like the real one.
- The fake model checks that it is loaded with the checkpoint's code and SDPA attention, imports `flash_attn_varlen_func` as the real model code does, checks the decoding settings (greedy, the three end tokens), and returns these outputs:
  - a hand-written layout list with every category, an HTML table and an invalid box (`tests/fixtures/dots-mocr/output.json`, not real model output);
  - an empty list (a blank page);
  - an out-of-memory error;
  - text that is not JSON;
  - a list that an end token cuts off;
  - a newspaper page whose list ends repeating one element and never closes.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

The SDPA replacement for `flash_attn_varlen_func` needs PyTorch and is exercised only by the GPU smoke test.

`make dots-mocr-smoke` (after `make dots-mocr-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page, and checks that every region has a polygon. A second run with `max_new_tokens = 40` must mark the page `truncated` and score the complete elements. A third run without `--cache` must fail explicitly.

The GPU smoke test passed on an RTX 4090. The dev-01 results are in the pull request for [#52](https://github.com/michal-hradis/ocr_benchmarking/issues/52).
