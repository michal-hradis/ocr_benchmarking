# DeepSeek-OCR-2 engine (page-image to Markdown with layout boxes, GPU)

[DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) by DeepSeek-AI ([#54](https://github.com/michal-hradis/ocr_benchmarking/issues/54)) reads a whole page image. With its document prompt it generates the page as Markdown, block by block in reading order. Each block starts with a **grounding tag**: a layout label and the block's box, `<|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>`. Tables are HTML and equations LaTeX. There is no separate detection or line recognition. The engine runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-deepseek-ocr2:2-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers 4.46.3, tokenizers 0.20.3 and the imports of the model's code, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the weights, the model's Python code, the tokenizer and the license at the pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make deepseek-ocr2-model`) |
| `deepseek_ocr2_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`; its settings are the single preset |

## Versions and license

- **Model:** [`deepseek-ai/DeepSeek-OCR-2`](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) at commit `aaa02f3811945a91062062994c5c4a3f4c0af2b0` (2026-02-03), **Apache-2.0** (`LICENSE.txt` in the repository, downloaded with the weights and checked against `models.sha256`). The [code repository](https://github.com/deepseek-ai/DeepSeek-OCR-2) is Apache-2.0 too; commit `2f3699ebbb96fa8af32212e8c170f2cc28730fad` was checked. Paper: [DeepSeek-OCR 2: Visual Causal Flow](https://arxiv.org/abs/2601.20552).
  - The vision encoder (DeepEncoder V2) is a SAM ViT-B followed by a Qwen2-0.5B decoder used as an encoder, which reorders the visual tokens before the language model. The language model is a DeepSeek-V2-style mixture of experts: 12 layers, 64 routed experts with 6 active per token and 2 shared experts.
  - The weights are 6.8 GB of bfloat16 in one `model-00001-of-000001.safetensors` file (about 3.4 billion parameters).
  - The model card says "multilingual" and names no languages. Historical prints, Fraktur, Glagolitic and Czech are not claimed.
- **Remote code.** The architecture is not part of transformers: the checkpoint ships its own Python code (`modeling_deepseekocr2.py`, `modeling_deepseekv2.py`, `deepencoderv2.py`, `configuration_deepseek_v2.py`, `conversation.py`), which transformers loads with `trust_remote_code`. These files are downloaded with the weights and checked against `models.sha256` before they are loaded. The default model directory must hold exactly these files, so no other code can be imported from it. transformers copies the code into `HF_MODULES_CACHE` (`/tmp/hf_modules` in the container) to import it.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **4.46.3** and tokenizers **0.20.3**, the versions of the model card and the official `requirements.txt`. Later transformers releases changed internals that the model's code overrides (the Qwen2 encoder's custom attention mask, the cache API), so a newer version is not used. The model's code also imports einops, addict, easydict and torchvision 0.29.1.
  - It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580), not on the torch 2.6.0 for CUDA 11.8 that the official instructions name. This combination was checked: on the synthetic smoke pages the adapter's output was identical to the official `model.infer()` in the same image (below).
  - `TORCH_DISABLE_NATIVE_JIT=1` is set. torch 2.14 would otherwise compile a Triton kernel for the rotary embedding at first use, which needs a C compiler that the image does not have. The standard ATen kernels are used instead, as in the [Surya image](../surya/README.md).
  - The model card loads the language model with flash-attention 2. It is not installed: the language model uses the code's eager attention (`_attn_implementation="eager"`), which gives the same results up to floating-point differences. The vision encoder uses PyTorch's scaled-dot-product attention, as in the official code.
  - vLLM 0.8.5, the official repository's fast backend, is not used, because it would need its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)). Its configuration is the same preset (below), but with a sliding-window n-gram block instead of transformers' `no_repeat_ngram_size`.
- **Weights are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)) in an explicit download step:

  ```sh
  make deepseek-ocr2-model                       # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/deepseek-ocr2/DeepSeek-OCR-2/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again, and it records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-deepseek-ocr2           # this image (no weights)
make deepseek-ocr2-model            # the pinned weights and code, 6.8 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/deepseek-ocr2/engine.toml \
    --dataset datasets/dev-01.json --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/deepseek-ocr2 --inspect
make deepseek-ocr2-smoke            # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. The model's code moves tensors to the GPU itself (`.cuda()`), so there is no CPU mode.

## Prompt, parameters and the preset

The adapter has **one preset**: the configuration in `engine.toml`, the official document configuration. That is the model card's `model.infer(..., base_size=1024, image_size=768, crop_mode=True)` with the document prompt, which is also the official vLLM `config.py`.

- The prompt is `<image>\n<|grounding|>Convert the document to markdown.`. The model card writes it with a trailing space, which the official `format_messages` (template `plain`) strips. `<|grounding|>` asks for the layout tags.
- The other main prompt, `<image>\nFree OCR.` (text without layout), is not used. There is no parameter for it; it would be a separate preset.
- The same prompt is used for full pages and for paragraph crops. The sample `language` and other metadata are not used.
- The prompt is recorded with every sample (`extra.models.prompt`).

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters that are left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>` |
| `model_dir` | `models/deepseek-ocr2/DeepSeek-OCR-2/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `max_new_tokens` | 8192 | Generated tokens before the output counts as truncated (`infer()` and the vLLM `max_tokens`); the context limits it further (below) |
| `no_repeat_ngram_size` | 20 | No sequence of this many tokens may occur twice in the prompt and output (`infer()`); 0 turns this off |

**Decoding.** Decoding is greedy, as `infer()` (temperature 0) and the vLLM script. Generation stops at the end token `<｜end▁of▁sentence｜>`. Like `infer()`, it runs under bfloat16 autocast and blocks every repeated 20-token sequence (`no_repeat_ngram_size=20`). This is the setting of `infer()` without `eval_mode`, the mode of the model card's example; `eval_mode` uses 35. The vLLM script blocks a repeated 20-token sequence only within the last 90 tokens and never blocks `<td>` and `</td>`; transformers has no such option. A large table with many empty cells could therefore come out differently than with vLLM.

**Custom checkpoints.** Any directory under the cache that transformers can load with the model's remote code (for example a fine-tuned DeepSeek-OCR-2) can be given as `model_dir`. Only the files are checked against `models.sha256`, never the directory name. The default directory must match exactly, or every sample fails. For another directory, the result is recorded:

- If the directory holds exactly the files in `models.sha256`, all matching, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded. Extra files count too, because the remote code could import them.

Either way, `model_dir` and the SHA-256 of every file in it are recorded. A custom directory's Python code is executed: use only code you trust. For a custom checkpoint, also set `model` in a copy of `engine.toml` to name what is used. That name is what reports show as the model.

**Image views.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, other modes converted to RGB). The official `load_image` applies the EXIF orientation; the benchmark's coordinates refer to the stored pixels ([PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)). It then builds the model's input exactly as `infer()` does, in the adapter's own code:

1. **Global view.** The whole image is scaled to fit 1024 × 1024 pixels, keeping its aspect ratio, and padded with gray (`ImageOps.pad`, bicubic). It becomes 16 × 16 = 256 image tokens and a view separator.
2. **Local tiles.** An image larger than 768 pixels on either side is also cut into tiles of 768 × 768 pixels. The grid has 2 to 6 tiles, the one whose aspect ratio is closest to the image's; of two equally close grids, the larger one is used when the image has enough pixels. The image is stretched to the grid (bicubic), not padded. Each tile becomes 12 × 12 = 144 image tokens. An A4 page gets 2 × 3 tiles, so the model reads it at 1536 × 2304 pixels. A smaller image (up to 768 × 768) gets the global view only.
3. **Prompt.** The prompt is the begin token, the image tokens (257 to 1,121) and the tokenized text after `<image>`.

The tile grid is recorded as `extra.tiles` (`null` without tiles). Small print on a dense newspaper page stretched to at most 2304 pixels may become illegible; that is the official configuration.

**Equivalence check.** On the three synthetic smoke pages, the adapter's `page.md` was byte for byte identical to the `result.mmd` written by the checkpoint's own `model.infer()` with the model card's arguments, in the engine image on an RTX 4090.

## Context, truncation and memory

The language model's context (`max_position_embeddings`) is 8,192 tokens for the prompt and the generated text together. The official vLLM script uses the same `max_model_len`. The model card's `infer()` sets `max_new_tokens=8192` without a limit for the prompt; the adapter keeps within the context instead. With the image tokens, a page can generate about 7,000 to 7,900 tokens. Generation stops at the end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- **Truncation.** Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the limit and `extra.stop_reason` set to `max_new_tokens` or `context` (`end-token` otherwise). The text produced so far is kept and scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). A model that repeats until the limit is also reported this way.
- **Dense pages.** A dense newspaper page with its layout tags can need more than the context. `max_new_tokens` cannot be raised past it. The column order is the model's; the flexible metrics score the text independently of that order.
- **Memory.** The bfloat16 weights take 6.8 GB of GPU memory. The key/value cache takes about 60 KB per token (12 layers, 10 heads of 128 dimensions for keys and values), so about 0.5 GB for the whole context. A single RTX 4090 (24 GB) is the target. The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`. On the synthetic smoke pages it was 7.0–7.2 GB. A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues. Host memory is limited to 24 GB by `[resources]`.
- **Time.** Generation is token by token with batch size 1. The mixture-of-experts layers of the model's code run expert by expert in PyTorch, and eager attention is used, so generation is slow for a model of this size: about 24–29 tokens/s on the RTX 4090 for short outputs. Checking and loading the weights took about 18 s. `timeouts.sample = 1200` and `timeouts.startup = 600` leave a margin. `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample.

## Outputs

| File | Content |
| --- | --- |
| `output.md` | The generated text up to the end token, decoded with special tokens (the layout tags are special tokens) and otherwise unchanged: the native output |
| `layout.json` | The size of the input image and the grounded blocks in order: `label`, `det` (the box text as written), `boxes` (lists of four numbers in [0, 999], or `null` when `det` is not such a list), `boxes_px` (below), the Markdown up to the next block (`text`) and the character offsets of the tag (`span`), in `output.md` stripped of surrounding white space |
| `page.md` | The official `result.mmd`: the stripped output with the layout tags removed and image blocks turned into Markdown image references: the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `blocks`, `image_blocks`, `invalid_boxes`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `tiles`, the timings, `gpu_peak_memory_mb`, the transformers/tokenizers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (`verified_official_checkpoint`; `repository` and `revision` only when verified; `model_dir`, file checksums, the configured `model` and the prompt) and the parameters |

**Markdown.** `page.md` is made as the official `infer(save_results=True)` makes `result.mmd`:

- The tag of every `image` block becomes `![](images/<n>.jpg)` and a line break. The model writes no text for an image; the reference adds none.
- The tags of all other blocks are removed; their Markdown stays. If there was such a tag, `\coloneqq` and `\eqqcolon` become `:=` and `=:`, as in the official code.
- Text outside any block stays.

**Boxes.** The model gives each box as `[x1, y1, x2, y2]` scaled to [0, 999] of the image width and height; a block can have several boxes. `boxes_px` are the same boxes in float pixel-edge coordinates of the original stored image ([PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)): x × width / 999 and y × height / 999, not clipped. The official drawing code computes the same, rounded down. The official code reads `det` with Python's `eval`; the adapter reads it as a literal only (`ast.literal_eval`), because model output is untrusted. A `det` that is not a list of four-number lists gets `boxes: null` and no `boxes_px`, and is counted in `extra.invalid_boxes`; no box is invented.

**Plain text and the prediction.** The shared Markdown converter turns `page.md` into the standardized prediction on the host ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), not in the container. Markdown and HTML markup is therefore never part of the recognized text, and a conversion fix needs no new inference:

- Headings, paragraphs and list items become typed regions with their text only. HTML tables become `table` regions with one row per line and cells separated by tabs. Images and their alt text add nothing.
- `$$` formulas keep their TeX source. Emphasis and links keep only their text.
- The converter's limitations apply. For example, the `\(…\)` and `\[…\]` equation delimiters are read as Markdown escapes, so `\( E = mc^2 \)` becomes `( E = mc^2 )`.
- The converted prediction is **text-only**: it has no positions and no physical lines. The block boxes are real model geometry, but they stay in `layout.json` and are not converted into prediction regions, as for [Chandra OCR 2](../chandra-ocr2/README.md#outputs). That would need the shared converter to return positioned regions for Markdown blocks. No text geometry is invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. An empty output is a genuine empty result: `success` with empty text. On the synthetic blank page the model wrote a sentence that is not on the page; such text is scored as produced. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make deepseek-ocr2-model and run with --cache`);
- the model files do not match `models.sha256`, or the default directory holds other files;
- the model cannot be loaded.

## Tests

`tests/test_deepseek_ocr2.py` runs offline in `make test`. It checks the following without the network:

- parameter validation and the prompt;
- the tile grid against values computed with the official code, the global view and tiles, and the prompt tokens;
- the layout blocks, including boxes that are not valid literals, and the official Markdown post-processing;
- the model checksums, extra files in the model directory and the download script;
- the engine protocol through the runner.

The engine runs with fake `torch`, `torchvision` and `transformers` modules (`tests/fixtures/deepseek-ocr2/fake`).

- The fake model checks the prompt tokens and the image mask against the views and tiles, and the decoding settings. It returns these outputs:
  - a hand-written page with layout tags, an image block and an HTML table (`tests/fixtures/deepseek-ocr2/output.md`, not real model output);
  - an empty output (a blank page);
  - text without layout tags;
  - an out-of-memory error;
  - a newspaper page that never ends, cut at the context.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

`make deepseek-ocr2-smoke` (after `make deepseek-ocr2-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page. A second run with `max_new_tokens = 32` must mark the page `truncated` and score what was generated. A third run without `--cache` must fail explicitly.
