# HunyuanOCR-1.5 engine (page-image to text/Markdown, GPU)

[HunyuanOCR-1.5](https://huggingface.co/tencent/HunyuanOCR) by Tencent Hunyuan ([#55](https://github.com/michal-hradis/ocr_benchmarking/issues/55)) is a lightweight end-to-end OCR vision-language model (about 1 billion parameters). It reads a whole page image and generates its text in reading order, as Markdown with HTML tables and LaTeX formulas, or as plain text, depending on the task prompt. There is no separate detection or line recognition. The engine runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-hunyuanocr:1.5-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers, accelerate and their dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make hunyuanocr-model`) |
| `hunyuanocr_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`: the `doc_parse` preset |
| `presets/text.toml` | The `text` preset (`structured_parse` prompt) |

## Versions and license

- **Model:** [`tencent/HunyuanOCR`](https://huggingface.co/tencent/HunyuanOCR) at commit `47644ecc4fc854efa4f505155158831f36773ee4` (2026-08-28). Since 2026-07-06 the repository root holds **HunyuanOCR-1.5**. The previous HunyuanOCR-1.0 is archived under `v1.0/`, and the DFlash speculative-decoding draft model is under `dflash/`; neither is downloaded or used. Paper: [HunyuanOCR-1.5: Making Lightweight OCR VLMs Faster and Better](https://arxiv.org/abs/2607.04884).
  - The architecture is `HunYuanVLForConditionalGeneration`: a 27-layer vision transformer with 16-pixel patches (merged 2 × 2 into one token) and a 24-layer, 1024-wide language model with a 131,072-token context.
  - The weights are 2.24 GB of bfloat16 in one `model.safetensors` file. Every file is checked against `models.sha256`.
  - The model card lists multilingual, English and Chinese. Version 1.5 adds training data for low-resource and ancient-script OCR. Historical European prints, Fraktur, Glagolitic and Czech are not specifically claimed.
- **License:** [Tencent Hunyuan Community License Agreement](https://huggingface.co/tencent/HunyuanOCR/blob/47644ecc4fc854efa4f505155158831f36773ee4/LICENSE). The official [code repository](https://github.com/Tencent-Hunyuan/HunyuanOCR) (commit `1ef4179e53cf860f7e6fd8276a292e5d05b3e927`, checked for the prompts, decoding and post-processing below) is under the same license; no code from it is copied here.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **5.19.0**, which includes `HunYuanVLForConditionalGeneration` and its processor (the model card asks for ≥ 5.13.0), with torchvision 0.29.1 for the image processor and accelerate 1.15.0 for loading. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580).
  - The official clients use vLLM (optionally with DFlash speculative decoding) or transformers. vLLM is not used, because it would need a server inside the container and its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)). The official documentation states that vLLM, DFlash and transformers give aligned results.
  - The official transformers script uses eager attention by default and recommends flash-attention 2, which is not installed. The adapter uses PyTorch's scaled-dot-product attention (SDPA), which gives the same results up to floating-point differences. Eager attention is not usable here: the vision encoder attends over all patches of a page at once, up to 65,536 for a 4096 × 4096-pixel image, and eager attention would need far more memory than a 24 GB GPU has.
- **Weights are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)) in an explicit download step:

  ```sh
  make hunyuanocr-model                          # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/hunyuanocr/HunyuanOCR/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again, and it records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-hunyuanocr              # this image (no weights)
make hunyuanocr-model               # the pinned weights, 2.2 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/hunyuanocr/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench run start --engine engines/hunyuanocr/presets/text.toml \
    --dataset datasets/dev-01.json --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id>... -o reports/hunyuanocr --inspect
make hunyuanocr-smoke               # opt-in: build the image, run both presets on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` set explicitly, the model runs in float32 on the CPU. That is slow and is not the configuration recorded here.

## Presets, prompts and parameters

The model is steered by a task prompt. The official clients only accept prompts from a fixed list of task types (`inference/utils/tasks.py`), so that users cannot weaken the model with their own wording. Two of them transcribe a page, and each is a preset. Both prompts are in Chinese, as published, and are used byte for byte; tests check their SHA-256.

| Preset | `name` | `task` | Prompt (translation) | Purpose |
| --- | --- | --- | --- | --- |
| `doc_parse` (`engine.toml`) | `hunyuanocr` | `doc_parse` | "Extract all information of the main text of the document image as Markdown, ignoring page headers and footers; express tables in HTML and formulas in LaTeX; parse in reading order." | The official default for document parsing, named on the model card |
| `text` (`presets/text.toml`) | `hunyuanocr-text` | `structured_parse` | "Extract the text in the image." | The official task for non-document text, "ancient texts, street scenes and the like"; it does not ask to leave anything out |

- **Headers and footers.** `doc_parse` asks the model to leave out page headers and footers (running heads, page numbers). The reference transcriptions contain them, so text the model leaves out counts as deletions. The `text` preset asks for all text, but without the Markdown and table structure of `doc_parse`.
- **Historical spelling.** Neither prompt asks the model to normalize or modernize spelling. The output is scored as generated: the adapter does not change characters such as long s (`ſ`).
- The request is the official clients': an empty system message, then one user turn with the image before the prompt. The checkpoint's chat template turns it into `<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>`, the image placeholder, the prompt and `<｜hy_User｜>`.
- The same prompt is used for full pages and for paragraph crops. The sample `language` and other metadata are not used.
- The conversation and the SHA-256 of the prompt are recorded with every sample (`extra.models.prompt`, `extra.models.prompt_sha256`).

A new preset is a new TOML file with its own `name`; a changed setting is a new preset, not an edit of an existing one.

`[params]` in the TOML files are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters that are left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `model_dir` | `models/hunyuanocr/HunyuanOCR/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `task` | `doc_parse` | `doc_parse` or `structured_parse` (above) |
| `max_new_tokens` | 32768 | Generated tokens before the output counts as truncated (the official clients' `--max-tokens`) |
| `repetition_penalty` | 1.08 | The official clients' repetition penalty |
| `repetition_stop` | `true` | Stop generation when the output repeats itself at the end, as the official clients do (below) |

**Decoding.** As in the official clients: greedy decoding (temperature 0), a repetition penalty of 1.08, and up to 32,768 new tokens. Generation stops at either end token of the checkpoint's `generation_config.json` (`<｜hy_Assistant｜>`, id 120007, and `<｜hy_place▁holder▁no▁2｜>`, id 120020), at `max_new_tokens`, or at the end of the 131,072-token context.

**Repetition stop and clean-up.** Greedy decoding can fall into a loop that repeats a short piece of text. The official clients guard against it in two steps, and the adapter implements both in its own code with the same settings:

1. **Early stop.** During generation the output is decoded every 64 tokens. From 4,000 characters on, and then after every further 1,000 characters, the last 8,000 characters are checked: if they end with a non-blank piece of at most 256 characters repeated 8 times in a row, generation stops. `repetition_stop = false` turns this off (the official `--no-stream`).
2. **Clean-up.** In an output of at least 2,000 characters, a piece repeated at the end at least 10 times in a row is cut to one copy. The cut is applied to `page.md` only; `output.md` keeps the full output and `extra.trimmed_chars` counts the removed characters.

A sample stopped by the early stop is reported as `truncated` (below): the page was not finished.

**Not reproduced from the official clients.** For `doc_parse`, the official clients also apply `normalize_doc_parse_markdown`, ten rewrites that align the Markdown with the conventions of the OmniDocBench benchmark: splitting multi-line formula environments, moving table captions out of tables, wrapping arithmetic lines in `$$`, removing equation numbers and layout coordinate pairs such as `(39,337),(512,900)` at the end of lines, and similar. They target OmniDocBench's formula and table scoring, not text recognition, and are not applied. If the model writes such coordinate pairs, they stay in the text and are scored as produced.

**Custom checkpoints.** Any directory under the cache that transformers can load as a HunYuanVL model (for example a fine-tuned HunyuanOCR) can be given as `model_dir`. Only the files are checked against `models.sha256`, never the directory name. The default directory must match, or every sample fails. For another directory, the result is recorded:

- If every file in `models.sha256` is present and matches, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded.

Either way, `model_dir` and the SHA-256 of every file in it are recorded. For a custom checkpoint, also set `model` in a copy of the TOML file to name what is used, for example `"my-org/HunyuanOCR-finetuned@<revision>"`. That name is what reports show as the model.

**Image size.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, other modes converted to RGB), as the official clients send the file. The checkpoint's image processor (the PIL backend, the official `use_fast=False`) then scales it:

1. Both sides are rounded to whole 32-pixel units (16-pixel patches, merged 2 × 2), keeping the aspect ratio.
2. The image is kept between 262,144 (512²) and 16,777,216 (4096²) pixels: a smaller image is enlarged and a larger one reduced, with bicubic resampling.
3. The model sees one token per 32 × 32 pixels, one more per row of units, and two more. A 4194 × 6192-pixel newspaper scan is reduced to 3360 × 4960 pixels and 16,432 image tokens; an A4 page at 150 dpi (1240 × 1754) becomes 1248 × 1760 pixels and 2,202 image tokens.

The original size and the size the model saw are recorded as `source_image` and `model_image`.

## Truncation, memory and time

Generation stops at the first end token, when the output repeats itself, at `max_new_tokens`, or at the end of the context, whichever comes first.

- Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the reason and `extra.stop_reason` set to `repetition`, `max_new_tokens` or `context` (`end-token` otherwise).
- The text produced so far is kept and scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). For a `repetition` stop, `page.md` has the repeated tail cut to one copy (above).
- The column order is the model's. The flexible metrics score the text independently of that order.

**Repetition on dev-01.** The early stop ended several pages, all of them reported as `truncated`:

- With `doc_parse`, both Czech newspaper pages ended after 3,136 and 3,520 tokens. The model transcribed the masthead and the first articles, then repeated a short phrase (for example `ale bylo mimo,`) until the stop; neither page reached its later columns.
- With `text`, the first newspaper page ended the same way after 3,776 tokens. The second one finished with an end token after 4,629 tokens, but its text covers only part of the page (29 regions against 484 reference lines).
- On the Glagolitic page, `doc_parse` wrote a LaTeX formula of `X` and `dX` and `text` a row of Roman numeral symbols (`Ⅸ ⅩⅨ ⅩⅩ …`). Both repeated it until the stop, and the clean-up cut it to one copy. The model does not read Glagolitic.

**Memory.** Measured on an RTX 4090 (24 GB) for the dev-01 pages:

- The bfloat16 weights take about 2.2 GB of GPU memory. accelerate loads them directly onto the GPU.
- The key/value cache takes about 96 KB per token (24 layers, 8 key/value heads of 128 dimensions), so about 4.7 GB for 16,400 image tokens and 32,768 generated ones.
- The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`. It was 2.5–2.8 GB for the 1600-pixel book pages and the Glagolitic page, and 4.5 GB for the newspaper pages, which are reduced to the 16,777,216-pixel limit (about 16,400 image tokens). A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues.

**Time.** Generation is token by token with batch size 1. On the RTX 4090, generation ran at about 35–38 tokens/s, so a book page took 8–27 s and a newspaper page 92–128 s. The Glagolitic page took 73 s (`doc_parse`) and 158 s (`text`) until the repetition stop. A page that runs to `max_new_tokens` (32,768 tokens) would take about 15 minutes at that speed, within `timeouts.sample = 1800`. Checking and loading the weights took 12–13 s. `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample.

## Outputs

| File | Content |
| --- | --- |
| `output.md` | The generated text, decoded without special tokens and otherwise unchanged: the native output |
| `page.md` | The output with a repeated tail cut to one copy and surrounding white space stripped: the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `trimmed_chars`, `source_image`, `model_image`, the timings, `gpu_peak_memory_mb`, the transformers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (`verified_official_checkpoint`; `repository` and `revision` only when verified; `model_dir`, file checksums, the configured `model`, the prompt and its SHA-256) and the parameters |

**Plain text and the prediction.** `page.md` is converted to the standardized prediction on the host by the shared Markdown converter ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), not in the container, for both presets. Markdown and HTML markup is therefore never part of the recognized text:

- Headings, paragraphs and list items become typed regions with their text only. Plain text from the `text` preset becomes paragraphs at blank lines.
- HTML tables (`<table>`) become `table` regions with one row per line and cells separated by tabs.
- `$$` formulas keep their TeX source; `$…$` keeps the formula without the dollar signs. Emphasis and links keep only their text. The converter's limitations apply.

The converted prediction has **no positions** and no physical lines. Line breaks inside a paragraph are kept as the model wrote them, and the regions are the Markdown blocks in the model's order. The model can output text boxes only with other task prompts (spotting), which are not used, so no geometry is invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. An empty output is a genuine empty result: `success` with empty text. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make hunyuanocr-model and run with --cache`);
- the model files do not match `models.sha256`.

## Tests

`tests/test_hunyuanocr.py` runs offline in `make test`. It checks the following without the network:

- parameter validation, both presets, and the prompts against the official SHA-256;
- the repetition detection and the clean-up of a repeated tail;
- reading the image as stored;
- the model checksums and the download script;
- the engine protocol through the runner.

The engine runs with fake `torch` and `transformers` modules (`tests/fixtures/hunyuanocr/fake`).

- The fake processor checks the official request (system message, image, prompt), resizes like the real one with the checkpoint's limits, and counts image tokens like the real one.
- The fake model checks the decoding settings (greedy, with a repetition penalty), generates one token at a time through the stopping criteria, and returns these outputs:
  - a hand-written Markdown page with an HTML table (`tests/fixtures/hunyuanocr/output.md`, not real model output);
  - an empty output (a blank page);
  - an out-of-memory error;
  - a page that ends repeating one line, which the early stop must end (and, with `repetition_stop = false`, `max_new_tokens`);
  - a newspaper page that never ends.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

`make hunyuanocr-smoke` (after `make hunyuanocr-model`) builds the image and runs the real model on the GPU, once per preset. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page. A second run with `max_new_tokens = 16` must mark the page `truncated` and score what was generated. A third run without `--cache` must fail explicitly.

The GPU smoke test passed for both presets on an RTX 4090. The dev-01 results are in the pull request for [#55](https://github.com/michal-hradis/ocr_benchmarking/issues/55).
