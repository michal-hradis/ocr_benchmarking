# olmOCR-2 engine (page-image to Markdown, GPU)

[olmOCR-2](https://huggingface.co/allenai/olmOCR-2-7B-1025) by Ai2 ([#56](https://github.com/michal-hradis/ocr_benchmarking/issues/56)) reads a whole page image. It generates a short YAML **front matter** (language, rotation, table and diagram flags) followed by the page's text as Markdown, with tables as HTML and equations as LaTeX. There is no separate detection or line recognition. The engine runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-olmocr2:7B-1025-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers, accelerate and their dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `prompt.txt` | The official instruction, byte for byte |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make olmocr2-model`) |
| `olmocr2_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`; its settings are the single preset |

## Versions and license

- **Model:** [`allenai/olmOCR-2-7B-1025`](https://huggingface.co/allenai/olmOCR-2-7B-1025) at commit `e52d6f090b7a9007afffbbd6ce510876222fea93` (2025-10-22), **Apache-2.0**. The model card adds that it is intended for research and educational use under Ai2's [Responsible Use Guidelines](https://allenai.org/responsible-use).
  - It is [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (`Qwen2_5_VLForConditionalGeneration`) fine-tuned on [olmOCR-mix-1025](https://huggingface.co/datasets/allenai/olmOCR-mix-1025) and then trained with GRPO reinforcement learning ([paper](https://olmocr.allenai.org/papers/olmocr.pdf), [code](https://github.com/allenai/olmocr)).
  - The weights are 16.6 GB of bfloat16 in four `model-0000N-of-00004.safetensors` files. This is the full-precision release. The model card recommends the [FP8 version](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) for vLLM. FP8 needs vLLM's kernels and is not used here.
  - The model card lists English only. The training data is mostly English PDFs. Historical prints, Fraktur, Glagolitic and Czech are not claimed.
- **Official pipeline:** the [`olmocr`](https://pypi.org/project/olmocr/) toolkit; version **0.4.27** (commit `1e139a5ea61f1668164e6b63357d7284a8391615`) was checked. It is not installed: it runs a vLLM server and renders pages from PDFs with poppler. The adapter reproduces what the toolkit sends to the model on a page's first attempt: the prompt, the image size and the decoding (below). The toolkit's retries and rotation correction are not reproduced (below).
- **Inference:** [transformers](https://github.com/huggingface/transformers) **5.19.0** (`Qwen2_5_VLForConditionalGeneration`, the architecture in the checkpoint's `config.json`), with torchvision 0.29.1 for the image processor and accelerate 1.15.0 for loading. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580).
  - vLLM, the toolkit's backend, is not used, because it would need a server inside the container and its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)).
  - The checkpoint's `generation_config.json` asks for flash-attention 2. It is not installed: attention uses PyTorch's scaled-dot-product kernels, which give the same results up to floating-point differences.
- **Weights are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)) in an explicit download step:

  ```sh
  make olmocr2-model                             # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/olmocr2/olmOCR-2-7B-1025/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again, and it records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-olmocr2                 # this image (no weights)
make olmocr2-model                  # the pinned weights, 16.6 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/olmocr2/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/olmocr2 --inspect
make olmocr2-smoke                  # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` set explicitly, the model runs in float32 on the CPU. That is very slow, needs about 33 GB of host memory (more than `memory` in `[resources]`), and is not the configuration recorded here.

## Prompt, parameters and the preset

The adapter has **one preset**: the configuration in `engine.toml`, which is the toolkit's first attempt at a page.

- `prompt.txt` holds `build_no_anchoring_v4_yaml_prompt()` of `olmocr` 0.4.27 byte for byte, and a test checks its SHA-256. This is the prompt the model card names for olmOCR-2. It has no anchor text (text extracted from a PDF), because an image has none.
- The prompt asks for plain text read naturally, LaTeX equations, HTML tables, a Markdown image reference for every figure, and the front matter.
- The instruction comes **before** the image in one user turn, as in the toolkit. The checkpoint's chat template adds its default system message (`You are a helpful assistant.`), as vLLM does with the same template.
- The same prompt is used for full pages and for paragraph crops. There is no prompt parameter, and the sample `language` and other metadata are not used.
- The conversation and the SHA-256 of the instruction are recorded with every sample (`extra.models.prompt`, `extra.models.prompt_sha256`).

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters that are left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `model_dir` | `models/olmocr2/olmOCR-2-7B-1025/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `target_longest_image_dim` | 1288 | The image is scaled so that its longest side has this many pixels (the toolkit's `--target_longest_image_dim`); at least 56 |
| `max_new_tokens` | 8000 | Generated tokens before the output counts as truncated (`MAX_TOKENS` of the toolkit's page query) |
| `temperature` | 0.1 | Sampling temperature of the toolkit's first attempt; 0 is greedy decoding |
| `repetition_penalty` | 1.05 | From the checkpoint's `generation_config.json`, which vLLM applies by default |
| `seed` | 0 | Set before every sample, so a sample's sampling does not depend on the samples before it |

**Decoding.** The toolkit sends each page with temperature 0.1 and `max_tokens` 8000 and sets no other sampling parameter. Its vLLM server then uses the checkpoint's `generation_config.json` for the rest: a repetition penalty of 1.05, applied to the prompt and the generated tokens alike, and no top-p or top-k truncation. The adapter does the same with transformers' sampling: `top_p` 1 and `top_k` 0, because transformers would otherwise default `top_k` to 50. Generation stops at `<|im_end|>` or `<|endoftext|>` (`generation_config.json`).

- The model card's transformers example also samples at temperature 0.1.
- Unlike the other engines' greedy defaults, the result therefore depends on the seed. Sampling with a fixed seed repeats on the same GPU, driver and software versions. Different GPUs or kernels can still change the text. `temperature = 0` gives greedy decoding, which is deterministic except for floating-point differences between GPUs; that is a different configuration and would need its own development-set results.

**Not reproduced from the toolkit.** The toolkit judges each attempt and tries again (up to `--max_page_retries`, 8 by default) with temperatures rising from 0.1 to 1.0:

- **Retries.** It retries a page when generation does not end with an end token, when the prompt and output exceed 16,384 tokens, or when the front matter cannot be parsed. After the last attempt it falls back to `pdftotext`, which an image does not have.
- **Rotation.** It retries with the image turned when the front matter says `is_rotation_valid: false`, by the reported `rotation_correction`.

The adapter makes one attempt per page. A truncated output is reported as `truncated`, a missing or invalid front matter is recorded (below), and the image is never rotated. Results with retries could differ, would take up to eight times longer, and would not be deterministic. The front matter of every sample is recorded, so pages that the toolkit would have rotated or retried can be found in the results.

**Custom checkpoints.** Any directory under the cache that transformers can load as a Qwen2.5-VL model (for example a fine-tuned olmOCR) can be given as `model_dir`. Only the files are checked against `models.sha256`, never the directory name. The default directory must match, or every sample fails. For another directory, the result is recorded:

- If every file in `models.sha256` is present and matches, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded.

Either way, `model_dir` and the SHA-256 of every file in it are recorded. For a custom checkpoint, also set `model` in a copy of `engine.toml` to name what is used, for example `"my-org/olmOCR-finetuned@<revision>"`. That name is what reports show as the model.

**Image size.** The model was trained and is used on pages rendered with a longest side of 1288 pixels. The toolkit accepts PNG and JPEG files only by wrapping them into a PDF (`img2pdf`) and rendering that page with `pdftoppm` at the resolution that gives 1288 pixels. The adapter reads the raster image directly instead:

1. **Scale.** It decodes the image as stored with Pillow (EXIF orientation not applied, other modes converted to RGB). It scales it with Lanczos resampling so that its longest side is `target_longest_image_dim` pixels, keeping the aspect ratio, with the other side rounded to whole pixels. Larger scans are reduced, and small pages and paragraph crops are enlarged, as the toolkit's rendering does. The resampling filter is Pillow's, not poppler's, so pixels can differ slightly from the toolkit's.
2. **Processor.** The checkpoint's Qwen2-VL image processor rounds both sides to whole 28-pixel units (14-pixel patches, merged 2 × 2), one image token each, within its limits of 3,136 to 12,845,056 pixels. A 1288-pixel side is exactly 46 units. An A4 page (aspect 1.41) is scaled to 911 × 1288 pixels and seen by the model as 924 × 1288, 1,518 image tokens.

The original size and the size the model saw are recorded as `source_image` and `model_image`. Small print on a dense newspaper page scaled to 1288 pixels may become illegible; that is the official configuration. `target_longest_image_dim` can be raised, but the model was not trained for larger images and such a run is a different configuration.

## Front matter, truncation and memory

**Front matter.** The adapter splits the output as the toolkit's `FrontMatterParser` does. A front matter starts the output with `---` and ends at the next line starting with `---`. It is parsed as YAML and stored as `extra.front_matter`; the Markdown after it is the text.

- With no front matter, or invalid YAML, the whole output is the text and `extra.front_matter_error` says why. The status does not change, because the text is still the model's transcription.
- The fields must be exactly `primary_language`, `is_rotation_valid`, `rotation_correction`, `is_table` and `is_diagram`. Missing or unexpected fields are recorded in `front_matter_error`, and the fields that were present are kept in `front_matter`. The toolkit would retry in all of these cases.
- Values are stored as YAML reads them. For example, a `primary_language` of `no` (Norwegian) is YAML's `false`.

**Context and truncation.** The model's context is 128,000 tokens for the prompt and the generated text together. With 1288-pixel images, a page needs at most about 2,200 image tokens, so `max_new_tokens` is the limit in practice. Generation stops at the first end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the limit and `extra.stop_reason` set to `max_new_tokens` or `context` (`end-token` otherwise).
- The text produced so far is kept and scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). A model that repeats a line until the limit is also reported this way.
- A page with 2,000+ lines can need more than 8,000 tokens and is then truncated. `max_new_tokens` can be raised, together with `timeouts.sample`.
- The column order is the model's. The flexible metrics score the text independently of that order.

**Memory.** These are estimates from the model's size; they have not been measured yet.

- The bfloat16 weights take 16.6 GB of GPU memory. accelerate loads them directly onto the GPU, so host memory (limited to 24 GB by `[resources]`) does not hold a full copy.
- The key/value cache takes about 56 KB per token (28 layers, 4 key/value heads of 128 dimensions), so about 0.6 GB for 2,200 image tokens and 8,000 generated ones.
- The vision encoder (32 layers, mostly windowed attention) runs on up to about 8,500 patches.
- A single RTX 4090 (24 GB) is the target. The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`. A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues.

**Time.** Generation is token by token with batch size 1, for a 7-billion-parameter model. A truncated sample generates all 8,000 tokens and is the slowest. `timeouts.sample = 1200` assumes at least about 7 tokens/s, and `timeouts.startup = 600` allows for checking and loading 16.6 GB of weights. `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample.

## Outputs

| File | Content |
| --- | --- |
| `output.md` | The generated text, decoded without special tokens and otherwise unchanged: front matter and Markdown, the native output |
| `page.md` | The Markdown after the front matter, stripped of surrounding white space, as the toolkit's `natural_text`: the primary output |
| `result.json` | Status, processing time and, in `extra`: `front_matter` (or `null`) and `front_matter_error` (only when there is a problem), `stop_reason`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `source_image`, `model_image`, the timings, `gpu_peak_memory_mb`, the transformers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (`verified_official_checkpoint`; `repository` and `revision` only when verified; `model_dir`, file checksums, the configured `model`, the prompt and its SHA-256) and the parameters |

**Plain text and the prediction.** `page.md` is converted to the standardized prediction on the host by the shared Markdown converter ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), not in the container. Markdown and HTML markup is therefore never part of the recognized text:

- Headings, paragraphs and list items become typed regions with their text only.
- HTML tables (`<table>`, as the prompt asks) become `table` regions with one row per line and cells separated by tabs.
- **Figures add no text.** The prompt asks for a Markdown image reference with a description of each figure as its alt text (`![…](page_x_y_w_h.png)`). The converter drops images and their alt text, so the description is not scored. The reference stays in `page.md` and `output.md`. Its box is in the model's own pixel coordinates of the scaled image and is not used.
- `$$` formulas keep their TeX source. Emphasis and links keep only their text.
- The converter's limitations apply. For example, the `\(…\)` and `\[…\]` equation delimiters that the model writes are read as Markdown escapes, so `\( E = mc^2 \)` becomes `( E = mc^2 )`.

The converted prediction has **no positions** and no physical lines. Line breaks inside a paragraph are kept as the model wrote them, and the regions are the Markdown blocks in the model's order. The model gives no text geometry, so none is invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. A front matter without text (`natural_text` empty) is a **genuine empty result**: `success` with empty text. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make olmocr2-model and run with --cache`);
- the model files do not match `models.sha256`.

## Tests

`tests/test_olmocr2.py` runs offline in `make test`. It checks the following without the network:

- parameter validation, and the prompt against the official SHA-256;
- image scaling to the longest side, including enlarging small images;
- the front matter split, for valid, missing and invalid front matters and wrong fields;
- the model checksums and the download script;
- the engine protocol through the runner.

The engine runs with fake `torch` and `transformers` modules (`tests/fixtures/olmocr2/fake`).

- The fake processor checks that the instruction precedes the image, and resizes like the real one with the checkpoint's limits.
- The fake model checks the decoding settings (sampling without top-p or top-k truncation, or greedy, with a repetition penalty) and both end tokens. It returns these outputs:
  - a hand-written page with a front matter, a figure reference and an HTML table (`tests/fixtures/olmocr2/output.md`, not real model output);
  - a front matter only (a blank page);
  - text without a front matter;
  - an out-of-memory error;
  - a newspaper page that never ends.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

`make olmocr2-smoke` (after `make olmocr2-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page. A second run with `max_new_tokens = 64` must mark the page `truncated` and score what was generated. A third run without `--cache` must fail explicitly.

**Not yet run:** the GPU smoke test and the development-set evaluation with the real weights. GPU memory, speed and accuracy are therefore not reported here yet.
