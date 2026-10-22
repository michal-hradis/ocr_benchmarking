# FireRed-OCR engine (end-to-end image-to-Markdown, GPU)

Second direct page-image model ([#15](https://github.com/michal-hradis/ocr_benchmarking/issues/15)), after [LightOnOCR-2](../lightonocr/README.md): [FireRed-OCR](https://huggingface.co/FireRedTeam/FireRed-OCR) reads a whole page or paragraph image and generates its content as Markdown, with tables as HTML and formulas as LaTeX. There is no separate detection or line recognition. It runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-fireredocr:2B-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `prompt.txt` | The official instruction, byte for byte |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make fireredocr-model`) |
| `fireredocr_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions

- **Model:** [`FireRedTeam/FireRed-OCR`](https://huggingface.co/FireRedTeam/FireRed-OCR) at commit `7bf5e9d91d3bb6bf91a3b6bc66d163096b696e50`, Apache-2.0 for the weights and the code ([GitHub](https://github.com/FireRedTeam/FireRed-OCR), [technical report](https://arxiv.org/abs/2603.01840)).
  - It is FireRed-OCR-2B, the only released model: [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) fine-tuned for document parsing.
  - The weights are 4.3 GB of bfloat16 (`model.safetensors`).
  - The model card lists no languages. Its benchmarks (OmniDocBench, FireRedBench) are mostly English and Chinese documents.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **5.19.0** (`Qwen3VLForConditionalGeneration`, the architecture in the checkpoint's `config.json`), with torchvision 0.29.1 for the image processor. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580).
  - The requirements are the same as LightOnOCR's. The image is separate, so either engine can change versions on its own.
  - The official repository has a transformers script and a vLLM script. vLLM is faster for large batches. It is not used here, because it would need a server inside the container and its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)).
  - The model card recommends flash-attention 2. It is not installed: attention uses PyTorch's scaled-dot-product kernels, which give the same results up to floating-point differences.
- **Weights are not in the image.** At 4.3 GB they go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)), downloaded by an explicit step:

  ```sh
  make fireredocr-model                          # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/fireredocr/FireRed-OCR/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again and records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-fireredocr              # this image (no weights)
make fireredocr-model               # the pinned weights, 4.3 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/fireredocr/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/fireredocr --inspect
make fireredocr-smoke               # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` set explicitly, the model runs in float32 on the CPU; that is very slow and not the configuration recorded here.

## Prompt and parameters

The prompt is the **official instruction**: `PROMPT` in `conv_for_infer.py` of the FireRed-OCR repository at commit `9a368c297c8e54ab3c4c66f403a5d3b4e792552c`, which both official inference scripts use. `prompt.txt` holds it byte for byte, including its indentation, and a test checks its SHA-256.

- The instruction asks for Markdown, LaTeX formulas, HTML tables and ignored figures.
- In the upstream source, the backslashes of the formula delimiters became commas (`with,(,)`, `,[,frac{-b,pm,...`). The text is kept exactly as published, because it is what the model was released with.
- The chat template puts the image and then the instruction into one user turn, with no system message.
- The same prompt is used for full pages and for paragraph crops. There is no prompt parameter, and the sample `language` and other metadata are not used.
- The conversation and the SHA-256 of the instruction are recorded with every sample (`extra.models.prompt`, `extra.models.prompt_sha256`).

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `model_dir` | `models/fireredocr/FireRed-OCR/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `max_new_tokens` | 8192 | Generated tokens before the output counts as truncated, as in the official scripts and the model card |
| `temperature` | 0 | 0 is greedy decoding, as in the official vLLM script; above 0, sampling with `top_p` and `top_k` |
| `top_p`, `top_k` | 0.8, 20 | Used only when sampling; the values from the model's `generation_config.json` |
| `seed` | 0 | Set before every sample, so a sample's sampling does not depend on the samples before it |
| `max_pixels` | 16,777,216 | The processor scales larger images down to at most this many pixels (the checkpoint's `preprocessor_config.json`); at least 65,536 |

**Decoding.** The model's `generation_config.json` samples (temperature 0.7), and so does the transformers example in the model card. The official vLLM script, which processes directories of pages as a benchmark would, uses greedy decoding (temperature 0) with 8,192 tokens. The adapter uses greedy decoding, because the result then does not depend on a seed. The end tokens are those of `generation_config.json` (`<|im_end|>`, `<|endoftext|>`). No repetition penalty is applied.

**Custom checkpoints.** Any directory under the cache that transformers can load as a Qwen3-VL model (for example a fine-tuned FireRed-OCR) can be given as `model_dir`. Only the files are checked against `models.sha256`, never the directory name. The default directory must match, or every sample fails. For another directory, the result is recorded:

- If every file in `models.sha256` is present and matches, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded.

Either way, `model_dir` and the SHA-256 of every file in it are recorded. For a custom checkpoint, also set `model` in a copy of `engine.toml` to name what is used, for example `"my-org/FireRed-OCR-finetuned@<revision>"`. That name is what reports show as the model.

**Image size.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, modes other than RGB converted to RGB) and passes it to the processor unchanged. The processor works in units of 32 × 32 pixels, one image token each (16-pixel patches, merged 2 × 2):

- It keeps the aspect ratio and rounds both sides to whole units.
- It scales an image down if it has more than `max_pixels` pixels, and up if it has fewer than 65,536. A small paragraph crop is therefore enlarged slightly, as by the official scripts.
- The default limit is 16,384 image tokens. An A4 page scanned at 300 dpi (2480 × 3508) is about 8,500 image tokens, at full resolution.
- The size the model saw is recorded as `model_image`.

## Context, truncation and memory

The model's context is 262,144 tokens for the prompt (mostly image tokens) and the generated text together, far more than a page needs. In practice `max_new_tokens`, memory and time are the limits. Generation stops at the first end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- **Truncation.** Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the limit and `extra.stop_reason` set to `max_new_tokens` or `context` (`end-token` otherwise). The Markdown produced so far is kept and scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). A model that repeats a line until the limit is also reported this way.
- **Dense and multi-column pages.** A newspaper page is read in one pass in the model's own reading order, and up to `max_pixels` it keeps its resolution.
  - A page with 2,000+ lines can need more than 8,192 tokens. It is then truncated and scored against every reference line, so the missing text counts as deletions in the strict and flexible metrics.
  - `max_new_tokens` can be raised, together with `timeouts.sample`.
  - The column order is the model's. The flexible metrics score the text independently of that order.
  - Tiling the page inside the engine is a possible follow-up, not part of this adapter.
- **Memory.** The bfloat16 weights take 4.3 GB of GPU memory.
  - The key/value cache takes about 112 KB per token (28 layers, 8 key/value heads of 128 dimensions). That is about 2.7 GB for 16,384 image tokens and 8,192 generated ones.
  - The vision encoder (24 layers) runs on all patches of the image at once, up to 65,536 at the default `max_pixels`.
  - A single RTX 4090 (24 GB) is the target. The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`.
  - A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues. Lowering `max_pixels` reduces the memory needed. Host memory is limited to 16 GB by `[resources]`.
- **Time.** Generation is token by token with batch size 1. A truncated sample generates all `max_new_tokens` and is the slowest; `timeouts.sample = 900` seconds allows for that. `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample.

## Outputs

| File | Content |
| --- | --- |
| `page.md` | The generated text, decoded without special tokens and otherwise unchanged: the native output and the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `model_image`, the timings, `gpu_peak_memory_mb`, the transformers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (`verified_official_checkpoint`; `repository` and `revision` only when verified; `model_dir`, file checksums, the configured `model`, the prompt and its SHA-256) and the parameters |

**Plain text and the prediction.** `page.md` is converted to the standardized prediction on the host by the shared Markdown converter ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), not in the container. Markdown and HTML markup is therefore never part of the recognized text:

- Headings, paragraphs and list items become typed regions with their text only.
- HTML tables (`<table>`, as the prompt asks) become `table` regions with one row per line and cells separated by tabs.
- `$$` formulas keep their TeX source. Emphasis, links and images keep only their text.
- The converter's limitations apply. For example, the `\(…\)` and `\[…\]` formula delimiters that the prompt asks for are read as Markdown escapes, so `\( E = mc^2 \)` becomes `( E = mc^2 )`.

The converted prediction has **no positions** and no physical lines: line breaks inside a paragraph are kept as the model wrote them, and the regions are the Markdown blocks in the model's order. The model gives no geometry, so none is invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make fireredocr-model and run with --cache`);
- the model files do not match `models.sha256`.

A page for which the model immediately generates an end token is a **genuine empty result**: `success` with empty text.

## Tests

`tests/test_fireredocr.py` runs offline in `make test`. It checks parameter validation, the prompt against the upstream SHA-256, the model checksums, the download script (without the network) and the engine protocol through the runner.

- The engine runs with fake `torch` and `transformers` modules (`tests/fixtures/fireredocr/fake`). The fake processor resizes like the real one.
- The fake model returns a hand-written Markdown page with an HTML table (`tests/fixtures/fireredocr/page.md`, not real model output), an empty page, an out-of-memory error and a newspaper page that never ends.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

`make fireredocr-smoke` (after `make fireredocr-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page. It also runs without `--cache`, which must fail explicitly.
