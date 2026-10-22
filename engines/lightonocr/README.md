# LightOnOCR-2 engine (end-to-end image-to-text, GPU)

Compact end-to-end vision-language baseline ([#14](https://github.com/michal-hradis/ocr_benchmarking/issues/14)): [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) reads a whole page or paragraph image and generates its transcription as Markdown, with no separate detection or line recognition. It runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-lightonocr:2-1B-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make lightonocr-model`) |
| `lightonocr_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions

- **Model:** [`lightonai/LightOnOCR-2-1B`](https://huggingface.co/lightonai/LightOnOCR-2-1B) at commit `be7c41313c30e940fe5571de59e5fabc779e00c8`, Apache-2.0. This is LightOn's recommended OCR variant ([paper](https://arxiv.org/abs/2601.14251)): a Pixtral vision encoder and a Qwen3 decoder with about 1B parameters in total, stored as 2.0 GB of bfloat16 weights (`model.safetensors`). The model card lists English, French, German, Spanish, Italian, Dutch, Portuguese, Swedish, Danish, Chinese and Japanese; Czech is not among them. The `-bbox` variants, which also output boxes for embedded images, are not supported: their box syntax would end up in the page text.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **5.19.0** (`LightOnOcrForConditionalGeneration`, available since transformers 5.0), with torchvision 0.29.1 for the image processor, on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580). The adapter has been run with it on an RTX 4090. Loading prints two transformers warnings, both harmless here: the checkpoint's `config.json` says `model_type: mistral3` (all 532 tensors load into `LightOnOcrForConditionalGeneration`), and the tokenizer's pre-tokenization regex is flagged as the faulty Mistral one. That regex only affects encoding text, and the prompt contains no text, only special and image tokens. The model card also documents vLLM serving, which is faster for large batches. It is not used here, because it would need a server inside the container and its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)).
- **Weights are not in the image.** At 2 GB they go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)), downloaded by an explicit step:

  ```sh
  make lightonocr-model                          # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/lightonocr/LightOnOCR-2-1B/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again and records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-lightonocr              # this image (no weights)
make lightonocr-model               # the pinned weights, 2 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/lightonocr/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/lightonocr --inspect
make lightonocr-smoke               # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` set explicitly, the model runs in float32 on the CPU; that is very slow and not the configuration recorded here.

## Prompt and parameters

The prompt is the model's chat template with the **image alone** and no text instruction, as in the model card: `<|im_start|>system<|im_end|>`, then a user turn containing the image tokens, then `<|im_start|>assistant`. The conversation is recorded with every sample (`extra.models.prompt`). There is no prompt parameter; the sample `language` and other metadata are not used.

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `model_dir` | `models/lightonocr/LightOnOCR-2-1B/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `max_new_tokens` | 8192 | Generated tokens before the output counts as truncated, `1 <= value < 16384` |
| `temperature`, `top_p` | 0.2, 0.9 | Sampling, as in the model's `generation_config.json` and the model card; `temperature = 0` is greedy decoding |
| `seed` | 0 | Set before every sample, so a sample's sampling does not depend on the samples before it |
| `longest_edge` | 1540 | The processor scales larger images down so that the longest side is at most this many pixels (the model's training resolution). It does not scale smaller images up |

**Custom checkpoints.** Any directory under the cache that transformers can load as LightOnOCR (for example a fine-tuned model) can be given as `model_dir`. Only the files are checked against `models.sha256`, never the directory name. The default directory must match, or every sample fails. For another directory, the result is recorded:

- If every file in `models.sha256` is present and matches, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded.

Either way, `model_dir` and the SHA-256 of every file in it are recorded. For a custom checkpoint, also set `model` in a copy of `engine.toml` to name what is used, for example `"my-org/LightOnOCR-2-1B-finetuned@<revision>"`. That name is what reports show as the model.

**Image size.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, modes other than RGB converted to RGB) and passes it to the processor unchanged. The processor keeps the aspect ratio, scales the image down to `longest_edge` if needed and rounds it to whole patches; one image token covers 28 × 28 pixels, so a 1540 × 1100 page is 55 × 40 = 2,200 image tokens. The size the model saw is recorded as `model_image`. A paragraph crop is not enlarged.

**Reproducibility.** Sampling with a fixed seed repeats on the same GPU, driver and software versions. Different GPUs or kernels can still change the generated text; use `temperature = 0` for greedy decoding, which is deterministic except for floating-point differences between GPUs.

## Context, truncation and memory

The model's context is **16,384 tokens** for the prompt (mostly image tokens) and the generated text together. Generation stops at the first end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- **Truncation.** Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the limit and `extra.stop_reason` set to `max_new_tokens` or `context` (`end-token` otherwise). The Markdown produced so far is kept and scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). A model that repeats a line until the limit is also reported this way.
- **Newspapers.** The whole page is one image of at most 1540 pixels, so the text on a large newspaper page becomes very small, and 2,000+ lines need far more than the available tokens. Expect truncated, low-recall results; the strict and flexible metrics still score them against every reference line. Tiling the page inside the engine is a possible follow-up, not part of this adapter.
- **Memory.** The bfloat16 weights take 2 GB of GPU memory, and the key/value cache about 112 KB per token (28 layers, 8 key/value heads of 128 dimensions), under 2 GB at the full context. The vision encoder runs on all patches of the image at once. A single RTX 4090 (24 GB) is the target. The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`. A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues. Host memory is limited to 16 GB by `[resources]`.
- **Time.** Generation is token by token with batch size 1. A truncated sample generates all `max_new_tokens` and is the slowest; `timeouts.sample = 600` seconds allows that. `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample.

## Outputs

| File | Content |
| --- | --- |
| `page.md` | The generated text, decoded without special tokens and otherwise unchanged: the native output and the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `model_image`, the timings, `gpu_peak_memory_mb`, the transformers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (`verified_official_checkpoint`; `repository` and `revision` only when verified; `model_dir`, file checksums, the configured `model` and the prompt) and the parameters |

**Plain text and the prediction.** `page.md` is converted to the standardized prediction on the host by the shared Markdown converter ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), not in the container. Markdown delimiters are therefore never part of the recognized text: headings, paragraphs, list items, tables (one row per line, cells separated by tabs), `$$` formulas (TeX source) and HTML embedded in Markdown become typed regions with their text only, and emphasis, links and images keep only their text. The converter's limitations apply, for example `\[…\]` and `\(…\)` math delimiters are read as Markdown escapes. The converted prediction has **no positions** and no physical lines: line breaks inside a paragraph are kept as the model wrote them, and the regions are the Markdown blocks in the model's order. The model gives no geometry, so none is invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. An unusable GPU, invalid parameters, a missing model directory (`... does not exist; download the weights with make lightonocr-model and run with --cache`) or model files that do not match `models.sha256` fail every sample. A page for which the model immediately generates an end token is a **genuine empty result**: `success` with empty text. The model does not reliably do that for a blank image: on the synthetic blank page it generates `$\mathcal{M}$` three times, which is kept and scored as a hallucination, like any other output.

## Tests

`tests/test_lightonocr.py` checks parameter validation, the model checksums, the download script (without the network) and the engine protocol through the runner with fake `torch` and `transformers` modules (`tests/fixtures/lightonocr/fake`), all offline in `make test`. The fake model returns a hand-written Markdown page (`tests/fixtures/lightonocr/page.md`, not real model output), an empty page, an out-of-memory error and a newspaper page that never ends. That run is then evaluated and reported, including the truncated page against a 2,000-line reference. `make lightonocr-smoke` (after `make lightonocr-model`) builds the image and runs the real model on the GPU through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page, plus a run without `--cache`, which must fail explicitly.
