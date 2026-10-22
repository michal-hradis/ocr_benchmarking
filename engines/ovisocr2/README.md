# OvisOCR2 engine (end-to-end image-to-Markdown, GPU)

Third direct page-image model ([#16](https://github.com/michal-hradis/ocr_benchmarking/issues/16)), after [LightOnOCR-2](../lightonocr/README.md) and [FireRed-OCR](../fireredocr/README.md): [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) is a compact 0.8B model that reads a whole page image and generates its content as one Markdown document in reading order, with tables as HTML, formulas as LaTeX and figures as `<img>` tags with a bounding box. There is no separate detection or line recognition. It runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-ovisocr2:0.8B-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `prompt.txt` | The official instruction, byte for byte |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make ovisocr2-model`) |
| `ovisocr2_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions

- **Model:** [`ATH-MaaS/OvisOCR2`](https://huggingface.co/ATH-MaaS/OvisOCR2) at commit `1fc9221b7823a371d6e97f92d527cc847e24e107`, Apache-2.0 ([technical report](https://arxiv.org/abs/2607.13639)).
  - It is [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) post-trained for page-level document parsing (`Qwen3_5ForConditionalGeneration`). Its language model mixes 18 linear-attention (gated delta net) layers with 6 full-attention layers.
  - The weights are 1.7 GB of bfloat16 (`model.safetensors`).
  - The model card lists no languages. Its benchmarks (OmniDocBench, PureDocBench) are mostly English and Chinese documents.
  - A successor, OvisOCR3, appeared on Hugging Face on 2026-10-21. It is not part of this adapter.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **5.19.0** (`Qwen3_5ForConditionalGeneration`), with torchvision 0.29.1 for the image processor. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580).
  - The requirements are the same as FireRed-OCR's and LightOnOCR's. The image is separate, so each engine can change versions on its own.
  - The model card's only inference example uses vLLM 0.22.1. vLLM is not used here, because it would need a server inside the container and its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)). The prompt, image sizes and decoding below are the example's.
  - The optional `flash-linear-attention` and `causal-conv1d` kernels are not installed. transformers then runs the linear-attention layers with its PyTorch implementation, which gives the same results up to floating-point differences.
- **Weights are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)), downloaded by an explicit step:

  ```sh
  make ovisocr2-model                            # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/ovisocr2/OvisOCR2/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again and records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-ovisocr2                # this image (no weights)
make ovisocr2-model                 # the pinned weights, 1.7 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/ovisocr2/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/ovisocr2 --inspect
make ovisocr2-smoke                 # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` set explicitly, the model runs in float32 on the CPU; that is very slow and not the configuration recorded here.

## Prompt and parameters

The prompt is the **official instruction** from the vLLM example in the model card at the pinned revision. `prompt.txt` holds it byte for byte, including its leading newline, and a test checks its SHA-256.

- The instruction asks for one Markdown document in natural reading order, LaTeX formulas, HTML tables, and charts and images as `<img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />` with coordinates scaled to [0, 1000). It asks for the original text, without translation or paraphrasing.
- The chat template puts the image and then the instruction into one user turn, with no system message. Thinking is turned off (`enable_thinking=False`, as in the example), so the template ends the prompt with an empty `<think>` block and the model answers directly.
- The same prompt is used for full pages and for paragraph crops. There is no prompt parameter, and the sample `language` and other metadata are not used.
- The conversation and the SHA-256 of the instruction are recorded with every sample (`extra.models.prompt`, `extra.models.prompt_sha256`).

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `model_dir` | `models/ovisocr2/OvisOCR2/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `max_new_tokens` | 16384 | Generated tokens before the output counts as truncated, as in the model card |
| `max_pixels` | 8,294,400 | The processor scales larger images down to at most this many pixels (2880 × 2880, as in the model card); at least 200,704 (448 × 448) |

**Decoding.** Decoding is greedy, as in the model card (temperature 0); there are no sampling parameters. The checkpoint has no `generation_config.json`, and its `config.json` names `<|endoftext|>` as the end token, while the chat turn ends with the tokenizer's end token `<|im_end|>`. Generation stops at either of them, as it does in vLLM. No repetition penalty is applied.

**Post-processing in the model card.** The example also removes the `<img>` figure tags and, for outputs of at least 8,000 characters, cuts a repeated tail down to one copy (`_clean_truncated_repeats`). The adapter does neither: `page.md` is the model's output unchanged. Figure tags carry no text and are dropped by the host conversion anyway (below). A repeated tail stays in the output and is scored, so runaway generation remains visible; how to handle it across engines is open in [#33](https://github.com/michal-hradis/ocr_benchmarking/issues/33).

**Custom checkpoints.** Any directory under the cache that transformers can load as a Qwen3.5 model (for example a fine-tuned OvisOCR2) can be given as `model_dir`. Only the files are checked against `models.sha256`, never the directory name. The default directory must match, or every sample fails. For another directory, the result is recorded:

- If every file in `models.sha256` is present and matches, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded.

Either way, `model_dir` and the SHA-256 of every file in it are recorded. For a custom checkpoint, also set `model` in a copy of `engine.toml` to name what is used, for example `"my-org/OvisOCR2-finetuned@<revision>"`. That name is what reports show as the model.

**Image size.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, modes other than RGB converted to RGB) and passes it to the processor unchanged. The processor works in units of 32 × 32 pixels, one image token each (16-pixel patches, merged 2 × 2):

- It keeps the aspect ratio and rounds both sides to whole units.
- It scales an image down if it has more than `max_pixels` pixels, and up if it has fewer than 200,704 (448 × 448), as the model card sets. A paragraph crop is therefore enlarged.
- The default limit is 8,100 image tokens. An A4 page scanned at 300 dpi (2480 × 3508) is scaled to 2400 × 3424, about 8,000 image tokens.
- The size the model saw is recorded as `model_image`.

## Context, truncation and memory

The model's context is 262,144 tokens for the prompt (mostly image tokens) and the generated text together, far more than a page needs. In practice `max_new_tokens`, memory and time are the limits. Generation stops at the first end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- **Truncation.** Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the limit and `extra.stop_reason` set to `max_new_tokens` or `context` (`end-token` otherwise). The Markdown produced so far is kept and scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). A model that repeats a line until the limit is also reported this way.
- **Dense and multi-column pages.** A newspaper page is read in one pass in the model's own reading order, scaled down to at most `max_pixels`.
  - A page with 2,000+ lines can need more than 16,384 tokens. It is then truncated and scored against every reference line, so the missing text counts as deletions in the strict and flexible metrics.
  - `max_new_tokens` can be raised, together with `timeouts.sample`; `max_pixels` too, beyond what the model card uses.
  - The column order is the model's. The flexible metrics score the text independently of that order.
  - Tiling the page inside the engine is a possible follow-up, not part of this adapter.
- **Memory.** The bfloat16 weights take 1.7 GB of GPU memory.
  - Only the 6 full-attention layers keep a key/value cache, about 12 KB per token (2 key/value heads of 256 dimensions). That is about 0.3 GB for 8,100 image tokens and 16,384 generated ones. The 18 linear-attention layers keep a state of fixed size.
  - The vision encoder (12 layers) runs on all patches of the image at once, up to 32,400 at the default `max_pixels`.
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
- `$$` formulas keep their TeX source. Emphasis and links keep only their text.
- `<img>` figure tags have no text and are dropped, with their boxes.
- The converter's limitations apply. For example, `\(…\)` and `\[…\]` formula delimiters are read as Markdown escapes, so `\( E = mc^2 \)` becomes `( E = mc^2 )`.

The converted prediction has **no positions** and no physical lines: line breaks inside a paragraph are kept as the model wrote them, and the regions are the Markdown blocks in the model's order. The figure boxes are real model geometry, but only for figures, which have no text to score. They are kept in `page.md` and not converted into prediction regions; that is a possible follow-up. No text geometry is invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make ovisocr2-model and run with --cache`);
- the model files do not match `models.sha256`.

A page for which the model immediately generates an end token is a **genuine empty result**: `success` with empty text.

## Tests

`tests/test_ovisocr2.py` runs offline in `make test`. It checks parameter validation, the prompt against the model card's SHA-256, the model checksums, the download script (without the network) and the engine protocol through the runner.

- The engine runs with fake `torch` and `transformers` modules (`tests/fixtures/ovisocr2/fake`). The fake processor resizes like the real one and checks that thinking is off. The fake model checks greedy decoding with both end tokens.
- The fake model returns a hand-written Markdown page with a figure tag and an HTML table (`tests/fixtures/ovisocr2/page.md`, not real model output), a page ended by `<|endoftext|>`, an empty page, an out-of-memory error and a newspaper page that never ends.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

`make ovisocr2-smoke` (after `make ovisocr2-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page. A second run with `max_new_tokens = 16` must mark the page `truncated` and keep and score its partial Markdown. It also runs without `--cache`, which must fail explicitly.
