# GLM-OCR engine (compact page OCR, GPU)

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) ([#50](https://github.com/michal-hradis/ocr_benchmarking/issues/50)) is a 0.9B document OCR model from Z.ai: a CogViT vision encoder and a GLM-0.5B language decoder. This adapter gives it a whole page image with the model card's **text recognition prompt** and keeps the generated Markdown as the native output. There is no separate detection or line recognition. It runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-glm-ocr:0.9B-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make glm-ocr-model`) |
| `glm_ocr_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions and license

- **Model:** [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) at commit `2e85a62840ccac27daa451df36c736c4636b8628` (2026-09-11), **MIT** ([technical report](https://arxiv.org/abs/2603.10910)).
  - `GlmOcrForConditionalGeneration`: a 24-layer CogViT vision encoder (14-pixel patches, merged 2 × 2) and a 16-layer GLM language decoder with 8 key/value heads.
  - The weights are 2.65 GB of bfloat16 (`model.safetensors`); its SHA-256 equals Hugging Face's LFS hash.
  - The model card lists Chinese, English, French, Spanish, Russian, German, Japanese and Korean. Its benchmarks (OmniDocBench and others) are modern documents.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **5.19.0** (`GlmOcrForConditionalGeneration`, the `Glm46V` processor), with torchvision 0.29.1 for the image processor. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580).
  - The requirements are OvisOCR2's; the lock differs only in `huggingface-hub` 2.3.0 (released since). The image is separate, so each engine can change versions on its own.
- **Weights are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)), downloaded by an explicit step:

  ```sh
  make glm-ocr-model                             # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes the 8 files of `models.sha256` to `models/glm-ocr/GLM-OCR/<revision>/` under the cache directory and checks every file. The model card's `README.md`, `.gitattributes` and `.eval_results/` are not downloaded. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again and records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Inference path

The model card recommends the official [GLM-OCR SDK](https://github.com/zai-org/GLM-OCR) for document parsing. The SDK runs PP-DocLayoutV3 layout detection and then reads every block with the model, served by vLLM, SGLang or Ollama. This adapter uses **one different, supported path: the transformers model on the whole page**, as in the model card's transformers example:

- **Whole page, no layout model.** The issue asks for compact page OCR. A layout stage would make this a second [PaddleOCR-VL](../paddleocr-vl/README.md)-style pipeline (same PP-DocLayoutV3), and its block boxes would be the only geometry. The SDK's layout-plus-blocks pipeline is a possible follow-up preset.
- **No server.** vLLM/SGLang would need a server inside the container and their own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)).
- The decoding settings are the SDK's (below), so the model behaves as in its own pipeline except for the input: a full page instead of a layout block.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-glm-ocr                 # this image (no weights)
make glm-ocr-model                  # the pinned weights, 2.7 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/glm-ocr/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/glm-ocr --inspect
make glm-ocr-smoke                  # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` set explicitly, the model runs in float32 on the CPU; that is slow and not the configuration recorded here.

## Prompt and parameters

The prompt is fixed: **`Text Recognition:`**, the model card's document parsing prompt for text. The model card says the model supports only its fixed task prompts (`Text Recognition:`, `Formula Recognition:`, `Table Recognition:` and JSON information extraction). A custom "literal transcription" instruction, for example one asking to keep historic spelling, would therefore be outside what the model was trained for and is not used. The prompt does not ask the model to normalize or translate; what it does with historic spelling is measured, not instructed (below).

- The chat template puts the image and then the prompt into one user turn, with no system message. It is used with its defaults, as in the transformers example and the SDK. No `enable_thinking` argument is passed, so the template adds no `/nothink` or empty `<think>` block.
- The same prompt is used for full pages and for paragraph crops. There is no prompt parameter, and the sample `language` and other metadata are not used.
- The conversation is recorded with every sample (`extra.models.prompt`).

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `model_dir` | `models/glm-ocr/GLM-OCR/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `max_new_tokens` | 8192 | Generated tokens before the output counts as truncated: the SDK's `max_tokens` |
| `max_pixels` | 9,633,792 | The processor's `longest_edge` from the checkpoint's `preprocessor_config.json`; at least 12,544. The image gets at most half of it (below) |
| `repetition_penalty` | 1.1 | The SDK's repetition penalty, from 1.0 (none) to 2.0 |

**Decoding.** Decoding is greedy (the SDK's temperature 0 and top-k 1, and `do_sample: false` in the checkpoint's `generation_config.json`), with the SDK's **repetition penalty of 1.1**. The penalty is applied by transformers to every token already in the sequence, prompt included, as vLLM does. The transformers example in the model card uses no penalty; the SDK's value was chosen because the SDK is the model card's recommended way to run the model. Generation stops at either end token of `generation_config.json`: `<|endoftext|>` or `<|user|>` (the start of a next turn).

**Custom checkpoints.** Any directory under the cache that transformers can load as a GLM-OCR model (for example a fine-tuned one) can be given as `model_dir`. The directory must hold exactly the files of `models.sha256` (subdirectories included), because transformers may load other files. The default directory must match, with no other files, or every sample fails. For another directory, the result is recorded:

- If it holds exactly the files in `models.sha256`, all matching, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded.

Either way, `model_dir` and the SHA-256 of every file in it are recorded. For a custom checkpoint, also set `model` in a copy of `engine.toml` to name what is used. That name is what reports show as the model.

**Image size.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, modes other than RGB converted to RGB) and passes it to the processor unchanged. The processor works in units of 28 × 28 pixels, one image token each (14-pixel patches, merged 2 × 2):

- It keeps the aspect ratio and rounds both sides to whole units.
- A single image counts as two temporal frames against the pixel range. An image is therefore scaled down if it has more than `max_pixels` / 2 pixels (4,816,896, about 2195 × 2195) and up if it has fewer than 6,272.
- The default limit is **6,144 image tokens**, the SDK's `image_expect_length`. An A4 page scanned at 300 dpi (2480 × 3508) is scaled to 1820 × 2604, about 6,000 image tokens. Small print on a large newspaper page is therefore read at a much lower resolution than the scan.
- The size the model saw is recorded as `model_image`.

## Context, truncation and memory

The model's context is 131,072 tokens for the prompt (mostly image tokens) and the generated text together, far more than a page needs. In practice `max_new_tokens`, memory and time are the limits. Generation stops at the first end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- **Truncation.** Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the limit and `extra.stop_reason` set to `max_new_tokens` or `context` (`end-token` otherwise). The Markdown produced so far is kept and scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)).
- **Repetition.** The repetition penalty makes loops less likely, but does not prevent them. A loop that runs to `max_new_tokens` is reported as `truncated` in the same way. The adapter does not cut repeated text: it stays in `page.md` and is scored, so runaway generation remains visible. The SDK's own clean-up of repeated patterns (`find_consecutive_repeat`) is not applied; handling across engines is open in [#33](https://github.com/michal-hradis/ocr_benchmarking/issues/33).
- **Dense and multi-column pages.** A newspaper page is read in one pass in the model's own reading order, at no more than 6,144 image tokens. `max_new_tokens` can be raised, together with `timeouts.sample`; so can `max_pixels`, beyond what the checkpoint sets. The flexible metrics score the text independently of the column order.
- **Memory.** The bfloat16 weights take about 2.2 GB of GPU memory, the peak with a small image. The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`. On the RTX 4090 it was 2.9–3.2 GB for dev-01 pages at 3,200–6,100 image tokens. A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues. Host memory is limited to 16 GB by `[resources]`.
- **Time.** Generation is token by token with batch size 1. `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample. A truncated sample generates all `max_new_tokens`; `timeouts.sample = 900` seconds allows for that. On the RTX 4090, generation runs at about 51–59 tokens/s, so 8,192 tokens take about 145 s; book pages took 6–17 s, and loading the model takes about 12 s.

**Observed on dev-01** (seven non-representative pages, RTX 4090):

- **Repetition.** Three of seven outputs ran into a loop until `max_new_tokens`, despite the repetition penalty. Both Czech newspapers were read for several thousand tokens as long paragraphs, then repeated one phrase to the end. The Glagolitic page was read as unrelated Georgian-like text from the start. All three are reported as `truncated` and scored as produced.
- **Historic spelling.** The model never wrote long s (`ſ`), of which the references have 27–40 per book page: it wrote `s` (`sey` for `ſey`), and sometimes `f`. It wrote the 1692 page's superscript-e umlaut (`uͤ`, 7 in the reference) as `ü` (`Mücke` for `Muͤcke`). The default evaluation counts each as an error. Historic word forms and punctuation (`sey`, `formiret`, `/` as a comma) were kept as printed. On the 1692 page the printed lines, with their end-of-line hyphens, were kept; the other book pages were partly or wholly joined into paragraphs.
- **Lines.** Long paragraphs give few hypothesis units (5–29 per book page, 10–12 per newspaper), so the flexible alignment has little to reorder, and fCER equals CER on every page.

## Outputs

| File | Content |
| --- | --- |
| `page.md` | The generated text, decoded without special tokens and otherwise unchanged: the native output and the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `model_image`, the timings, `gpu_peak_memory_mb`, the transformers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (`verified_official_checkpoint`; `repository` and `revision` only when verified; `model_dir`, file checksums, the configured `model`, the prompt) and the parameters |

**Plain text and the prediction.** `page.md` is converted to the standardized prediction on the host by the shared Markdown converter ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), not in the container. Markdown and HTML markup is therefore never part of the recognized text:

- Headings, paragraphs and list items become typed regions with their text only.
- Tables (Markdown or HTML) become `table` regions with one row per line and cells separated by tabs.
- `$$` formulas keep their TeX source. Emphasis and links keep only their text.
- A fenced code block keeps its content verbatim as an `other` region. On a blank page the model wrote an empty ```` ```markdown ```` fence, which gives an empty prediction.

The converted prediction has **no positions** and no physical lines: line breaks inside a paragraph are kept as the model wrote them, and the regions are the Markdown blocks in the model's order. The model generates no coordinates with this prompt, and none are invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make glm-ocr-model and run with --cache`);
- the model files do not match `models.sha256`, or the default model directory holds other files.

A page for which the model immediately generates an end token, or only an empty fence, is a **genuine empty result**: `success` with empty text.

## Tests

`tests/test_glm_ocr.py` runs offline in `make test`. It checks parameter validation, the prompt, the model checksums, the download script (without the network) and the engine protocol through the runner.

- The engine runs with fake `torch` and `transformers` modules (`tests/fixtures/glm-ocr/fake`). The fake processor resizes like the real one (28-pixel units, two frames) and checks that the chat template gets no extra arguments. The fake model checks greedy decoding with a repetition penalty and both end tokens.
- The fake model returns a hand-written Markdown page with historic spelling, a formula and a table (`tests/fixtures/glm-ocr/page.md`, not real model output), a page ended by `<|user|>`, the empty fence that the real model wrote for a blank page, a tiny image, an out-of-memory error and a newspaper page that never ends.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

`make glm-ocr-smoke` (after `make glm-ocr-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page. A second run with `max_new_tokens = 16` must mark the page `truncated` and keep and score its partial Markdown. It also runs without `--cache`, which must fail explicitly.
