# Nanonets-OCR2-3B engine (page image to Markdown, vLLM, GPU)

[Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) by Nanonets ([#99](https://github.com/michal-hradis/ocr_benchmarking/issues/99)) is an open-weight image-to-Markdown model fine-tuned from Qwen2.5-VL-3B-Instruct. It reads a whole page image and writes its text in reading order as Markdown, with tables as HTML or Markdown and equations as LaTeX. It also adds semantic tags: `<page_number>…</page_number>`, `<watermark>…</watermark>`, `<signature>…</signature>`, `<img>…</img>` for a picture's description or caption, and ☐/☑ for check boxes.

The engine has one preset, `nanonets-ocr2-3b` (`engine.toml`). It sends the model card's vLLM request: the page image as stored, the official prompt, greedy decoding and up to 15,000 generated tokens. The output is Markdown without positions. The model gives no boxes, so none are invented.

The engine runs **vLLM inside its container** as a child process of the adapter, on the loopback interface only ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)), with vLLM's built-in Qwen2.5-VL model. The image uses the vLLM 0.30 base unchanged ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-nanonets-ocr2:3B-v1` from `ocrbench/base-cu130-vllm030:vllm0.30.0-py3.12-v1`, without weights and without extra packages |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `prompt.txt` | The official OCR prompt, byte for byte |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make nanonets-ocr2-model`) |
| `nanonets_ocr2_engine.py` | The adapter |
| `engine.toml` | The preset for `ocrbench run start` |

## Versions and license

- **Model:** [`nanonets/Nanonets-OCR2-3B`](https://huggingface.co/nanonets/Nanonets-OCR2-3B) at revision `c3886ff00bb037ce7da24988c9eafaf1fe2bed72` (2025-10-16). Its architecture is `Qwen2_5_VLForConditionalGeneration`: the Qwen2.5-VL vision encoder (14-pixel patches merged 2 × 2) and a 36-layer, 3B language model with a 128,000-token context. The weights are 7.5 GB in bfloat16, in two files: `model-00001-of-00002.safetensors` (`84c3306e…`) and `model-00002-of-00002.safetensors` (`8002ad9c…`). The weights' SHA-256 is the Hugging Face LFS object ID. `models.sha256` pins the files vLLM loads: the configuration, generation configuration, weights and their index, chat template, processor configurations and tokenizer files. The model card, the `.gitattributes` file and the Ollama `Modelfile` are not pinned. There is no model code: vLLM implements the architecture, so the server runs without `--trust-remote-code`.
- **License: not declared.** The model card has no `license` field, and the [cookbook repository](https://github.com/NanoNets/Nanonets-OCR2) has no license file (checked at commit `9d4a86c85fbfbdee53a7e32bc3041e73ed80fc07`). The base model, [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), is under the Qwen Research License, which allows non-commercial use. Read the terms before using this model for anything other than research.
- **Languages.** The model card lists English, Chinese, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Arabic "and many more", plus handwritten documents. Czech, Fraktur and Glagolitic are not named.
- **Serving path:** the model card's "Using vLLM" example: `vllm serve nanonets/Nanonets-OCR2-3B` and an OpenAI chat request with the image and the prompt, `temperature=0.0` and `max_tokens=15000`. The model card's transformers example (greedy `generate`, also with `max_new_tokens=15000`) is not used.
- **Inference:** vLLM **0.30.0** with torch **2.13.0** (CUDA 13.0), transformers **5.16.1** and huggingface-hub **1.33.0**, from the vLLM 0.30 base, which registers Qwen2.5-VL. NVIDIA driver ≥ 580.
- **Weights are not in the image.** They go to the `/cache` volume in an explicit download step:

  ```sh
  make nanonets-ocr2-model            # 7.5 GB into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/nanonets-ocr2/Nanonets-OCR2-3B/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks that the directory holds exactly these files, all matching, and records the SHA-256 of every file with each sample. vLLM reads the model from that directory, offline (`HF_HUB_OFFLINE=1`).
- **Compiled kernels.** At start-up vLLM compiles the model with torch.compile, captures CUDA graphs and compiles Triton kernels, which it loads as shared objects. The runner's `/tmp` is mounted `noexec`, so these caches are written to `runtime/nanonets-ocr2/` in the cache directory. Later runs reuse them. They are not model files and are not checked; delete the directory to start clean.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-vllm030           # shared vLLM 0.30 base, once
make docker-nanonets-ocr2           # this image (no weights)
make nanonets-ocr2-model            # the pinned weights, 7.5 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/nanonets-ocr2/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/nanonets-ocr2 --inspect
make nanonets-ocr2-smoke            # opt-in: build the image, run the preset on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`), and the server sees only the GPU of `device` (`CUDA_VISIBLE_DEVICES`). If the requested GPU is not visible, every sample fails with `engine-error`; there is no CPU fallback.

## Pipeline and parameters

1. **Server.** The adapter checks the model files and starts `vllm serve` on the model directory, on `127.0.0.1` and a free port, with served model name `nanonets-ocr2`, bfloat16 (the checkpoint's dtype), one image per prompt (`--limit-mm-per-prompt '{"image": 1}'`), a context (`--max-model-len`) of **32,768** tokens and at most 8 concurrent sequences, as pages are sent one at a time. The context holds the largest image the processor makes (16,384 image tokens), the prompt and the 15,000 generated tokens. The model card runs the server with its defaults, that is with the full 128,000-token context; a shorter context reserves less key/value cache and changes no output that fits in it.

   - `VLLM_USE_FLASHINFER_SAMPLER=0`: FlashInfer's top-k/top-p sampling kernels would be compiled at start-up with `nvcc`, which the image (like the base) does not contain. vLLM's PyTorch sampler is used instead.

   The adapter waits until `/health` answers (at most `startup_seconds`). The full command, the environment it sets, the GPU and its compute capability are recorded per sample (`extra.environment`).
2. **Image.** The image is decoded as stored (EXIF orientation is not applied; other modes are converted to RGB) and sent at its stored size as a lossless PNG data URL. The adapter does not resize it, as in the model card. The checkpoint's Qwen2.5-VL processor (`preprocessor_config.json`) resizes it in the server: it rounds each side to a multiple of 28 pixels and, if the image has more than `max_pixels` = 12,845,056 pixels, scales it down to at most that many, keeping the aspect ratio (bicubic). Each 28 × 28 block is one image token, so an image has at most 16,384 image tokens. A 1600 × 2325 book page keeps its size (about 4,700 image tokens). A 4194 × 6192 newspaper page (26 megapixels) is scaled down to 2940 × 4340 pixels (16,275 image tokens). The image tokens are part of `extra.prompt_tokens`. Pages are not tiled.
   - The cookbook (`image2md.ipynb`) also tries scaling the short side to 1024 or 2048 pixels, depending on the document type. That choice is tuning per document, so it is not used here ([#99](https://github.com/michal-hradis/ocr_benchmarking/issues/99)).
3. **Request.** One chat request per page, as in the model card's vLLM example: one user message with the image first and then the official prompt (`prompt.txt`, the same in the model card's transformers example and the cookbook). The chat template adds the default system message "You are a helpful assistant.", as in the model card's transformers example. The sample `language` is not used. The prompt asks for HTML tables, LaTeX equations, `<img>` descriptions or captions, `<watermark>` and `<page_number>` tags and ☐/☑ check boxes.
   - **Decoding:** greedy (temperature **0**), up to **15,000** generated tokens, as in the model card. The checkpoint's `generation_config.json` sets a repetition penalty of **1.05**, which vLLM and transformers apply by default. The adapter sends it explicitly, with seed 0, so that the request does not depend on the server's handling of generation defaults. The temperature of `generation_config.json` (sampling at 10⁻⁶) is overridden by the request.
   - The model card also suggests `repetition_penalty=1` and a variant of the prompt for financial tables. That is document-specific advice, so it is not used.

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample. A configured `model` must be `nanonets/Nanonets-OCR2-3B@c3886ff00bb037ce7da24988c9eafaf1fe2bed72`; otherwise every sample fails.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>` |
| `max_tokens` | 15000 | Generated tokens per page before the output counts as truncated; below the 32,768-token context. The context also holds the image tokens, so vLLM caps a larger value at the tokens the image leaves, and the sample is reported as stopped at the end of the context |
| `temperature` | 0 | Greedy decoding |
| `repetition_penalty` | 1.05 | The checkpoint's generation configuration |
| `seed` | 0 | vLLM's per-request seed |
| `gpu_memory_utilization` | **0.6**, also the maximum ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)) | Share of the GPU's memory vLLM reserves for the weights, CUDA graphs and key/value cache. vLLM refuses to start when less is free |
| `max_num_seqs` | 8 | vLLM's limit of concurrent sequences; requests are sent one at a time |
| `startup_seconds` | 900 | Time allowed for the server to become ready |

## Conversion to the prediction

`page.md` is the model output, unchanged: the native output and the primary output. The shared Markdown converter turns it into the standardized prediction on the host ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)). Headings, paragraphs, list items, tables (one row per line, cells separated by tabs), `$$` formulas (TeX source) and HTML embedded in Markdown become typed regions with their text only, without positions and without physical lines. The prediction records `conversion` `{"from": "markdown", "version": 2}`.

Converter version 2 was added for this model ([#99](https://github.com/michal-hradis/ocr_benchmarking/issues/99)):

- `<page_number>14</page_number>`: tags with an underscore in their name are inline markup, so the page number `14` is kept as text. Before version 2, Markdown kept the tags as text, because CommonMark does not recognize such names. `<watermark>` and `<signature>` already kept only their text.
- `<img>A woodcut of a fox.</img>`: the text between `<img>` and an explicit `</img>` adds nothing, like an image's alt text. The model writes a description of the picture there, or the picture's caption if the page has one, so a real caption inside `<img>` is lost. Following the maintainer's decision for picture descriptions ([#53](https://github.com/michal-hradis/ocr_benchmarking/issues/53)), it is not scored.

The check box symbols ☐ and ☑ are recognized text, not markup, and are kept. No conversion happens in the container ([ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)).

## Truncation, repetition, failures, memory and time

- A request stops at the end token, at `max_tokens` or at the end of the context. Without an end token (stop reason `length`) the sample is `truncated`, and the output is scored as produced ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). The message says which limit was reached: the end of the context (prompt plus generated tokens fill it) or `max_tokens`.
- **Repetition.** The repetition penalty of 1.05 is the only repetition guard; there is no n-gram processor. The adapter checks whether the output ends in a loop (`extra.tail_repetition`): a unit of fewer than 64 characters repeated 8 times in a row, or a longer one of up to 4,096 characters (a sentence or paragraph) repeated 3 times. It adds "the output ends in a repetition loop" to the message of a truncated sample. The output is not trimmed.
- An empty output with an end token is an empty page (`success`, with the message `the model returned an empty page`). Any other stop reason fails the sample (`engine-error`). The native output and the request diagnostics in `extra` are kept.
- A failed request (for example an out-of-memory error in the server) fails only that sample. If the server exits, that sample and every later one fail with its log tail.
- Every sample fails when the GPU is unusable, the parameters are invalid, the configured model is not the pinned checkpoint, the model directory is missing or does not match its checksums, or the server does not start (the message has the end of its log).

**Memory.** vLLM reserves `gpu_memory_utilization` of the GPU's memory up front: 0.6 × 24 GB = 14.1 GiB on an RTX 4090. The bfloat16 weights take about 7 GiB of it. The language model keeps about 36 KiB of key/value cache per token (36 layers, 2 key/value heads of 128 dimensions), so a full 32,768-token context needs about 1.2 GiB. It fits an RTX 4090 at the benchmark's share: all seven dev-01 pages, including the 26-megapixel newspaper page at 16,433 prompt tokens, ran without an out-of-memory error. The peak memory per page is not measured.

**Time.** `extra.request_seconds` is the time of the request (inference), `extra.tokens_per_second` the generated tokens per second of it, and the sample's `seconds` the whole page including image encoding. Server start-up is not part of the per-sample engine time.

## Outputs

| File | Content |
| --- | --- |
| `page.md` | The model output, unchanged: the native output and the primary output, converted on the host |
| `result.json` | Status, processing time and, in `extra`: `finish_reason`, `tail_repetition`, `prompt_tokens`, `generated_tokens`, `max_tokens`, `request_seconds`, `tokens_per_second`, `source_image`; the vLLM, transformers, huggingface-hub, torch and CUDA versions, the GPU and its compute capability, the server command and environment, the start-up time, the model (repository, revision, file checksums, the configured `model`, the prompt and its SHA-256, the context) and the parameters |

## Results on dev-01

Measured in run `dev01-nanonets-ocr2-3b-v1` (image `sha256:5c47aed3…`) on an RTX 4090; the scores are in the pull request for [#99](https://github.com/michal-hradis/ocr_benchmarking/issues/99). The development set is seven pages and not representative.

- **Book pages.** The four 1600-pixel German and Latin prints ended normally, at 0.8–10.7 % CER. No long s (ſ) was written in Fraktur, and Fraktur letters are misread (`Dertter`, `daffelbe`). Page numbers came as `<page_number>` tags and are kept as text by converter version 2.
- **Newspapers: repetition loops.** Both Czech newspaper pages (4194 × 6192 and 3509 × 5239, about 16,400 image tokens each) start with correct Czech text with diacritics. They then fall into a loop that repeats one phrase until `max_tokens` (`truncated`, "the output ends in a repetition loop"). On `3e6309d0` the loop starts after about 10,400 of 31,900 characters, so the output is twice as long as the reference and the CER is above 100 %. On `bd497a62` it starts after about 4,400 characters. The repetition penalty of 1.05 does not prevent it.
- **Glagolitic page.** The model does not read the script. It returned `<img>original_image.jpg</img>` (dropped by the converter) and `<signature>匿名</signature>` (Japanese for "anonymous", kept as text): CER 100 %.
- **Time.** About 95–112 generated tokens/s. A book page took 4–9 s and a newspaper page that runs to `max_tokens` about 142 s. Server start-up (load, compilation and CUDA graph capture) took 74 s and is not part of the per-sample time.

## Tests

`tests/test_nanonets_ocr2.py` runs offline in `make test`:

- the preset (the pinned model and image, the default parameters), the prompt (byte for byte the official one), parameter validation, the server command, the request (one user message with the image and then the prompt, greedy decoding, `max_tokens`, the repetition penalty and the seed, nothing else) and the pinned model files;
- the repetition check, the checksums and the download script;
- the engine protocol through the runner, with a fake `vllm` executable that serves fixed outputs over the loopback interface and checks the command line and every request, and a fake `torch` module (`tests/fixtures/nanonets-ocr2/fake`). The outputs are a hand-written page in the model's output style (`tests/fixtures/nanonets-ocr2/page.md`, not real model output) with a page number tag, a picture description, an HTML table, a check box, a watermark and a formula, converted on the host against the expected plain text; an empty output (success); a large page sent at its stored size; a newspaper page cut off at `max_tokens` (with and without a repetition loop) and at the end of the context; an aborted request with its diagnostics; a failed request; and a server that exits. The run is evaluated and reported, including the truncated page against a 2,000-line reference;
- start-up failures: invalid parameters, another model, no GPU, missing, changed or extra model files and a server that exits during start-up.

`make nanonets-ocr2-smoke` (after `make nanonets-ocr2-model`) builds the image and runs the preset with the real model on the GPU through `ocrbench run start`, `evaluate` and `report` on a synthetic full page, a paragraph crop and a blank page.
