# LightOnOCR-3 engine (page image to Markdown or layout blocks, vLLM, GPU)

[LightOnOCR-3](https://huggingface.co/blog/lightonai/lightonocr-3) by LightOn ([#92](https://github.com/michal-hradis/ocr_benchmarking/issues/92)) is a family of three end-to-end OCR models that read a whole page image and write its text as Markdown, with HTML tables and LaTeX math, in reading order. They have two modes, chosen by the prompt:

- **plain:** the image alone. The output is the page as Markdown, without positions, as LightOnOCR-2 ([engines/lightonocr](../lightonocr/README.md)) writes it.
- **grounding:** the image and the word `grounding`. The model writes a labeled box, normalized to 0–1000 of the image, before every layout block:

  ```text
  ![title](80,70,920,140) # Company report 2025
  ![text](80,175,920,225) Revenue rose 20%.
  ![image](80,285,460,545) A solar-powered factory.
  ```

  There are no text lines.

This engine is separate from the LightOnOCR-2 engine, which is unchanged. Each of the six presets names one checkpoint and one mode and is an independent benchmark result:

| Preset | File | Checkpoint | Mode |
| --- | --- | --- | --- |
| `lightonocr3-4b-grounding` | `engine.toml` | LightOnOCR-3-4B | grounding |
| `lightonocr3-4b-plain` | `presets/4b-plain.toml` | LightOnOCR-3-4B | plain |
| `lightonocr3-1b-grounding` | `presets/1b-grounding.toml` | LightOnOCR-3-1B | grounding |
| `lightonocr3-1b-plain` | `presets/1b-plain.toml` | LightOnOCR-3-1B | plain |
| `lightonocr3-0.8b-grounding` | `presets/0.8b-grounding.toml` | LightOnOCR-3-0.8B | grounding |
| `lightonocr3-0.8b-plain` | `presets/0.8b-plain.toml` | LightOnOCR-3-0.8B | plain |

The engine runs **vLLM inside its container** as a child process of the adapter, on the loopback interface only ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)), with vLLM's built-in models for the checkpoints. The image uses the vLLM 0.30 base unchanged ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-lightonocr3:3-v1` from `ocrbench/base-cu130-vllm030:vllm0.30.0-py3.12-v1`, without weights and without extra packages |
| `models-0.8B.sha256`, `models-1B.sha256`, `models-4B.sha256` | Checksums of each checkpoint's model files at its pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make lightonocr3-model`) |
| `lightonocr3_engine.py` | The adapter |
| `engine.toml`, `presets/*.toml` | The six configurations (above) for `ocrbench run start` |

## Versions and license

- **Models** (all published 2026-10-09, [Apache 2.0](https://huggingface.co/lightonai/LightOnOCR-3-4B)):

  | Checkpoint | Revision | Architecture | Weights (bfloat16) | `model.safetensors` SHA-256 |
  | --- | --- | --- | --- | --- |
  | [`lightonai/LightOnOCR-3-0.8B`](https://huggingface.co/lightonai/LightOnOCR-3-0.8B) | `36d8636925327c671bd8e479c517a004f7566bd9` | Qwen3.5 (`Qwen3_5ForConditionalGeneration`): a 12-layer vision encoder and a 24-layer hybrid language model (gated linear attention with a full-attention layer every fourth) | 1.7 GB | `ae46aa6f…a3efdf` |
  | [`lightonai/LightOnOCR-3-1B`](https://huggingface.co/lightonai/LightOnOCR-3-1B) | `23d544085c886bbdace647bcfab91203299417e2` | LightOnOCR-2's (`LightOnOCRForConditionalGeneration`): a Pixtral vision encoder and a 28-layer Qwen3 decoder | 2.0 GB | `32879861…358dbb` |
  | [`lightonai/LightOnOCR-3-4B`](https://huggingface.co/lightonai/LightOnOCR-3-4B) | `b71010f095dc3735de7ee6b9969bff8f8f50b39a` | Qwen3.5, as the 0.8B with a 24-layer vision encoder and a 32-layer language model | 9.1 GB | `6b0dab74…aaeb4e67` |

  The weights' SHA-256 is the Hugging Face LFS object ID. Each `models-<checkpoint>.sha256` pins the files vLLM loads: the configuration, generation configuration, weights, chat template, processor configuration and tokenizer files (not the model card, logo or benchmark results). There is no model code to download: vLLM implements both architectures, so the server runs without `--trust-remote-code` (the model cards' `vllm serve` example passes it, but the checkpoints contain no code). The model cards list no languages beyond English and French examples and "20+ languages" of synthetic training pages; Czech, Fraktur and Glagolitic are not named.
- **Serving path:** LightOn's [LightOnOCR repository](https://github.com/lightonai/LightOnOCR) at commit `36755d461be079737860a5f03ae0c803501269e9` (2026-10-08), whose README gives the `vllm serve` flags per model and whose client (`lightonocr/client.py`, `models.py`, `render.py`) sends the request this adapter reproduces. Its `vllm` extra pins **vLLM 0.30.0** with **transformers 5.16.1**: transformers 5.17 renamed `PixtralRotaryEmbedding`, which vLLM 0.30.0 imports for the 1B architecture, and vLLM 0.27.1 makes every model loop.
- **Inference:** vLLM **0.30.0** with torch **2.13.0** (CUDA 13.0), transformers **5.16.1** and huggingface-hub **1.33.0**, from the new vLLM 0.30 base. NVIDIA driver ≥ 580.
- **Not used:** the model cards' Transformers path (`Qwen3_5ForConditionalGeneration` or `LightOnOcrForConditionalGeneration` with `generate`). vLLM is the serving path that LightOn documents and benchmarks, and that issue #92 asks for. The existing vLLM 0.26 base registers both architectures too, but LightOn has not tested it.
- **Weights are not in the image.** They go to the `/cache` volume in an explicit download step:

  ```sh
  make lightonocr3-model                      # all three, 12.8 GB, into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  make lightonocr3-model CHECKPOINT=4B        # one checkpoint: 0.8B, 1B or 4B
  ```

  This writes `models/lightonocr3/LightOnOCR-3-<size>/<revision>/` under the cache directory and checks every file against the checkpoint's checksums. Runs need `--cache <dir>`. At start-up the adapter checks that the directory of the preset's checkpoint holds exactly these files, all matching, and records the SHA-256 of every file with each sample. vLLM reads the model from that directory, offline (`HF_HUB_OFFLINE=1`).
- **Compiled kernels.** At start-up vLLM compiles the model with torch.compile, captures CUDA graphs and compiles Triton kernels, which it loads as shared objects. The runner's `/tmp` is mounted `noexec`, so these caches are written to `runtime/lightonocr3/` in the cache directory. Later runs reuse them. They are not model files and are not checked; delete the directory to start clean.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-vllm030           # shared vLLM 0.30 base, once
make docker-lightonocr3             # this image (no weights)
make lightonocr3-model              # the pinned weights, 12.8 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/lightonocr3/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench       # or --engine engines/lightonocr3/presets/<preset>.toml
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/lightonocr3 --inspect
make lightonocr3-smoke              # opt-in: build the image, run every preset on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in the configurations), and the server sees only the GPU of `device` (`CUDA_VISIBLE_DEVICES`). If the requested GPU is not visible, every sample fails with `engine-error`; there is no CPU fallback.

## Pipeline and parameters

1. **Server.** The adapter checks the model files and starts `vllm serve` on the checkpoint's directory, on `127.0.0.1` and a free port, with LightOn's flags for that checkpoint: one image per prompt (`--limit-mm-per-prompt '{"image": 1}'`), no multimodal processor cache and no prefix caching, a context (`--max-model-len`) of **18,000** tokens for 0.8B and 4B and **16,384** (the language model's `max_position_embeddings`) for 1B, and, for 0.8B and 4B, `--default-chat-template-kwargs '{"enable_thinking": false}'` (their Qwen3.5 chat template can think; the models were trained without it). It adds served model name `lightonocr3`, bfloat16 and at most 8 concurrent sequences, as pages are sent one at a time.

   - `VLLM_USE_FLASHINFER_SAMPLER=0`: FlashInfer's top-k/top-p sampling kernels would be compiled at start-up with `nvcc`, which the image (like the base) does not contain. vLLM's PyTorch sampler is used instead.

   The adapter waits until `/health` answers (at most `startup_seconds`). The full command, the environment it sets, the GPU and its compute capability are recorded per sample (`extra.environment`).
2. **Image.** The image is decoded as stored (EXIF orientation is not applied; other modes are converted to RGB) and, as LightOn's client does (`render.fit`), scaled down with Lanczos so that its longest side is at most **2048** pixels for 0.8B and 4B and **1540** for 1B, keeping the aspect ratio; a smaller image is not enlarged. The size sent is recorded as `extra.sent_image`. A 4194 × 6192 newspaper page becomes 1387 × 2048 for 0.8B and 4B and 1043 × 1540 for 1B.
   - **0.8B and 4B** (Qwen3.5 image processor, 16-pixel patches merged 2 × 2): the image is rounded to multiples of 32 pixels and each 32 × 32 block is one image token, so a 2048-pixel page has about 2,000–4,000 image tokens. The checkpoint's processor accepts up to 16.7 million pixels; the 2048-pixel limit is the client's, and the training resolution according to LightOn. LightOn's best olmOCR-bench scores used 400 DPI renderings capped at 5 megapixels (about 4,800 image tokens); this engine keeps the client default, not a benchmark-specific resolution.
   - **1B** (Pixtral, 14-pixel patches merged 2 × 2): the processor would also scale to 1540 pixels; each 28 × 28 block is one image token, about 1,800–2,300 on the dev-01 pages.
   - **Tall images** keep their aspect ratio: a 1600 × 6975 strip of three book pages becomes 470 × 2048 (or 353 × 1540), so its text gets small. Pages are not tiled.
3. **Request.** One chat request per page, as `lightonocr.client.LightOnOCR.ocr` sends it: a user message with the image as a PNG data URL and, in grounding mode, a text part `grounding`; `chat_template_kwargs` `{"enable_thinking": false}`. The prompts are the model cards' only two: "Other instructions are out of distribution". The sample `language` is not used.
   - **Sampling** is the client's for LightOnOCR-3: temperature **0.2** and top_p **1.0**, with a fixed `seed` (0) per request so that a sample does not depend on the samples before it. LightOn's README advises temperature 0.1 rather than greedy decoding and the model cards' examples use 0.2 with top_p 0.9; the client's values were chosen for #92 because they are what LightOn's own tool sends. The 1B checkpoint's `generation_config.json` (temperature 0.2, top_p 0.9) is overridden by the request.
   - **Generation budget.** Without `max_tokens` vLLM generates until the end token or the end of the context, as the client does: about 13,000–16,000 tokens after the image.
   - LightOn's client by default repairs escaped dollar delimiters (`\$x\$` to `$x$`) in the output. That is a markup repair, so it is not done in the container ([CONTRIBUTING.md](../../CONTRIBUTING.md#shared-text-conversion-for-engine-outputs)); the native output is kept as the model wrote it.

`[params]` in the configurations are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample; `checkpoint` and `mode` have no default.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | none, required | `0.8B`, `1B` or `4B` |
| `mode` | none, required | `plain` (the image alone; Markdown) or `grounding` (the image and `grounding`; labeled boxes) |
| `device` | `cuda:0` | `cuda:<n>` |
| `max_tokens` | none: up to the end of the context | Generated tokens per page before the output counts as truncated; below the context. The context also holds the image tokens, so vLLM caps a larger value at the tokens the image leaves, and the sample is reported as stopped at the end of the context |
| `temperature`, `top_p` | 0.2, 1.0 | Sampling (above) |
| `seed` | 0 | vLLM's per-request seed |
| `longest_edge` | none: 2048 (0.8B, 4B) or 1540 (1B) | Images with a longer side are scaled down to this many pixels. The 1B processor scales down to 1540 pixels itself |
| `gpu_memory_utilization` | **0.6**, also the maximum ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)) | Share of the GPU's memory vLLM reserves for the weights, CUDA graphs and key/value cache. vLLM refuses to start when less is free |
| `max_num_seqs` | 8 | vLLM's limit of concurrent sequences; requests are sent one at a time |
| `startup_seconds` | 900 | Time allowed for the server to become ready |

## Conversion to the prediction

**Plain mode.** `page.md` is the model output, unchanged: the native output and the primary output. The shared Markdown converter turns it into the standardized prediction on the host ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), as for LightOnOCR-2: headings, paragraphs, list items, tables (one row per line, cells separated by tabs), `$$` formulas (TeX source) and HTML embedded in Markdown become typed regions with their text only, without positions and without physical lines. The model gives no geometry, so none is invented.

**Grounding mode.** The adapter writes a positioned `prediction.json`. Every block becomes one region, in the model's order, which is its reading order; `reading_order` lists them.

- **Blocks.** As LightOn's parser (`lightonocr.grounding.parse_blocks`) splits them: each marker `![label](x1,y1,x2,y2)` owns the text up to the next marker, so a table or a list spans several lines and the text may start on the line after the marker. A marker is a label (letters, digits, `_`, `-`) with an optional `+`, and parentheses holding only numbers, commas and spaces; a Markdown image such as `![Figure](fig.png)` is therefore text, not a block. Text before the first marker, or an output without markers, becomes one region without a polygon (type `other`).
- **Positions.** The box is read as written, never evaluated, and scaled to the input image as stored: x × width / 1000 and y × height / 1000, as LightOn's `Block.to_pixels` does. The coordinates are relative to the image, so the scaling before the request does not change them. (The Qwen3.5 processor rounds the sent image to multiples of 32 pixels, which stretches it by at most 16 pixels on a side.) The corners are sorted and clipped to the image; the box becomes a four-point polygon in float pixel coordinates. A marker whose box is not four integers gives a region without a polygon; no position is invented. There are no line positions.
- **Continuations.** A `+` after the label (`text+`) marks a block that continues the previous one, for example in the next column. It stays its own region with its own box; `native_type` keeps the `+`, and `extra.continuation_blocks` counts them. LightOn's `to_markdown` joins such blocks into one paragraph; here they are separate regions, which does not change the page text beyond a paragraph break.
- **Types.** `title` and `section` become `heading`, `list` `list-item`, `caption` `caption`, `footnote` `footnote`, `header` and `footer` `page-header` and `page-footer`, `page_number` `page-number`, `table` `table`, `formula` `formula`, `text` and `aside_text` (side notes) `text`, and `image`, `header_image`, `footer_image` and `chart` `figure`; other labels (`code`) `other`. The label as written is kept as `native_type`.
- **Pictures and charts are not scored.** An `image` block holds a short description of the picture and a `chart` block an HTML table of data read from the chart: neither is text on the page. Following the maintainer's decision for Chandra OCR 2 ([#53](https://github.com/michal-hradis/ocr_benchmarking/issues/53)), these blocks (with `header_image` and `footer_image`) keep their box and label but get no text. Their content stays in `output.md` and `layout.json`, and `extra.unscored_blocks` counts them.
- **Text.** Every other block's content (Markdown; tables as HTML) is written as region **markup** (`{"format": "markdown", "content": …}`), not converted in the container. The host converts it with the shared Markdown rules when it validates the prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md#markup-in-positioned-regions), [ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)) and records `conversion` `{"from": "markdown", "version": 1}`. Converter version 1 reads `\[…\]` and `\(…\)` math delimiters as Markdown escapes, so they lose their backslashes (a documented limitation).
- **Page text.** The engine writes an empty page text; the host builds it from the converted regions, joined by blank lines.

## Truncation, repetition, failures, memory and time

- A request stops at the end token, at `max_tokens` or at the end of the context. Without an end token (stop reason `length`) the sample is `truncated`, and the output is scored as produced, including an unfinished last block ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). The message says which limit was reached: the end of the context (prompt plus generated tokens fill it) or `max_tokens`.
- **Repetition.** There is no repetition penalty or n-gram processor in LightOn's serving path. The adapter checks whether the output ends in a loop (`extra.tail_repetition`): a unit of fewer than 64 characters repeated 8 times in a row, or a longer one of up to 4,096 characters (a sentence or paragraph) repeated 3 times. It adds "the output ends in a repetition loop" to the message of a truncated sample. The output is not trimmed.
- An empty output with an end token is an empty page (`success`, with the message `the model returned an empty page`). Any other stop reason fails the sample (`engine-error`). The native output, the parsed blocks and the request diagnostics in `extra` are kept.
- A failed request (for example an out-of-memory error in the server) fails only that sample. If the server exits, that sample and every later one fail with its log tail.
- Every sample fails when the GPU is unusable, the parameters are invalid, the checkpoint's model directory is missing or does not match its checksums, or the server does not start (the message has the end of its log).

**Memory.** vLLM reserves `gpu_memory_utilization` of the GPU's memory up front: 0.6 × 24 GB = 14.1 GiB on an RTX 4090. Measured at start-up (vLLM's log):

| Checkpoint | Weights | Key/value cache | Cache in tokens |
| --- | ---: | ---: | ---: |
| 0.8B | 1.7 GiB | 10.8 GiB | 846,000 (most layers keep a fixed-size linear-attention state instead) |
| 1B | 1.9 GiB | 11.5 GiB | 108,000 |
| 4B | 8.6 GiB | 3.6 GiB | 104,700 |

Each holds more than five full contexts, so all three checkpoints fit an RTX 4090 at the benchmark's share. The peak memory actually used per page is not measured.

**Time.** See the dev-01 results below. `extra.request_seconds` is the time of the request (inference), `extra.tokens_per_second` the generated tokens per second of it, and the sample's `seconds` the whole page including image encoding and parsing. Start-up (server and model load, compilation, CUDA graph capture) took 66–98 s with the compile caches of earlier runs, and 208 s for the 4B checkpoint's first start; it is not part of the per-sample engine time.

## Outputs

| File | Mode | Content |
| --- | --- | --- |
| `page.md` | plain | The model output, unchanged: the native output and the primary output, converted on the host |
| `output.md` | grounding | The model output, unchanged, with its markers: the native output |
| `layout.json` | grounding | The image size and the parsed blocks: the marker as written, label, continuation, box in 0–1000, box in pixels, content and the character offsets of the marker |
| `prediction.json` | grounding | The positioned prediction with region markup (above): the primary output |
| `result.json` | both | Status, processing time and, in `extra`: `finish_reason`, `tail_repetition`, `prompt_tokens`, `generated_tokens`, `max_tokens`, `request_seconds`, `tokens_per_second`, `source_image`, `sent_image`; in grounding mode `blocks`, `labels`, `blocks_without_box`, `continuation_blocks`, `unscored_blocks`; the vLLM, transformers, huggingface-hub, torch and CUDA versions, the GPU and its compute capability, the server command and environment, the start-up time, the model (repository, revision, file checksums, the configured `model`, the mode, the prompt, the chat template arguments, the context and the longest edge) and the parameters |

## Results on dev-01

Measured in runs `dev01-lightonocr3-<preset>-v1` (image `sha256:4e6cbd21…`) on an RTX 4090; the scores are in the pull request for [#92](https://github.com/michal-hradis/ocr_benchmarking/issues/92). The development set is seven pages and not representative.

- **Book pages.** The four 1600-pixel German and Latin prints ended normally with every preset, at 0.6–12.3 % CER; the checkpoints differ little there. In grounding mode the three checkpoints give nearly the same boxes for the same page, mostly with running heads, page numbers and library stamps as their own blocks. No model writes a long s (ſ) in Fraktur.
- **Newspapers.** The two Czech newspaper pages (4194 × 6192 and 3509 × 5239) are sent at 1387 × 2048 (1043 × 1540 for 1B), so their text is small. On `bd497a62` every preset except 4B plain ends in a loop that repeats a sentence or paragraph until the end of the context (`truncated`), after reading part of the page; on `3e6309d0` only 1B grounding does. **4B in plain mode returns an empty page (an immediate end token) on both newspapers.** The same happened with seeds 1 and 2 and with greedy decoding, while 4B in grounding mode reads them.
- **Glagolitic page.** No checkpoint reads the script: 1B and 4B write Ethiopic and 0.8B Devanagari syllables until the end of the context (CER far above 100 %, which dominates the micro-averaged CER). The one exception is 1B in grounding mode, which returns an empty page.
- **Repetition.** No preset has a repetition guard upstream; every truncated page ends in a loop (`extra.tail_repetition`).
- **Time.** At about 100 generated tokens/s for 4B, 300 for 1B and 400 for 0.8B, a book page took 4–11 s (4B) or 1.4–3.3 s (0.8B, 1B). A newspaper page that runs to the end of the context took 157 s (4B) or 39–47 s.
- **Tall images.** A test image of one 1600 × 2325 book page repeated three times (1600 × 6975, not committed) is sent as 470 × 2048 (353 × 1540 for 1B). 4B in grounding mode read only the first two copies, in order, and its boxes covered x 488–910 instead of the text's 100–930, which it gets right on the single page. 1B read all three copies in order, with more misreadings and one paragraph it made up. The image path works for tall pages, but very narrow images are outside what the models handle well.

## Tests

`tests/test_lightonocr3.py` runs offline in `make test`:

- the six presets (each names its checkpoint, revision and mode), parameter validation, the server command against LightOn's flags per checkpoint, the request (image only or image and `grounding`, sampling, seed, thinking disabled, no `max_tokens` by default), the image scaling and the pinned model files of each checkpoint;
- the parsing of grounding markers (boxes as written, sorted and clipped, invalid boxes, continuations, text before the first marker, Markdown images that are not markers), the conversion of blocks to positioned regions with markup, including picture and chart blocks without text, a box outside the image and a block without a valid box, and its host-side conversion against the expected plain text; the repetition check;
- the checksums and the download script;
- the engine protocol through the runner, with a fake `vllm` executable that serves fixed outputs over the loopback interface and checks the command line and every request, and a fake `torch` module (`tests/fixtures/lightonocr3/fake`). In grounding mode (4B) the outputs are hand-written blocks (`tests/fixtures/lightonocr3/output.md`, not real model output), an empty output (success), text without markers, a large page whose boxes must map to the original pixels, a newspaper page cut off at the end of the context in a repetition loop and at `max_tokens`, an aborted request with its diagnostics, a failed request and a server that exits. In plain mode (1B) a hand-written Markdown page (`page.md`), a large page scaled to 1540 pixels, a truncated newspaper page and an empty page. Both runs are evaluated, and the grounding run is reported, including the truncated page against a 2,000-line reference;
- start-up failures: invalid or missing parameters, no GPU, missing, changed or extra model files and a server that exits during start-up.

`make lightonocr3-smoke` (after `make lightonocr3-model`) builds the image and runs each of the six presets with the real models on the GPU through `ocrbench run start`, `evaluate` and `report` on a synthetic full page, a paragraph crop and a blank page. In grounding mode it checks that every region of the text samples has a polygon; in both modes that the markup was converted on the host.
