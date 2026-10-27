# Unlimited-OCR engine (page image to layout blocks, vLLM, GPU)

[Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) by Baidu ([#94](https://github.com/michal-hradis/ocr_benchmarking/issues/94)) is a long-horizon page parser built on DeepSeek-OCR. It has the same vision stack (a SAM-ViT-B and CLIP-L DeepEncoder with a linear projector) with a DeepSeek-V2 mixture-of-experts language model (12 layers, 64 routed and 2 shared experts, plain multi-head attention), about 3.3 billion parameters. While it generates, the image and prompt tokens stay visible, but the generated tokens attend only to a sliding window of the last 128 generated tokens (reference sliding window attention, R-SWA). This keeps the cost per token flat on long outputs. It reads a whole page image and writes the page as a sequence of **layout blocks** in reading order. Each block is a tag with a label and a box, scaled to [0, 999] of the image, followed by its Markdown (tables as HTML):

```text
<|det|>title [100, 80, 900, 120]<|/det|>The Daily Chronicle
<|det|>text [100, 130, 900, 300]<|/det|>The quick brown fox jumps over …
```

There are no text lines.

The engine runs **vLLM inside its container** as a child process of the adapter, on the loopback interface only ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)), with vLLM's built-in model for the checkpoint. The image uses the vLLM 0.26 base unchanged ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-unlimited-ocr:1.0-v1` from `ocrbench/base-cu130-vllm026:vllm0.26.0-py3.12-v1`, without weights and without extra packages |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make unlimited-ocr-model`) |
| `unlimited_ocr_engine.py` | The adapter |
| `engine.toml` | The configuration (below) for `ocrbench run start` |

## Versions and license

- **Model:** [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) (Unlimited-OCR 1.0) at commit `07dea832e22aefee32ad281d4b80551282e1c168` (2026-07-29). The weights are 6.67 GB of bfloat16 in `model-00001-of-000001.safetensors`, whose SHA-256 is the Hugging Face LFS object ID (`2bc48a7a…b2855fc6`). `models.sha256` pins the 8 files vLLM loads: the configuration, the weight index and weights, the processor configuration, the tokenizer and the license. The model's own Python code is not downloaded: vLLM implements the model, its image processor and its configuration itself, so the server runs without `--trust-remote-code`.
  - **Why this revision.** The latest revision (`8249d20b`, 2026-10-14, "compatibility with transformers") renames `model_type` in `config.json` from `unlimited-ocr` to `unlimited_ocr` for a future Transformers release, adds a chat template and a duplicate `model.safetensors`, and leaves the weights unchanged (the same LFS object). vLLM 0.26 registers only `unlimited-ocr` and does not load the newer revision.
  - **Unlimited-OCR 2** (`baidu/Unlimited-OCR-2`, released 2026-10-27) adds sparse reference sliding window attention (SR-SWA) for multi-page documents. It runs only through its own Transformers code or Baidu's SGLang build, not vLLM 0.26, and is not part of this engine.
- **License:** [MIT](https://huggingface.co/baidu/Unlimited-OCR/blob/07dea832e22aefee32ad281d4b80551282e1c168/LICENSE) for the model and code.
- **Inference:** vLLM **0.26.0**, the first stable release with the built-in `UnlimitedOCRForCausalLM` (the [vLLM recipe](https://recipes.vllm.ai/baidu/Unlimited-OCR) says 0.25.0+ and gives a dedicated `vllm/vllm-openai:unlimited-ocr` image), with torch **2.11.0** (CUDA 13.0), transformers **5.14.1** and huggingface-hub **1.24.0** from the base. NVIDIA driver ≥ 580.
- **Not used:** the model card's Transformers path (`AutoModel` with `trust_remote_code`, tested upstream with transformers 4.57.1 and torch 2.10.0) and Baidu's SGLang build. vLLM is the serving path that issue #94 asks for, and it needs no extra packages in the image.
- **Weights are not in the image.** They go to the `/cache` volume in an explicit download step:

  ```sh
  make unlimited-ocr-model                      # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/unlimited-ocr/Unlimited-OCR/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks that the directory holds exactly these files, all matching, and records the SHA-256 of every file with each sample. vLLM reads the model from that directory, offline (`HF_HUB_OFFLINE=1`).
- **Compiled kernels.** At start-up vLLM compiles the model with torch.compile, captures CUDA graphs and compiles Triton kernels, which it loads as shared objects. The runner's `/tmp` is mounted `noexec`, so these caches are written to `runtime/unlimited-ocr/` in the cache directory. Later runs reuse them. They are not model files and are not checked; delete the directory to start clean.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-vllm026           # shared vLLM 0.26 base, once
make docker-unlimited-ocr           # this image (no weights)
make unlimited-ocr-model            # the pinned weights, 6.7 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/unlimited-ocr/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/unlimited-ocr --inspect
make unlimited-ocr-smoke            # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`), and the server sees only the GPU of `device` (`CUDA_VISIBLE_DEVICES`). If the requested GPU is not visible, every sample fails with `engine-error`; there is no CPU fallback.

## Configuration

There is one configuration, `unlimited-ocr` (`engine.toml`): the model card's single-page document parsing, one request per page, generating until the end token or the end of the model's context. No second long-output preset is configured: this one already allows the whole 32,768-token context.

## Pipeline and parameters

1. **Server.** The adapter checks the model files and starts `vllm serve` on the model directory, on `127.0.0.1` and a free port, with the recipe's settings: the n-gram logits processor `vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor`, no prefix caching and no multimodal processor cache. It adds served model name `unlimited-ocr`, bfloat16, a 32,768-token context (the language model's `max_position_embeddings`, the model card's `max_length`), one image per prompt and at most 8 concurrent sequences, as pages are sent one at a time. vLLM chooses the attention backend for R-SWA itself: FlashAttention 4 where available (Hopper), otherwise Triton attention with an R-SWA mask, which it uses on an RTX 4090. The vision encoder uses FlashAttention. vLLM uses its fallback chat template for this model, which writes the begin-of-sentence token and the message text.

   - `VLLM_USE_FLASHINFER_SAMPLER=0`: vLLM 0.26 would compile FlashInfer's sampling kernels at start-up with `nvcc`, which the image (like the base) does not contain. vLLM's PyTorch sampler is used instead; with greedy decoding both pick the most probable token.

   The adapter waits until `/health` answers (at most `startup_seconds`). The full command, the environment it sets, the GPU and its compute capability are recorded per sample (`extra.environment`).
2. **Request.** For each sample the image is decoded as stored (EXIF orientation is not applied; other modes are converted to RGB) and sent with the prompt as PNG in one chat request. The prompt is the model card's and the recipe's single-image prompt, which must start with the image placeholder:

   > `<image>document parsing.`

   Decoding is the model card's: greedy (temperature 0), no repeated 35-token n-gram within the last 128 generated tokens (`vllm_xargs` `ngram_size` 35, `window_size` 128) and `skip_special_tokens` false so that the layout tags are kept. Without `max_tokens`, vLLM generates up to the end of the context: 32,768 tokens minus the prompt. The sample `language` is not used.
3. **Image processing** (vLLM's, crop or "gundam" mode, the model card's default for a single image): a 1024 × 1024 global view (the whole image scaled to fit, keeping its aspect ratio, padded with gray), plus, when the image is larger than 640 pixels on a side, local 640 × 640 tiles. The image is stretched to the tile grid of 2 to 32 tiles closest to its aspect ratio. The global view gives 273 image tokens and each tile 100, plus a separator per tile row. A 1600-pixel book page gave about 900 prompt tokens and a newspaper page about 2,700.
   - **Tall images.** A tall image gets a tall grid (up to 1 × 32 tiles). A test page made of one 1600 × 2325 book page repeated three times (1600 × 6975, not committed) gave 2,167 prompt tokens and was read top to bottom, with all three copies, in 6.3 s.

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>` |
| `max_tokens` | none: up to the end of the context | Generated tokens per page before the output counts as truncated; below the 32,768-token context. The context also holds the image tokens, so vLLM 0.26 caps a larger value at the tokens the image leaves, and the sample is reported as stopped at the end of the context. The recipe's client example uses 8,192 |
| `gpu_memory_utilization` | **0.6**, also the maximum ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)) | Share of the GPU's memory vLLM reserves for the weights, CUDA graphs and key/value cache. vLLM refuses to start when less is free |
| `max_num_seqs` | 8 | vLLM's limit of concurrent sequences; requests are sent one at a time |
| `startup_seconds` | 900 | Time allowed for the server to become ready |

## Conversion to the prediction

The adapter writes a positioned `prediction.json`. Every block becomes one region, in the model's order, which is its reading order; `reading_order` lists them.

- **Positions.** The box `[x1, y1, x2, y2]` is read as written, never evaluated, and scaled to the input image as stored: x × width / 999 and y × height / 999, as the model card's own drawing code does. The corners are sorted and clipped to the image; the box becomes a four-point polygon in float pixel coordinates. A tag whose box is not exactly four numbers gives a region without a polygon; no position is invented. Text before the first tag, or an output without tags, becomes one region without a polygon (type `other`). There are no line positions.
- **Malformed tags.** A tag ends at its closing `<|/det|>`. Without one, it ends at the end of its line or at the next `<|det|>`, whichever comes first, so a later tag keeps its label and box; the block is marked `unterminated` in `layout.json` and counted in `extra.unterminated_tags`. Its first line is read as the label and box, and the following lines are its content. A header cut off at the end of the output therefore gives a region with neither text nor position.
- **Grounding references.** The [vLLM recipe](https://recipes.vllm.ai/baidu/Unlimited-OCR) says the raw output can also carry `<|ref|>…<|/ref|>` wrappers and says to unwrap them. A reference directly before a tag, as in DeepSeek-OCR's `<|ref|>title<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>`, gives the tag its label, and the tag holds only the box (single or double brackets). If the reference text is not a label (one word), it is kept as the block's text and the block gets no label. A tag with several boxes gives no polygon. Anywhere else, a reference is unwrapped and its text kept. Grounding delimiters without a partner are removed from the region markup. `output.txt` keeps them all, and `page.md` stays the model card's rendering. None of the dev-01 outputs contain a reference or an unterminated tag.
- **Types.** `title` becomes `heading`, `page_number` `page-number`, `header` and `footer` `page-header` and `page-footer`, `image`, `figure`, `chart` and `seal` `figure`, `image_caption` and `table_caption` `caption`, the footnote labels `footnote`, `equation` `formula`, and `text` and `table` themselves; other labels `other`. The label is kept as `native_type`. On dev-01 the model used only `text`, `title`, `page_number`, `header` and `footer`.
- **Text.** The block's content (Markdown; tables as HTML) is written as region **markup** (`{"format": "markdown", "content": …}`), not converted in the container. The host converts it with the shared Markdown rules when it validates the prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md#markup-in-positioned-regions), [ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)): emphasis markers add nothing, a table becomes one line per row with tab-separated cells and `$…$` math keeps its TeX. The standardized prediction records `conversion` `{"from": "markdown", "version": 1}`. Converter version 1 reads `\[…\]` and `\(…\)` math delimiters as Markdown escapes, so they lose their backslashes (a documented limitation). Image blocks have no text and are regions without text.
- **Page text.** The engine writes an empty page text; the host builds it from the converted regions, joined by blank lines.

## Truncation, repetition, failures, memory and time

- A request stops at the end token, at `max_tokens` or at the end of the context. Without an end token (stop reason `length`) the sample is `truncated`, and every block is scored as produced, including the unfinished last one ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). The message says which limit was reached: the end of the context (prompt plus generated tokens fill 32,768) or `max_tokens`.
- **Repetition.** The n-gram processor forbids exact repeats of 35 tokens within 128, but a longer loop can still fill the context. The adapter checks whether the output ends with a unit of at most 256 characters repeated 8 times in a row (`extra.tail_repetition`) and adds "the output ends in a repetition loop" to the message of a truncated sample. The output is not trimmed.
- An empty output with an end token is an empty page (`success`, with the message `the model returned an empty page`). Any other stop reason fails the sample (`engine-error`). `output.txt`, the parsed outputs and the request diagnostics in `extra` are kept.
- A failed request (for example an out-of-memory error in the server) fails only that sample. If the server exits, that sample and every later one fail with its log tail.
- Every sample fails when the GPU is unusable, the parameters are invalid, the model directory is missing or does not match `models.sha256`, or the server does not start (the message has the end of its log).

**Memory.** vLLM reserves `gpu_memory_utilization` of the GPU's memory up front: 0.6 × 24 GB = 14.1 GiB on an RTX 4090. The bfloat16 weights take 6.2 GiB, the peak activation 2.8 GiB, and the key/value cache gets 4.9 GiB (86,000 tokens, more than two full contexts). The peak memory actually used per page is not measured.

**Time.** See the dev-01 results below. `extra.request_seconds` is the time of the request (inference), `extra.tokens_per_second` the generated tokens per second of it, and the sample's `seconds` the whole page including image encoding and parsing.

## Outputs

| File | Content |
| --- | --- |
| `output.txt` | The model output, unchanged, with its layout tags: the native output |
| `layout.json` | The image size and the parsed blocks: the tag as written, a reference before it, label, box in [0, 999], box in pixels, content, the character offsets of the tag and whether it is unterminated |
| `page.md` | The model card's Markdown rendering (its `remove_det` for OmniDocBench): tags removed, image blocks dropped, blocks separated by blank lines |
| `prediction.json` | The positioned prediction with region markup (above): the primary output |
| `result.json` | Status, processing time and, in `extra`: `finish_reason`, `tail_repetition`, `prompt_tokens`, `generated_tokens`, `max_tokens`, `request_seconds`, `tokens_per_second`, `source_image`, `blocks`, `labels`, `blocks_without_box`, `unterminated_tags`, the vLLM, transformers, huggingface-hub, torch and CUDA versions, the GPU and its compute capability, the server command and environment, the start-up time, the model (repository, revision, file checksums, the configured `model`, the prompt, the sampling and the context) and the parameters |

## Results on dev-01

Measured in run `dev01-unlimited-ocr-1.0-v1` on an RTX 4090; the scores are in the pull request for [#94](https://github.com/michal-hradis/ocr_benchmarking/issues/94).

- **All seven pages ended with the end token**; none reached the end of the context, and no output ends in a repetition loop. The longest output was 7,606 generated tokens (a Czech newspaper page), less than a quarter of the context.
- **Time.** Start-up (server and model load, compilation, CUDA graph capture) took 73 s with the compile caches from earlier runs in the cache directory; it is not part of the per-sample engine time. A 1600-pixel book page took 2.0–3.2 s and a newspaper page 18–21 s, at 250–490 generated tokens/s. The images give 900–3,150 prompt tokens.
- **Newspapers.** The two Czech newspaper pages (4194 × 6192 and 3509 × 5239) are read through the 1024-pixel global view and 640-pixel tiles. The model returns one block per paragraph, with the paragraph's own line breaks, mostly column by column from left to right, which the reference order does not always follow (strict CER 64.7 % against 16.1 % order-independent on the first page).
- **Glagolitic page.** The model does not read the script: it wrote Tibetan-script text with a few syllables repeated many times. The n-gram processor does not stop this, because the repeated syllables vary slightly, and the output ended with the end token.
- **Book pages.** The four 1600-pixel German and Latin prints ended normally, with 7–20 blocks each, including running headers, page numbers and footers as their own blocks. The model writes no long s (ſ) in Fraktur.

## Tests

`tests/test_unlimited_ocr.py` runs offline in `make test`:

- parameter validation, the server command against the recipe's settings, the request (prompt, image, decoding, the n-gram arguments, no `max_tokens` by default) and the pinned model files;
- the parsing of layout tags (boxes as written, sorted and clipped, invalid boxes, text before the first tag, unterminated and cut-off tags, grounding references before tags and in text, stray delimiters), the conversion of blocks to positioned regions with markup, including an image block without text, a box outside the image and a block without a valid box, and its host-side conversion against the expected plain text; the model card's Markdown rendering; the repetition check;
- the model checksums and the download script;
- the engine protocol through the runner, with a fake `vllm` executable that serves fixed outputs over the loopback interface and checks the command line and every request, and a fake `torch` module (`tests/fixtures/unlimited-ocr/fake`). The outputs are hand-written blocks (`tests/fixtures/unlimited-ocr/output.txt`, not real model output), an empty output (success), text without tags, a newspaper page cut off at the end of the context in a repetition loop, at `max_tokens` and with a `max_tokens` beyond the context the image leaves, an aborted request with its diagnostics, a failed request and a server that exits. That run is then evaluated and reported, including the truncated page against a 2,000-line reference;
- start-up failures: invalid parameters, no GPU, missing, changed or extra model files and a server that exits during start-up.

`make unlimited-ocr-smoke` (after `make unlimited-ocr-model`) builds the image and runs the real model on the GPU through `ocrbench run start`, `evaluate` and `report` on a synthetic full page, a paragraph crop and a blank page. It checks that every region of the text samples has a polygon and that the markup was converted on the host.
