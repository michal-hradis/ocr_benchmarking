# Surya 2 engine (full-page OCR to layout blocks, vLLM, GPU)

[Surya 2](https://github.com/datalab-to/surya) by Datalab ([#67](https://github.com/michal-hradis/ocr_benchmarking/issues/67)) is the second generation of Surya (0.20 and newer). It replaces Surya's line detector and line recognizer with one 650-million-parameter vision-language model, [`surya-ocr-2`](https://huggingface.co/datalab-to/surya-ocr-2). Surya runs that model behind an OpenAI-compatible server. Its default OCR reads a whole page image in one request and generates the page as **HTML layout blocks** in reading order. Each block is a `<div>` with a label, a box and the block's HTML. There are no text lines.

The engine runs **vLLM inside its container** as a child process of the adapter, on the loopback interface only ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)). It never starts Surya's default sibling Docker container. The image uses the vLLM base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol). Surya 0.17.1, the first generation, is a separate engine ([engines/surya](../surya/README.md)).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-surya2:0.22.1-v1` from `ocrbench/base-cu130-vllm:vllm0.20.1-py3.12-v1`, without weights |
| `requirements.in`, `overrides.txt`, `requirements.txt` | Surya and its dependencies, hash-locked against the base (`make docker-lock`); `overrides.txt` replaces two of Surya's pins (below) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make surya2-model`) |
| `surya2_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions and license

- **Surya 0.22.1** (2026-07-20, wheel SHA-256 `f84cdba1…69a2d`). The adapter uses Surya's own full-page prompt, its image scaling (`scale_to_fit`), its request format (`_build_messages`), its output parser (`parse_full_page_html`) and its post-processing helpers, imported from the installed package. No Surya code is copied into this repository. The code license is **Apache-2.0**.
  - Surya pins `pillow<11` and `opencv-python-headless==4.11.0.86`, which conflict with vLLM's Pillow 13 and OpenCV 5. `overrides.txt` installs the base's versions instead. The Surya modules the adapter imports do not load OpenCV (checked in the image) and use Pillow only to crop, resize and encode images.
- **Model:** [`datalab-to/surya-ocr-2`](https://huggingface.co/datalab-to/surya-ocr-2) at commit `3b3d4cdf88d6928b0acdc75181b13206ea67c4a3`. It is a `Qwen3_5ForConditionalGeneration` model: a 12-layer vision encoder with 16-pixel patches merged 2 × 2, and a 24-layer, 1024-wide language model that mixes linear and full attention, with one multi-token-prediction (MTP) layer. The weights are 1.37 GB of bfloat16 in one `model.safetensors`, whose SHA-256 is the Hugging Face LFS object ID. `models.sha256` pins all nine files the server loads; `README.md`, `LICENSE`, `.gitattributes` and the example images are not downloaded.
- **Weights license:** Datalab's [modified AI Pubs Open RAIL-M license](https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/LICENSE). Check it before using the weights.
- **Inference:** vLLM **0.20.1**, the server version Surya 0.22.1 starts (`vllm/vllm-openai:v0.20.1`), with the torch **2.11.0** (CUDA 13.0) it pins and transformers **5.12.1**, the oldest release Surya accepts. NVIDIA driver ≥ 580.
- **Weights are not in the image.** They go to the `/cache` volume in an explicit download step:

  ```sh
  make surya2-model                              # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/surya2/surya-ocr-2/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks that the directory holds exactly these files, all matching, and records the SHA-256 of every file with each sample. vLLM reads the model from that directory, offline (`HF_HUB_OFFLINE=1`).
- **Compiled kernels.** At start-up vLLM compiles the model with torch.compile, captures CUDA graphs and compiles Triton kernels, which it loads as shared objects. The runner's `/tmp` is mounted `noexec`, so these caches are written to `runtime/surya2/` in the cache directory (`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`). Later runs reuse them. They are not model files and are not checked; delete the directory to start clean.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-vllm              # shared vLLM base, once (8.7 GB image)
make docker-surya2                  # this image (no weights)
make surya2-model                   # the pinned weights, 1.4 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/surya2/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id>... -o reports/surya2 --inspect
make surya2-smoke                   # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`), and the server sees only the GPU of `device` (`CUDA_VISIBLE_DEVICES`). If the requested GPU is not visible, every sample fails with `engine-error`; there is no CPU fallback.

## Pipeline and parameters

There is one configuration (`engine.toml`, `name = "surya2"`): Surya's default OCR, `surya_ocr`, which is full-page OCR.

1. **Server.** The adapter checks the model files and starts `vllm serve` on the model directory, on `127.0.0.1` and a free port. It uses the settings Surya passes to its vLLM container for its default GPU type, the RTX 4090: bfloat16, an 18,000-token context (`--max-model-len`), up to 32 sequences and 8,192 batched tokens, prefix caching, CUDA graphs (`--no-enforce-eager`), image pixel limits of 3,136 to 6,291,456 and MTP speculative decoding with two tokens. The adapter waits until `/health` answers (at most `startup_seconds`). The full command is recorded per sample (`extra.environment.server_command`).
2. **Request.** For each sample the image is decoded as stored (EXIF orientation is not applied; other modes are converted to RGB, as Surya's `load_image`). Surya's `scale_to_fit` then scales it to at most 3072 × 2048 pixels' worth (6,291,456) and to whole 28-pixel units; a smaller image is enlarged to at least 50,176 pixels. The adapter sends one chat request with the image as PNG followed by Surya's full-page prompt, byte for byte (`HIGH_ACCURACY_BBOX_PROMPT`; a test checks its SHA-256):

   > OCR this image to HTML. Each block is a div with data-label and data-bbox (x0 y0 x1 y1, normalized 0-1000).

   The request uses Surya's client defaults: temperature 0 (greedy) and top-p 0.1, and `max_tokens` 12,288 (`SURYA_MAX_TOKENS_FULL_PAGE`). vLLM's image processor rescales the PNG again to whole 32-pixel units for the model's patches. The sample `language` is not used.
3. **Parsing.** Surya's `parse_full_page_html` reads the top-level `<div data-label data-bbox>` blocks and their inner HTML; a div without a label or a four-number box is left out. Then, as in Surya's `_full_page_ocr`: blocks whose canonical label carries no text (Surya's `SKIP_CANON_LABELS`: `Image`, `Figure`, `Complex-Block`, which Surya relabels `Figure`, `Diagram` and `Blank-Page`) keep their box but no HTML, and a text-like block whose crop of the page is essentially blank (`is_blank_region`) is dropped as a hallucination.

**Greedy only.** Surya can make other requests that are not used here:

- Its client retries a request whose output ends in a repeated sequence up to three times, sampling at temperature 0.2–0.6 with top-p 0.95.
- When a full-page output loops or cannot be parsed, Surya falls back to a layout request (with vLLM's JSON-schema guided decoding) and per-block OCR requests.

The adapter makes exactly one greedy request per page, so that runs can be repeated. Whether Surya's loop check (`_detect_repeat_loop`) fires is recorded as `extra.repeat_loop`, but nothing is regenerated. Surya does not look at the stop reason; the adapter reports it (below).

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>` |
| `max_tokens` | 12288 | Generated tokens before the output counts as truncated (Surya's full-page limit); vLLM also stops at the 18,000-token context |
| `gpu_memory_utilization` | **0.6**, also the maximum ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)); Surya uses 0.85 | Share of the GPU's memory vLLM reserves for the weights, CUDA graphs and key/value cache. vLLM refuses to start when less is free. Samples are sent one at a time, so the share does not change outputs |
| `mtp` | `true` | MTP speculative decoding with two tokens (Surya's `VLLM_ENABLE_MTP`). With greedy decoding, speculation does not change which tokens are chosen, up to floating-point differences |
| `startup_seconds` | 900 | Time allowed for the server to become ready |

## Conversion to the prediction

The adapter writes the native output unchanged (`output.html`) and a positioned `prediction.json`. Every remaining block becomes one region, in output order, which is the model's reading order; `reading_order` lists them.

- **Positions.** The box, normalized to 0–1000 of the image size, becomes a four-point polygon in float pixel coordinates of the input image. There are no line positions: a block is a paragraph, a heading, a table and so on.
- **Types.** `Section-Header` becomes `heading`, `Page-Header` and `Page-Footer` become `page-header` and `page-footer`, `Equation-Block` becomes `formula`, `Image`, `Figure`, `Complex-Block` and `Diagram` become `figure`, `Text`, `List-Group` and `Bibliography` become `text`, `Caption`, `Footnote` and `Table` their lower-case names, and the others `other`. The label is kept as `native_type`.
- **Text.** The block's HTML is written as region **markup** (`{"format": "html", "content": …}`), not converted in the container. The host converts it with the shared HTML rules when it validates the prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md#markup-in-positioned-regions), [ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)). For example, paragraphs, list items and `<br>` become lines, a table becomes one line per row with tab-separated cells, `<math>` keeps its TeX, and tags such as `<b>`, `<i>` or `<sup>` add nothing. The standardized prediction records `conversion` `{"from": "html", "version": 1}`. Blocks without text (figures, empty blocks) are regions without text.
- **Page text.** The engine writes an empty page text; the host builds it from the converted regions, joined by blank lines.

## Truncation, failures, memory and time

- Generation stops at the end token, at `max_tokens` or at the end of the 18,000-token context. vLLM caps the request at the context left after the prompt: a large page has about 6,200 prompt tokens, so it can generate only about 11,800 tokens, below `max_tokens`. Without an end token the sample is `truncated` (`extra.stop_reason` `length`). The complete blocks and the unclosed last block are scored as produced ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)).
- An output with text but no layout blocks fails the sample (`engine-error`, `the output has no layout blocks`); `output.html` keeps it. An empty output is a genuine empty result only when the page is blank (`is_blank_region` on the whole image, as in Surya). On any other page it fails the sample (`the output is empty but the page is not blank`), because Surya would fall back to block mode there, which this adapter does not do.
- A failed request (for example an out-of-memory error in the server) fails only that sample. If the server exits, that sample and every later one fail with its log tail.
- Every sample fails when the GPU is unusable, the parameters are invalid, the model directory is missing or does not match `models.sha256`, or the server does not start (the message has the end of its log).

**Memory.** vLLM reserves `gpu_memory_utilization` of the GPU's memory up front: 0.6 × 24 GB = 14.4 GB on an RTX 4090. The bfloat16 weights take about 1.4 GB; most of the rest is key/value cache and CUDA graphs. The peak memory actually used per page is not measured.

**Time.** Measured on an RTX 4090 for dev-01 (run `dev01-surya2-0.22.1-v1`):

- Start-up took 187 s the first time (compilation and CUDA graph capture) and 58 s with the compile caches in the cache directory. It is not part of the per-sample engine time.
- Generation ran at 160–220 tokens/s with MTP. A 1600-pixel book page took 7–16 s and a newspaper page 67–76 s (to the context limit).

## Outputs

| File | Content |
| --- | --- |
| `output.html` | The generated text, unchanged: the native output |
| `prediction.json` | The positioned prediction with region markup (above): the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `prompt_tokens`, `generated_tokens`, `max_tokens`, `source_image`, `model_image` (after `scale_to_fit`), `blocks` (parsed), `blank_blocks` (dropped), `skipped_blocks` (labels without text), `repeat_loop`, the timings and `tokens_per_second`, the vLLM, Surya, transformers, openai, torch and CUDA versions, the GPU, the server command, the start-up time, the model (repository, revision, file checksums, the configured `model`, the prompt and its SHA-256, temperature and top-p) and the parameters |

## Results on dev-01

Measured in run `dev01-surya2-0.22.1-v1`; the scores are in the pull request for [#67](https://github.com/michal-hradis/ocr_benchmarking/issues/67).

- **Newspapers.** Both Czech newspaper pages (4194 × 6192 and 3509 × 5239) were reduced to 2044 × 3052 pixels and stopped at the context limit after 11,798 tokens: `truncated`, with 74 % and 52 % of the reference characters. Repetition was not the cause. Neither Surya's request retry check (`detect_repeat_token`) nor its full-page loop check (`_detect_repeat_loop`) fires on any dev-01 output (checked in the image), and every output parses, so Surya's own pipeline would not have regenerated any page either.
- **Glagolitic page.** The model does not read the script. It wrote Ethiopic-script text in `<b>` tags and ended normally.
- **Book pages.** The four 1600-pixel German and Latin prints ended normally, with 6–11 blocks each.
- **Repeatability.** A second run (`dev01-surya2-0.22.1-v1-repeat`, same image and GPU) produced byte-identical `output.html` for all seven pages.

## Tests

`tests/test_surya2.py` runs offline in `make test`:

- parameter validation, the server command against Surya's settings, and the pinned model;
- the conversion of blocks to positioned regions with markup, including blank and figure blocks and a `Complex-Block` skipped by its canonical label, and its host-side conversion against the expected plain text;
- the model checksums and the download script;
- the engine protocol through the runner, with a fake `vllm` executable that serves fixed outputs over the loopback interface and checks every request, and fake `surya`, `openai` and `torch` modules (`tests/fixtures/surya2/fake`). The outputs are hand-written layout blocks (`tests/fixtures/surya2/output.html`, not real model output), an empty output on a blank page (success) and on a non-blank page (failure), text without blocks, a newspaper page cut off at `max_tokens` in a repetition loop, a failed request and a server that exits. That run is then evaluated and reported, including the truncated page against a 2,000-line reference;
- start-up failures: invalid parameters, no GPU, missing or changed model files and a server that exits during start-up.

The fake `surya` modules are small stand-ins written for the test. `make surya2-smoke` (after `make surya2-model`) builds the image and runs the real Surya and vLLM on the GPU through `ocrbench run start`, `evaluate` and `report` on a synthetic full page, a paragraph crop and a blank page. It checks that every region has a polygon and that the markup was converted on the host.
