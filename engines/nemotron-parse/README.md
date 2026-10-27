# Nemotron 3.5 Parse engine (page image to layout blocks, vLLM, GPU)

[Nemotron 3.5 Parse](https://huggingface.co/nvidia/Nemotron-3.5-Parse) by NVIDIA ([#93](https://github.com/michal-hradis/ocr_benchmarking/issues/93)) is a compact page parser: a C-RADIOv4-SO400M vision encoder and a 10-layer mBART decoder, about 0.63 billion parameters. It reads a whole page image and generates the page as a sequence of **layout blocks** in reading order. Each block is its text (Markdown, tables as LaTeX), framed by its box, normalized to the page, and followed by its class:

```text
<x_0.1211><y_0.0566>Text of the block …<x_0.8789><y_0.1133><class_Text>
```

There are no text lines.

The engine runs **vLLM inside its container** as a child process of the adapter, on the loopback interface only ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)), with NVIDIA's vLLM plugin for the model. The image uses the vLLM 0.26 base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-nemotron-parse:3.5-v1` from `ocrbench/base-cu130-vllm026:vllm0.26.0-py3.12-v1`, without weights |
| `requirements.in`, `requirements.txt` | The dependencies of NVIDIA's `nemotron-parse` package, hash-locked against the base (`make docker-lock`) |
| `source.sha256` | Checksums of the `nemotron-parse` package source at the pinned model revision, installed in the image |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make nemotron-parse-model`), or the package source while the image is built |
| `nemotron_parse_engine.py` | The adapter |
| `engine.toml`, `presets/double.toml` | The two configurations (below) for `ocrbench run start` |

## Versions and license

- **Model:** [`nvidia/Nemotron-3.5-Parse`](https://huggingface.co/nvidia/Nemotron-3.5-Parse) at commit `62cb1e4ba153d113ff64aaf2be6e068c377cf53c` (model release 3.5.0, 2026-10-17). The weights, including the multi-token-prediction (MTP) head, are 2.52 GB of float32 in one `model.safetensors` (630.8 million values; served in bfloat16), whose SHA-256 is the Hugging Face LFS object ID (`81dd7609…787e9`, also listed in the repository's `reference/model/runtime_compatibility.json`). `models.sha256` pins the 15 files the server loads: configuration, tokenizer, chat template, the processor and configuration code (`hf_*.py`, loaded with `--trust-remote-code`) and the license. Reference results, examples and documentation in the repository are not downloaded.
- **License:** [OpenMDW License 1.1](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/62cb1e4ba153d113ff64aaf2be6e068c377cf53c/LICENSE) for the model and the code; the tokenizer is MIT with attribution. Check them before using the weights.
- **Package:** `nemotron-parse` **3.5.0.post7**, the Python package in the same repository. It holds the vLLM plugin (`vllm_tied_patch`, registered as a `vllm.general_plugins` entry point), the parser that sends requests and merges crops (`nemotron_parse.inference.DocumentParser`) and the postprocessing that renders blocks (`postprocess_document`). It is not on PyPI. The image build downloads its 37 source files from the pinned revision, checks them against `source.sha256` and installs them without dependencies; no NVIDIA code is copied into this repository. Its exact pins that vLLM also depends on (numpy 2.2.6, safetensors 0.8.0, einops 0.8.1, openai 2.30.0, httpx 0.28.1, Pillow < 13) are in the base, the others (accelerate 1.12.0, timm 1.0.22, beautifulsoup4 4.13.3) in `requirements.in`. The package names setuptools 80.9.0 for building; the image builds it with the base's setuptools 80.10.2.
- **Inference:** vLLM **0.26.0**, NVIDIA's validated serving version (the repository's `docs/serving.md` and its Docker image `vllm/vllm-openai:v0.26.0`), with the torch **2.11.0** (CUDA 13.0) it pins, transformers **5.14.1** and huggingface-hub **1.24.0**. The plugin refuses other versions outside `vllm>=0.25,<0.33`. NVIDIA driver ≥ 580.
- **Not used:** the Transformers path of the model card (`AutoModel` with `trust_remote_code`). It runs without the plugin but without MTP and is not the reference serving path; the issue prefers the official vLLM plugin.
- **Weights are not in the image.** They go to the `/cache` volume in an explicit download step:

  ```sh
  make nemotron-parse-model                      # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/nemotron-parse/Nemotron-3.5-Parse/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks that the directory holds exactly these files, all matching, and records the SHA-256 of every file with each sample. vLLM reads the model from that directory, offline (`HF_HUB_OFFLINE=1`).
- **Compiled kernels.** At start-up vLLM compiles the model with torch.compile, captures CUDA graphs and compiles Triton kernels, which it loads as shared objects. The runner's `/tmp` is mounted `noexec`, so these caches are written to `runtime/nemotron-parse/` in the cache directory. Later runs reuse them. They are not model files and are not checked; delete the directory to start clean.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-vllm026           # shared vLLM 0.26 base, once
make docker-nemotron-parse          # this image (no weights)
make nemotron-parse-model           # the pinned weights, 2.5 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/nemotron-parse/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench run start --engine engines/nemotron-parse/presets/double.toml \
    --dataset datasets/dev-01.json --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id>... -o reports/nemotron-parse --inspect
make nemotron-parse-smoke           # opt-in: build the image, run both presets on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`), and the server sees only the GPU of `device` (`CUDA_VISIBLE_DEVICES`). If the requested GPU is not visible, every sample fails with `engine-error`; there is no CPU fallback.

## Configurations

| Configuration | File | Requests per page |
| --- | --- | --- |
| `nemotron-parse` (single pass) | `engine.toml` | One full-page request |
| `nemotron-parse-double` (double pass) | `presets/double.toml` | The full-page request, then one request per Table or Chart block it predicted, on a crop of that block; the crop's prediction refines the block |

They are independent benchmark results. NVIDIA's third mode (a separate detection request, then crops of the union of detected and parsed regions) is not configured. On pages without tables or charts the double pass makes no extra request and gives the same output as the single pass.

## Pipeline and parameters

1. **Server.** The adapter checks the model files and starts `vllm serve` on the model directory, on `127.0.0.1` and a free port, with the settings of the repository's `docs/serving.md`: served model name `nemotron-parse`, `--trust-remote-code`, bfloat16, a 16,384-token context, 16,384 batched tokens, no prefix caching, one image per prompt and the model's chat template (which concatenates the message contents). The plugin enables MTP with one speculative token by itself. Two H100 settings are changed:
   - `--max-num-seqs` is 16, not 192: pages are sent one at a time.
   - `--attention-config '{"flash_attn_version":3}'` is left out. FlashAttention 3 runs only on Hopper GPUs (compute capability 9.0); on an RTX 4090 (8.9) vLLM chooses its attention backend itself. `flash_attn_version` can request version 2 or 3 explicitly. The plugin's FlashAttention 3 scheduler patch has no effect on other backends.

   - `VLLM_USE_FLASHINFER_SAMPLER=0`: vLLM 0.26 would compile FlashInfer's top-k/top-p sampling kernels at start-up with `nvcc`, which the image (like the base) does not contain. vLLM's PyTorch sampler is used instead; with greedy decoding both pick the most probable token.

   The adapter waits until `/health` answers (at most `startup_seconds`). The full command, the environment it sets, the GPU and its compute capability are recorded per sample (`extra.environment`).
2. **Requests.** For each sample the image is decoded as stored (EXIF orientation is not applied; other modes are converted to RGB) and given to NVIDIA's `DocumentParser`. The adapter supplies the transport (`Backend`): the same chat request as the parser's own HTTP client, with the prompt and then the whole image as PNG, `skip_special_tokens` false, and the stage's sampling settings. The model's NaViT image processor shrinks a larger image to fit 1664 pixels wide and 2048 high, keeping its aspect ratio (bilinear; a smaller image is not enlarged), and pads it with white on the right and bottom to whole 64 × 16-pixel units. The prompt is the model card's, byte for byte (a test checks its SHA-256):

   > `</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>`

   - **Full page:** greedy decoding (temperature 0, top-k 1) with repetition penalty 1.1 and `max_tokens` 8,744: the model card's generation contract.
   - **Crops** (double pass): for each Table or Chart block of the full page, at most 80, the parser crops the original image to the block's box plus 4 % and 12 pixels, encodes the crop as JPEG (quality 95, NVIDIA's setting) and sends it with the same prompt, greedy, with repetition penalty 1.0 (the parser's crop default) and the same `max_tokens`.

   `DocumentParser`'s own defaults sample (temperature 0.2 for the full page and 0.1 for crops, top-k 50); the adapter makes every request greedy so that runs can be repeated. The sample `language` is not used.
3. **Merging** (double pass). Crop blocks are mapped back to the page. NVIDIA's default policy, `alternatives`, keeps both the full-page block and a differing crop block when they overlap, which would put the text of a table into the prediction twice. The adapter uses NVIDIA's `guarded` policy instead: a crop's table replaces the full-page table only when its structure passes NVIDIA's checks (a valid grid and compatible column counts), and an overlapping crop block of the same class is not added. Both policies use only the model's predictions; no ground truth is involved. Every crop response is kept in `nemotron.json`, also the ones not used.
4. **Postprocessing.** NVIDIA's `postprocess_document` with its default options renders the merged output: it removes exact duplicate blocks, repairs an empty first block, converts LaTeX tables to HTML (inferring header rows from spans and bold markup), renders charts as tables, removes soft hyphens between letters and normalizes list bullets in text blocks, and maps every box to pixels. It keeps the model's block order. The model's raw output stays unchanged in `output.txt`.

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>` |
| `mode` | `single` | `single` or `double` (the configurations above) |
| `max_tokens` | 8744 | Generated tokens per request before the output counts as truncated (the model card's limit); below the 16,384-token context, which also holds the image tokens |
| `gpu_memory_utilization` | **0.6**, also the maximum ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)) | Share of the GPU's memory vLLM reserves for the weights, CUDA graphs and key/value cache. vLLM refuses to start when less is free |
| `max_num_seqs` | 16 | vLLM's limit of concurrent sequences; requests are sent one at a time |
| `mtp` | `true` | MTP speculative decoding with one token, the plugin's default; `false` sets `NP_VLLM_MTP=0`. With greedy decoding, speculation does not change which tokens are chosen, up to floating-point differences |
| `flash_attn_version` | none (vLLM chooses) | `2` or `3` adds `--attention-config` with that FlashAttention version; 3 needs a Hopper GPU |
| `startup_seconds` | 900 | Time allowed for the server to become ready |

## Conversion to the prediction

The adapter writes a positioned `prediction.json`. Every rendered block becomes one region, in the model's order, which is its reading order; `reading_order` lists them.

- **Positions.** The model's coordinates are normalized to the page content: x is a fraction of the width and y of the height of the resized image without its white padding (NVIDIA's postprocessing needs no padding inverse), and the same fractions of the input image. NVIDIA's `transform_bbox_to_original` multiplies them by the size of the input image as stored, sorts the corners and clips them to the image; the box becomes a four-point polygon in float pixel coordinates. There are no line positions.
- **Types.** `Title` and `Section-header` become `heading`, `List-item` `list-item`, `Page-header` and `Page-footer` `page-header` and `page-footer`, `Picture` and `Chart` `figure`, `Bibliography` and `TOC` `text`, `Code` `other`, and `Text`, `Caption`, `Footnote`, `Table` and `Formula` their lower-case names. The class is kept as `native_type`.
- **Text.** The block's rendered content (Markdown; tables as HTML) is written as region **markup** (`{"format": "markdown", "content": …}`), not converted in the container. The host converts it with the shared Markdown rules when it validates the prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md#markup-in-positioned-regions), [ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)): emphasis markers add nothing, a list item loses its bullet, a table becomes one line per row with tab-separated cells, and `$…$` math keeps its TeX. The standardized prediction records `conversion` `{"from": "markdown", "version": 1}`. The model also transcribes text inside pictures (`<predict_text_in_pic>`), which stays in the `Picture` region. Blocks without text are regions without text.
- **Page text.** The engine writes an empty page text; the host builds it from the converted regions, joined by blank lines.

## Truncation, failures, memory and time

- A request stops at the end token or at `max_tokens`. Without an end token the sample is `truncated` (`extra.stop_reasons` has `length`), and the complete blocks are scored as produced ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)); the unfinished last block has no closing box and class and is left out. An output cut off inside its first block is therefore an empty `truncated` prediction; the real model does this on a small blank synthetic page, where it generates text in a repetition loop up to `max_tokens`. If a crop is cut off in the double pass, the sample is `truncated` and the full-page output is used without any crop refinement. `DocumentParser` itself raises an error in both cases; the adapter keeps the responses it already has.
- An empty full-page output with an end token is an empty page (`success`, with the message `the model returned an empty page`). An output with text but no blocks fails the sample (`engine-error`, `the output has no layout blocks`); `output.txt` keeps it.
- A failed request (for example an out-of-memory error in the server) fails only that sample. If the server exits, that sample and every later one fail with its log tail.
- Every sample fails when the GPU is unusable, the parameters are invalid, the model directory is missing or does not match `models.sha256`, or the server does not start (the message has the end of its log).

**Memory.** vLLM reserves `gpu_memory_utilization` of the GPU's memory up front: 0.6 × 24 GB = 14.4 GB on an RTX 4090. The bfloat16 weights take about 1.3 GB; most of the rest is key/value cache and CUDA graphs. The peak memory actually used per page is not measured.

**Time.** Measured on an RTX 4090 for dev-01 (run `dev01-nemotron-parse-3.5-v1`), with FlashAttention 2 and MTP:

- Start-up (server and model load, compilation, CUDA graph capture) took 52 s, with the compile caches from earlier smoke runs in the cache directory. It is not part of the per-sample engine time.
- A 1600-pixel book page took 1.4–2.8 s, and a newspaper page 15–17 s for 7,000–8,700 generated tokens (about 500 tokens/s). The images give 2,400–3,100 prompt tokens.

## Outputs

| File | Content |
| --- | --- |
| `output.txt` | The full-page model output, unchanged, with its coordinate and class tokens: the native output |
| `nemotron.json` | Every request's response (`stages`: stage, raw output, stop reason, token usage, image size, time) and NVIDIA's rendered document (`markdown`, `blocks` with class, normalized box, literal text, rendered content and pixel box, `warnings`, `raw_out` after merging and cleanup) |
| `page.md` | The rendered page as Markdown with HTML tables (NVIDIA's `markdown`) |
| `prediction.json` | The positioned prediction with region markup (above): the primary output |
| `result.json` | Status, processing time and, in `extra`: `mode`, `requests`, `crop_requests`, `stop_reasons`, `prompt_tokens` and `generated_tokens` (all requests), `fullpage_generated_tokens`, `max_tokens`, `source_image`, `blocks`, `postprocessing_warnings`, the vLLM, nemotron-parse, transformers, huggingface-hub, torch and CUDA versions, the GPU and its compute capability, the server command and environment, the start-up time, the model (repository, revision, file checksums, the configured `model`, the prompt and its SHA-256, the sampling of both stages, the merge policy and the postprocessing options) and the parameters |

## Results on dev-01

Measured in runs `dev01-nemotron-parse-3.5-v1` and `dev01-nemotron-parse-double-3.5-v1`; the scores are in the pull request for [#93](https://github.com/michal-hradis/ocr_benchmarking/issues/93).

- **Single and double pass are identical.** No page has a Table or Chart block, so the double pass makes no crop request, and every `output.txt` is byte-identical between the two runs.
- **Newspapers.** The two Czech newspaper pages (4194 × 6192 and 3509 × 5239) are reduced to fit 1664 × 2048. The model returns one block per paragraph, with the paragraph's text on one line, in column order. One page ended normally after 7,025 tokens; the other stopped at `max_tokens` (8,744) with 46 complete blocks, down to about 60 % of the page height: `truncated`. The context would allow about 13,500 generated tokens, but the configurations keep the model card's limit.
- **Glagolitic page.** The model does not read the script; it wrote Cyrillic-like text and ended normally.
- **Book pages.** The four 1600-pixel German and Latin prints ended normally, with 7–16 blocks each. The model writes no long s (ſ); the references of three of them have it 27–40 times each.

## Tests

`tests/test_nemotron_parse.py` runs offline in `make test`:

- parameter validation, the server command against the serving settings, the sampling and merge settings, and the pinned model and package source;
- the conversion of blocks to positioned regions with markup, including a picture without text and a box outside the image, and its host-side conversion against the expected plain text;
- the model checksums and the download script, for the weights and the package source;
- the engine protocol through the runner, with a fake `vllm` executable that serves fixed outputs over the loopback interface and checks every request, and fake `nemotron_parse` and `torch` modules (`tests/fixtures/nemotron-parse/fake`). The outputs are hand-written blocks (`tests/fixtures/nemotron-parse/output.txt`, not real model output), an empty output (success), text without blocks, a newspaper page cut off at `max_tokens` in a repetition loop, a failed request and a server that exits. That run is then evaluated and reported, including the truncated page against a 2,000-line reference. The double mode is run with a table crop that corrects a cell and with a crop cut off at `max_tokens`;
- start-up failures: invalid parameters, no GPU, missing or changed model files and a server that exits during start-up.

The fake `nemotron_parse` modules are small stand-ins written for the test; they do not render tables or clean up text. `make nemotron-parse-smoke` (after `make nemotron-parse-model`) builds the image and runs the real package and vLLM on the GPU, for both configurations, through `ocrbench run start`, `evaluate` and `report` on a synthetic full page, a paragraph crop and a blank page. It checks that every region has a polygon and that the markup was converted on the host.
