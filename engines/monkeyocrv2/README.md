# MonkeyOCRv2 parsing engine (page image to layout blocks, vLLM, GPU)

[MonkeyOCRv2](https://github.com/Yuliang-Liu/MonkeyOCRv2) ([#95](https://github.com/michal-hradis/ocr_benchmarking/issues/95)) is a document vision encoder family from HUST with multilingual parsing, understanding and recognition models. This engine runs its **parsing** model, **MonkeyOCRv2-B-Parsing**: the 0.1B ViT-B MonkeyOCRv2 encoder with a Qwen3 0.6B language model (0.7B parameters). The issue's alternative, MonkeyOCRv2-S-Parsing (0.6B), is not configured: B fits an RTX 4090 easily. The understanding models (`-Und`), the vision encoders and the recognition, detection and formula code of the same repository are other tasks and are not used.

Upstream parses a page in **two stages**, and this engine runs that pipeline:

1. **Layout.** One request with the whole page image and the layout prompt. The model answers with a Python list of blocks in reading order, each a label and a box scaled to [0, 1000]:

   ```text
   [{'bbox': [44, 85, 321, 151], 'label': 'Title'}, {'bbox': [44, 276, 492, 438], 'label': 'Text'}, …]
   ```

2. **Recognition.** One request per block, with the block's crop and its label's prompt: plain text for text-like blocks, LaTeX for formulas, OTSL for tables. There are no text lines.

The engine runs **upstream's `parsing/serve.py` inside its container**, as a child process of the adapter, on the loopback interface only ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)). `serve.py` imports upstream's vLLM model plugin (`parsing/modeling`) and starts `vllm serve`. The image uses the vLLM 0.26 base unchanged ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-monkeyocrv2:2-v1` from `ocrbench/base-cu130-vllm026:vllm0.26.0-py3.12-v1`, with upstream's pinned source files, without weights and without extra packages |
| `source.sha256` | Checksums of the upstream source files at the pinned commit, downloaded at build |
| `models-B-Parsing.sha256`, `models-B-Parsing-DFlash.sha256` | Checksums of the model files at the pinned revisions |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make monkeyocrv2-model`) and, in the Dockerfile, the source files |
| `monkeyocrv2_engine.py` | The adapter |
| `engine.toml`, `presets/b-parsing-dflash-nodewarp.toml` | The two configurations (below) for `ocrbench run start` |

## Versions and license

- **Model:** [`zenosai/MonkeyOCRv2-B-Parsing`](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing) at commit `e8871a0057c98378219b369f924a25ba10f392bf` (2026-09-08). The weights are 1.76 GB of bfloat16 in `model.safetensors`, whose SHA-256 is the Hugging Face LFS object ID (`0267fdc9…b2b3e702`). `models-B-Parsing.sha256` pins the 15 files the server loads: the configuration, the weights, the tokenizer, the chat template, the image processor configuration and the checkpoint's Python files (`configuration_monkeyocrv2.py` registers the model's configuration and processor classes with Transformers and is loaded with `--trust-remote-code`; the two `modeling_*.py` files are pinned with them but vLLM uses upstream's plugin instead). The two dewarping models of the checkpoint (`preprocessor1.pth`, `preprocessor2.pth`, 293 MB of pickled PyTorch state) and the training arguments (`args.json`) are not downloaded: both presets skip dewarping (below).
- **DFlash draft:** [`zenosai/MonkeyOCRv2-B-Parsing-DFlash`](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing-DFlash) at commit `5e87f46698d698e971039168c01c34e044445a9b` (2026-09-08): a 5-layer Qwen3 draft (168 MB, `b8a3800c…bc9f32539`) for [DFlash](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/spec_decode/dflash.py) block speculative decoding, which proposes 16 tokens per step from hidden states of target layers 1, 7, 13, 19 and 25. Only the DFlash preset loads it.
- **Source:** [`Yuliang-Liu/MonkeyOCRv2`](https://github.com/Yuliang-Liu/MonkeyOCRv2) at commit `6cc0c0bb4864f2541fa6bc3a8f976cffaac8f359` (2026-09-15). `source.sha256` pins the 7 files the engine uses: `parsing/serve.py`, `parsing/core_runner.py` (the pipeline), the vLLM plugin `parsing/modeling/modeling_monkeyocrv2_vllm.py` with the DFlash draft adapter `modeling_monkeyocrv2_dflash_vllm.py`, `modeling/__init__.py`, `modeling/modeling_preprocessor.py` (the dewarping model code, imported by `core_runner` but not run) and `LICENSE`. The Dockerfile downloads them from GitHub and checks them.
- **License:** the code and all MonkeyOCRv2 weights (encoders, Parsing, Und and the DFlash draft) are [Apache-2.0](https://github.com/Yuliang-Liu/MonkeyOCRv2/blob/6cc0c0bb4864f2541fa6bc3a8f976cffaac8f359/README.md), as stated by the repository's README and `LICENSE` and the Hugging Face model cards.
- **Inference:** vLLM **0.26.0** with torch **2.11.0** (CUDA 13.0) and transformers **5.14.1** from the base. Upstream installs vLLM **0.25.1** (cu129 wheels), which it requires for DFlash; its `serve.py` accepts any vLLM ≥ 0.25 and loads the same plugin. vLLM 0.26.0 has DFlash, and upstream's plugin and draft adapter load and run on it (smoke and dev-01 runs below); 0.25.1 itself was not tried. The checkpoint's `config.json` names transformers 4.57.1; its configuration classes load under 5.14.1. NVIDIA driver ≥ 580.
- **Not used:** upstream's CPU path (`parsing/cpu`, Transformers in-process), its end-to-end mode (`--end2end`, one request for layout and content), its repeated-output retries (`--retry-repeat`, off by default), and the Gradio and FastAPI front ends.
- **Weights are not in the image.** They go to the `/cache` volume in an explicit download step:

  ```sh
  make monkeyocrv2-model                      # both checkpoints into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  make monkeyocrv2-model CHECKPOINT=B-Parsing # only the target model, enough for the normal preset
  ```

  This writes `models/monkeyocrv2/MonkeyOCRv2-<checkpoint>/<revision>/` under the cache directory and checks every file against `models-<checkpoint>.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks that each directory it uses holds exactly these files, all matching, and records the SHA-256 of every file with each sample. vLLM reads the models from these directories, offline (`HF_HUB_OFFLINE=1`).
- **Compiled kernels and remote code.** At start-up vLLM compiles the model with torch.compile, captures CUDA graphs and compiles Triton kernels, which it loads as shared objects. The runner's `/tmp` is mounted `noexec`, so these caches, and the copy of the checkpoint's Python files that Transformers imports (`HF_MODULES_CACHE`), are written to `runtime/monkeyocrv2/` in the cache directory. Later runs reuse them. They are not model files and are not checked; delete the directory to start clean.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-vllm026           # shared vLLM 0.26 base, once
make docker-monkeyocrv2             # this image (no weights)
make monkeyocrv2-model              # the pinned weights and draft, 1.9 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/monkeyocrv2/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench run start --engine engines/monkeyocrv2/presets/b-parsing-dflash-nodewarp.toml \
    --dataset datasets/dev-01.json --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/monkeyocrv2 --inspect
make monkeyocrv2-smoke              # opt-in: build the image, run both presets on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in the configuration), and the server sees only the GPU of `device` (`CUDA_VISIBLE_DEVICES`). If the requested GPU is not visible, every sample fails with `engine-error`; there is no CPU fallback.

## Configurations

Each configuration is an independent benchmark result. Both run the same model, pipeline and prompts on the same image sizes; they differ only in how the server decodes.

| Configuration | File | Decoding |
| --- | --- | --- |
| `monkeyocrv2-b-parsing-nodewarp` | `engine.toml` | normal autoregressive greedy decoding (`serve.py` without a draft) |
| `monkeyocrv2-b-parsing-dflash-nodewarp` | `presets/b-parsing-dflash-nodewarp.toml` | greedy decoding with DFlash speculative decoding (`serve.py --draft-model`, 16 proposed tokens per step) |

**`nodewarp`: no dewarping.** Upstream's default pipeline first flattens each page with a neural dewarping step (`modeling_preprocessor.py`): a U²-Net page mask at 512 × 512 and a SwinV2 network that predicts a sampling grid, applied to the full-resolution image with `grid_sample`. It is meant for photographed pages (curl, folds, perspective). Its output has the input's size, but its content is moved, so the layout boxes it produces are not positions in the input image. Both presets use upstream's `--skip-preprocess` instead, as decided in [#95](https://github.com/michal-hradis/ocr_benchmarking/issues/95) for the benchmark's flat scans: the page image itself goes to the layout model, and every box is a genuine position in the input image. Upstream says skipping "may lead to worse accuracy but faster speed"; the effect on these pages is not measured.

**Speculative decoding is not output-identical here.** With greedy decoding, DFlash should in principle give the target model's own output. In practice the two presets give different outputs on some blocks, as repeated runs of the normal preset do too (dev-01 results below).

## Pipeline and parameters

1. **Server.** The adapter checks the model files and starts upstream's `serve.py` with the pinned target model on `127.0.0.1` and a free port. It passes `serve.py`'s own defaults explicitly, so that the recorded command shows them: served model name `MonkeyOCRv2`, a 16,384-token context, at most 128 concurrent sequences and a 16,384-token scheduler budget per step, one GPU, and `gpu_memory_utilization`. The DFlash preset adds `--draft-model` with the pinned draft and `--dflash-num-speculative-tokens 16` (the draft's `block_size`). `serve.py` then runs `vllm serve … --trust-remote-code` with, for DFlash, `--speculative-config {"method": "dflash", "model": <draft>, "num_speculative_tokens": 16}`. vLLM picks the attention backends.

   - `VLLM_USE_FLASHINFER_SAMPLER=0`: vLLM 0.26 would compile FlashInfer's sampling kernels at start-up with `nvcc`, which the image (like the base) does not contain. vLLM's PyTorch sampler is used instead; with greedy decoding both pick the most probable token.

   The adapter waits until `/health` answers (at most `startup_seconds`). The full command, the environment it sets, the GPU and its compute capability are recorded per sample (`extra.environment`).
2. **Pipeline code.** The adapter imports upstream's `parsing/core_runner.py` at the pinned commit and uses its functions for every pipeline step: the prompts (`ALL_PROMPT`), image resizing (`load_image`), the tolerant layout parser and box mapping (`_parse_tolerant_items`, `_normalize_model_item`, `_map_bbox_to_image`), the block crops (`_build_page_tasks`), the block formatting (`_format_block_fields`), the repetition check (`_should_retry_repeat_output`) and the Markdown rendering (`result2md`). It sends the requests itself, with the same content as upstream's client, so that it can record each request's stop reason, token counts and time; upstream's client returns only the text. Upstream's client retries HTTP errors and time-outs up to 5 times; the adapter does not retry.
3. **Image.** The image is decoded as stored and converted to RGB. EXIF orientation is not applied (upstream applies it to JPEG files only), so the boxes refer to the stored pixels. There is no dewarping.
4. **Layout request.** Upstream's `get_layout`: the page is resized proportionally (Lanczos) to about 1 megapixel, **enlarged** when smaller (`min_pixels` 1,003,520) and reduced when larger (`max_pixels` 1,003,520, parse.py's default). The scan's resolution therefore does not matter for layout: a 4194 × 6192 newspaper page and a 1600-pixel book page both reach the model as about 1 MP. The checkpoint's image processor (Qwen2-VL, 14-pixel patches merged 2 × 2, `max_pixels` 11,289,600) then rounds the size to multiples of 28 without resizing further. Prompt:

   > `Please output the categories and coordinates of the document elements in reading order.`

   Decoding: temperature 0 (greedy; the checkpoint's sampling defaults then do not apply), at most **4,096** tokens, no other stop. The output is parsed by upstream's tolerant parser (a list literal, else the longest list or the complete `{…}` items it can read); items without a label or a box of four numbers are dropped. Every box is mapped from [0, 1000] to the input image as upstream maps it: scaled by the image size, corners sorted, rounded to whole pixels and clipped to the image, at least one pixel wide and high.
5. **Recognition requests.** For each block in layout order, upstream's crop of the input image at the mapped box, **reduced** to at most about 1 MP when larger (never enlarged), with its label's prompt:

   | Labels | Prompt |
   | --- | --- |
   | `Text`, `Title`, `Section-header`, `List-item`, `Caption`, `Page-header`, `Page-footer` | `Please output the text content from the image.` |
   | `Formula` | `Please write out the expression of the formula in the image using LaTeX format.` |
   | `Table` | `Please extract the table from the image and represent it in OTSL format.` |

   Other labels get no request and no text: `Picture`, and `Footnote`, whose prompt is commented out in upstream's `ALL_PROMPT`, so upstream does not read footnotes. Decoding: temperature 0, at most **5,000** tokens per block. The requests of a page are sent concurrently, at most 128 at a time (the server's `max_num_seqs`), as upstream's pipeline keeps many in flight. The sample `language` is not used.
6. **Block formatting** (upstream's `_format_block_fields`): `Title` becomes a `# ` and `Section-header` a `## ` Markdown heading, a formula becomes a `$$ … $$` block (upstream also collapses runs of `\quad` and moves an equation number after it), and an OTSL table becomes an HTML `<table>` (upstream's `otsl_to_html`). Other text is kept as the model wrote it, stripped.

`[params]` in the configuration are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>` |
| `dflash` | `false` | Serve with the DFlash draft (the DFlash preset sets `true`) |
| `gpu_memory_utilization` | **0.5**, `serve.py`'s default; at most 0.6 ([ADR 0012](../../docs/adr/0012-inference-servers-inside-engine-containers.md)) | Share of the GPU's memory vLLM reserves for the weights, CUDA graphs and key/value cache. vLLM refuses to start when less is free |
| `startup_seconds` | 900 | Time allowed for the server to become ready |

The pipeline settings (image sizes, token limits, prompts, decoding, no dewarping, no retries) and the server settings are fixed and recorded per sample (`extra.models.pipeline`, `extra.models.server`).

## Conversion to the prediction

The adapter writes a positioned `prediction.json`. Every layout block becomes one region, in layout order, which is the model's reading order; `reading_order` lists them.

- **Positions.** The block's mapped box (above) becomes a four-point polygon in input-image pixels. Every block the parser keeps has a box; no position is invented.
- **Types.** `Text` becomes `text`, `Title` and `Section-header` `heading`, `List-item` `list-item`, `Caption` `caption`, `Footnote` `footnote`, `Page-header` and `Page-footer` `page-header` and `page-footer`, `Table` `table`, `Formula` `formula` and `Picture` `figure`; other labels `other`. The label is kept as `native_type`.
- **Text.** The block's formatted content (Markdown; tables as HTML) is written as region **markup** (`{"format": "markdown", "content": …}`), not converted in the container. The host converts it with the shared Markdown rules when it validates the prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md#markup-in-positioned-regions), [ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)): heading markers and emphasis add nothing, a table becomes one line per row with tab-separated cells and `$$…$$` math keeps its TeX. The standardized prediction records `conversion` `{"from": "markdown", "version": 2}`. Picture and footnote blocks, and blocks with empty output, are regions without text.
- **Page header and footer** blocks are kept as regions with their text. Upstream's own Markdown (`page.md`) drops them by default (`--keep-header-footer` off); the prediction does not.
- **Page text.** The engine writes an empty page text; the host builds it from the converted regions, joined by blank lines.

## Truncation, repetition, failures, memory and time

- **Layout cut off.** A layout output that stops at its 4,096-token limit (stop reason `length`) makes the sample `truncated`. Its complete items are still parsed and recognized, as upstream does; the message gives the number of blocks parsed. Every block is scored as produced ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)).
- **Block cut off.** A block output that stops at its 5,000-token limit also makes the sample `truncated`, with the number of such blocks in the message (`extra.truncated_blocks`). Its text is kept as produced.
- **Repetition.** Upstream's repetition check (`_should_retry_repeat_output`: the output, or the output without its last 50 characters, ends with a unit repeated more often than a length-dependent limit) is recorded per block (`repeated` in `blocks.json`) and counted in `extra.repeated_blocks`. Upstream would resample such blocks only with `--retry-repeat`, which is off by default and not used; the output is not trimmed.
- **Empty and malformed layouts.** `layout.txt` is written as soon as the layout request returns, before the output is parsed. A blank layout output or an empty list (`[]`) is an empty page (`success`, with the message `the layout output is empty`). A layout output that is not empty, ends with the end token and yields no readable block (prose, or only items without a label or a box of four numbers) fails the sample (`engine-error`, `the layout output is not empty but has no readable blocks`), as does an error while parsing it or mapping a box to pixels (`cannot parse the layout output: …`). A layout from which some items are read keeps those items, as upstream's tolerant parser does.
- Any other stop reason of the layout request or of a block request fails the sample (`engine-error`). A failed request (an HTTP error, for example an out-of-memory error in the server, or no answer within 300 s, parse.py's request time-out) fails only that sample. A failed block request does not stop the page's other block requests: `blocks.json` is written with every successful reply and the failed block's `error`, and the sample fails with the number of failed requests and the first error (`extra.failed_blocks`). If the server exits, that sample and every later one fail with its log tail. The outputs written so far and the request diagnostics in `extra` are kept.
- Every sample fails when the GPU is unusable, the parameters are invalid, a model directory the preset uses is missing or does not match its checksums, upstream's pipeline cannot be imported, or the server does not start (the message has the end of its log).

**Memory.** vLLM reserves `gpu_memory_utilization` of the GPU's memory up front: 0.5 × 24 GB = 12 GB on an RTX 4090. The bfloat16 weights take about 1.8 GB and the draft 0.2 GB; the rest is CUDA graphs, activations and the key/value cache. The peak memory actually used per page is not measured.

**Time.** `extra.layout_seconds` and `extra.recognition_seconds` are the wall-clock times of the two stages; `extra.layout.seconds` and each block's `request.seconds` in `blocks.json` are the request times; the sample's `seconds` is the whole page including image encoding and parsing. The DFlash preset records vLLM's speculative decoding counters for the page (`extra.spec_decode`: drafts, draft tokens, accepted tokens and the acceptance rate), read from the server's `/metrics` before and after the page.

## Outputs

| File | Content |
| --- | --- |
| `layout.txt` | The layout output, unchanged: native output |
| `blocks.json` | Written after recognition, also when a block request fails: the image size and every parsed block: label, box in [0, 1000] as parsed, box in pixels, the block's raw recognition output unchanged (`output`, native output), its formatted content, the OTSL of a table, the repetition check and the request diagnostics (stop reason, token counts, the size of the image sent, time), or the `error` of a failed request |
| `monkeyocrv2.json` | Upstream's page record as its pipeline writes `{stem}.json` for an image: image name and size and the blocks with box, label, formatted content and OTSL |
| `page.md` | Upstream's Markdown rendering (`result2md`): the formatted blocks joined by blank lines, without page headers and footers |
| `prediction.json` | The positioned prediction with region markup (above): the primary output |
| `result.json` | Status, processing time and, in `extra`: the layout request's diagnostics (`layout`), `blocks`, `labels`, `recognized_blocks`, `truncated_blocks`, `repeated_blocks`, the recognition token counts, `layout_seconds`, `recognition_seconds`, `source_image`, `spec_decode` (DFlash), the vLLM, transformers, torch and CUDA versions, the GPU and its compute capability, the server command and environment, the start-up time, the models (repositories, revisions, file checksums, the configured `model`), the source commit, the pipeline and server settings and the parameters |

## Results on dev-01

Measured in runs `dev01-monkeyocrv2-b-parsing-nodewarp-v1` (plus a repeat, `…-v1-repeat`) and `dev01-monkeyocrv2-b-parsing-dflash-nodewarp-v1` on an RTX 4090; the scores are in the pull request for [#95](https://github.com/michal-hradis/ocr_benchmarking/issues/95).

- **Status.** Six pages `success` with both presets. The Glagolitic page is `truncated`: the model does not read the script, and 2 of its 4 blocks repeat until the 5,000-token limit (Thai-script syllables and `10%` repeated). That page alone gives 12,148 of the 24,378 character errors (11,767 insertions); without it the micro CER is 28.2 % and fCER 3.4 % with both presets.
- **Newspapers.** The two Czech newspaper pages (4194 × 6192 and 3509 × 5239) reach the layout model at about 1 MP. It returned 35 and 54 blocks (paragraphs, headings, column headers), and no layout output reached its 4,096-token limit. Blocks larger than 1 MP (whole column sections) are reduced before recognition. The text is good (fCER 4.4 % and 2.4 %), but the block order often does not follow the reference order (strict CER 56.4 % and 15.1 %).
- **Book pages.** The four 1600-pixel German and Latin prints have CER 0.6–11.4 %, the same strict and flexible. Running headers, page numbers and signatures come out as `Page-header` and `Page-footer` blocks. The model writes no long s (ſ) in Fraktur.
- **Time.** Start-up (server and model load, compilation, CUDA graph capture) took 77 s (normal) and 81 s (DFlash) with warm compile caches; it is not part of the per-sample engine time. Engine time for the seven pages was 42.9 s and 43.2 s for the two normal runs and **19.3 s with DFlash**: 2.2 times faster. Per page, DFlash took 0.8–5.8 s against 1.8–16.0 s; the largest gain was on the Glagolitic page (16.0 → 4.0 s). vLLM accepted 18–52 % of the drafted tokens per page.
- **Output equivalence.** The layout request is deterministic with normal decoding: both normal runs gave byte-identical layouts on all seven pages. The block requests, sent concurrently and batched by vLLM, are not: 122 of 139 block outputs were identical between the two normal runs. DFlash gave identical layouts on 2 of 7 pages (the others differ by a coordinate unit or, once, a label) and 116 of 139 identical block outputs; most differences are single characters, often diacritics (`ž`/`z`, `í`/`i`); a few blocks whose boxes moved differ by a word or a line. The scores of all three runs agree within 0.1 points.

## Tests

`tests/test_monkeyocrv2.py` runs offline in `make test`:

- the pinned checkpoints and source files, parameter validation, and the server command of both presets against `serve.py`'s defaults;
- the conversion of block records to positioned regions with markup (every type, a picture and a footnote without text, an empty block, an unknown label) and its host-side conversion;
- the model checksums, the download script and the source download;
- the engine protocol through the runner, with a fake upstream source tree (`tests/fixtures/monkeyocrv2/fake/source`): a fake `serve.py` that checks the command line and every request (model name, image before prompt, PNG data URL, greedy decoding, the token limits, the layout image enlarged to about 1 MP and crops at most 1 MP) and serves fixed outputs over the loopback interface, small stand-ins for the `core_runner` functions the adapter uses (written for the test; the GPU smoke test runs the real ones) and a fake `torch` module. The outputs are a hand-written layout with every block type (`tests/fixtures/monkeyocrv2/layout.txt`, not real model output), a newspaper layout cut off at max_tokens, a block repeating up to max_tokens, malformed layouts (prose, a list without a valid item, a box that cannot be mapped) that fail with `layout.txt` kept, an empty list and a blank output (empty pages), an aborted layout request, a failed layout request, one failed block request among three whose other replies are kept in `blocks.json`, and a server that exits; the DFlash preset with its speculative decoding counters. That run is then evaluated and reported, including the truncated page against a 2,000-line reference;
- start-up failures: invalid parameters, no GPU, missing, changed or extra model files, a missing draft for the DFlash preset (and none needed for the normal one) and a server that exits during start-up.

`make monkeyocrv2-smoke` (after `make monkeyocrv2-model`) builds the image and runs both presets with the real model on the GPU through `ocrbench run start`, `evaluate` and `report` on a synthetic full page, a paragraph crop and a blank page. It checks that every region of the text samples has a polygon, that the markup was converted on the host and, for DFlash, that drafted tokens were accepted.
