# Chandra OCR 2 engine (page-image OCR to layout HTML, GPU)

[Chandra OCR 2](https://huggingface.co/datalab-to/chandra-ocr-2) by Datalab ([#53](https://github.com/michal-hradis/ocr_benchmarking/issues/53)) reads a whole page image. It generates the page as **HTML layout blocks** in reading order: each top-level `<div>` has a box (`data-bbox`), a label (`data-label`) and the block's content. Content is paragraphs, headings, lists, HTML tables and `<math>` LaTeX. There is no separate detection or line recognition. The engine runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-chandra-ocr2:2-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files and the license at the pinned revision |
| `prompt.txt` | The official `ocr_layout` instruction, byte for byte |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make chandra-ocr2-model`) |
| `chandra_ocr2_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`; its settings are the single preset |

## Versions and license

- **Model:** [`datalab-to/chandra-ocr-2`](https://huggingface.co/datalab-to/chandra-ocr-2) at commit `af93b47dba1b47b6640c86ccf487ed2260ab9a09` (2026-06-26). [Code](https://github.com/datalab-to/chandra) is Apache-2.0.
  - It is a Qwen3.5 model (`Qwen3_5ForConditionalGeneration`), the same architecture as [Infinity-Parser2-Flash](../infinityparser2/README.md) but larger. Its 32-layer language model mixes 24 linear-attention (gated delta net) layers with 8 full-attention layers. The vision encoder has 24 layers.
  - The weights are 10.6 GB of bfloat16 (`model.safetensors`, about 5.3 billion parameters).
  - The model card claims 90+ languages, handwriting, forms, tables and math. It names no restrictions on historical prints.
- **License of the weights: modified OpenRAIL-M** (`LICENSE` in the repository, downloaded with the weights and checked against `models.sha256`). It is not an open-source license. Read it before downloading. In short:
  - Use is free for research and personal use, and for organizations under US$2 million revenue and funding (Attachment A, 2(a) and 2(b)).
  - **Attachment A, 2(c) forbids any use**, research included, by anyone whose employer or affiliated entity offers a product or service that competes with Datalab's.
  - §8 applies the same license to the model's outputs. Outputs stay in local run directories and are never committed ([CONTRIBUTING.md](../../CONTRIBUTING.md#data-and-runtime-hygiene)).
  - §7 requires attribution and a link to the model when outputs are used.
  - The maintainer confirmed for [#53](https://github.com/michal-hradis/ocr_benchmarking/issues/53) that the benchmark's use is permitted. Anyone else running it must check the terms for themselves.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **5.19.0** (`AutoModelForImageTextToText`), with torchvision 0.29.1 for the image processor. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580).
  - The locked requirements are the same as Infinity-Parser2-Flash's. The image is separate, so each engine can change versions on its own.
  - The official pipeline is the [`chandra-ocr`](https://pypi.org/project/chandra-ocr/) package; version **0.2.0** was checked (wheel SHA-256 `11a9fffa…8a8ed4`). The package is not installed: it pulls in a vLLM/OpenAI client, PDF libraries, BeautifulSoup and markdownify. The adapter reproduces its transformers backend (`--method hf`) for image files: the prompt, the image resizing and the decoding (below).
  - vLLM, the package's recommended backend, is not used, because it would need its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)). Without vLLM there are also no retries on repetition (below).
  - The optional `flash-linear-attention` and `causal-conv1d` kernels are not installed. transformers then runs the linear-attention layers with its PyTorch implementation. The results are the same up to floating-point differences.
- **Weights are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)) in an explicit download step:

  ```sh
  make chandra-ocr2-model                        # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/chandra-ocr2/chandra-ocr-2/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again, and it records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-chandra-ocr2            # this image (no weights)
make chandra-ocr2-model             # the pinned weights, 10.6 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/chandra-ocr2/engine.toml \
    --dataset datasets/dev-01.json --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/chandra-ocr2 --inspect
make chandra-ocr2-smoke             # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` set explicitly, the model runs in float32 on the CPU. That is very slow and is not the configuration recorded here.

## Prompt, parameters and the preset

The adapter has **one preset**: the configuration in `engine.toml`, the official recommended one. That is the `ocr_layout` task, as used by the `chandra` command line, with its defaults.

- `prompt.txt` holds `OCR_LAYOUT_PROMPT` of `chandra-ocr` 0.2.0 byte for byte, and a test checks its SHA-256.
- The prompt asks for HTML layout blocks with boxes normalized to 0–1000 and one of 19 labels (`Caption`, `Footnote`, `Equation-Block`, `List-Group`, `Page-Header`, `Page-Footer`, `Image`, `Section-Header`, `Table`, `Text`, `Complex-Block`, `Code-Block`, `Form`, `Table-Of-Contents`, `Figure`, `Chemical-Block`, `Diagram`, `Bibliography`, `Blank-Page`).
- It allows a fixed set of tags and attributes. It asks for KaTeX math in `<math>`, tables with `colspan`/`rowspan`, paragraphs joined into `<p>`, and a description of every image.
- The other official task, `ocr` (HTML without layout blocks), is not used. There is no parameter for it.
- The chat template puts the image and then the instruction into one user turn, with no system message. The template always closes an empty thinking block (`<think>\n\n</think>`).
- The same prompt is used for full pages and for paragraph crops. The sample `language` and other metadata are not used.
- The conversation and the SHA-256 of the instruction are recorded with every sample (`extra.models.prompt`, `extra.models.prompt_sha256`).

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters that are left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `model_dir` | `models/chandra-ocr2/chandra-ocr-2/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `max_new_tokens` | 12384 | Generated tokens before the output counts as truncated (`MAX_OUTPUT_TOKENS` of `chandra-ocr`) |
| `min_image_dim` | 1536 | An image with a side shorter than this is scaled up first (`MIN_IMAGE_DIM` of `chandra-ocr`); 0 turns this off |

**Decoding.** Decoding is greedy (`do_sample=False`). That is the transformers backend's decoding: `generation_config.json` sets no sampling. The vLLM backend's temperature 0 with `top_p` 0.1 is also greedy. Generation stops at `<|endoftext|>` (`generation_config.json`) or `<|im_end|>` (the end of the chat turn), as in `chandra.model.hf`. No repetition penalty is applied.

**Custom checkpoints.** Any directory under the cache that transformers can load as a Qwen3.5 image-text model (for example a fine-tuned Chandra) can be given as `model_dir`. Only the files are checked against `models.sha256`, never the directory name. The default directory must match exactly, or every sample fails. For another directory, the result is recorded:

- If the directory holds exactly the files in `models.sha256`, all matching, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded. Extra files count too, because transformers may load them.

Either way, `model_dir` and the SHA-256 of every file in it are recorded. For a custom checkpoint, also set `model` in a copy of `engine.toml` to name what is used. That name is what reports show as the model.

**Image size.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, other modes converted to RGB). It then resizes it in three steps, as the `chandra` command line does for an image file:

1. **Upscale** (`chandra.input.load_image`). If either side is shorter than `min_image_dim`, the image is scaled up with Lanczos until its shorter side is `min_image_dim`. Paragraph crops and small pages are therefore enlarged.
2. **Fit** (`chandra.model.util.scale_to_fit`). The image is scaled to whole units of 28 pixels, between 1792 × 28 and 3072 × 2048 pixels (6.3 million), keeping the aspect ratio as well as possible, with Lanczos. An A4 page scanned at 300 dpi (2480 × 3508) becomes 2100 × 2968.
3. **Processor.** The Qwen2-VL image processor rounds the size again, to whole 32-pixel units (16-pixel patches, merged 2 × 2), with bicubic resampling, as in the official code. Each unit is one image token, so a page has at most about 6,200 image tokens.

The sizes after step 2 and step 3 are recorded as `scaled_image` and `model_image`.

## Context, truncation and memory

The model's context is 262,144 tokens for the prompt and the generated text together; in practice `max_new_tokens`, memory and time are the limits. Generation stops at the first end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- **Truncation.** Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the limit and `extra.stop_reason` set to `max_new_tokens` or `context` (`end-token` otherwise). The HTML is then cut off, often inside a block. The converter keeps everything generated up to the cut, including the text of the last, unfinished block; a start tag that was cut off adds nothing. The text is scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)).
- **Repetition.** The vLLM backend of `chandra-ocr` detects outputs that end in a repeated sequence and retries them up to six times with sampling (temperature 0.2–0.8, `top_p` 0.95). The transformers backend has no such retry, and neither has the adapter. A model that repeats until the limit is therefore reported as `truncated`. Results with retries could differ, and would not be deterministic. On the dev-01 pages, one of seven outputs looped until `max_new_tokens`: the Glagolitic print page. Its single block repeats one Myanmar-script sign, so the page is `truncated` with about 8,200 characters of wrong text against a 479-character reference.
- **Dense and multi-column pages.** A newspaper page is read in one pass, scaled to at most 6.3 million pixels, so small print loses resolution. 12,384 tokens may not hold a dense page's text and HTML. Both dev-01 newspaper pages finished with an end token, after 6,202 and 7,288 tokens. `max_new_tokens` can be raised, together with `timeouts.sample`. The column order is the model's; the flexible metrics score the text independently of that order.
- **Memory.** The bfloat16 weights take 10.6 GB of GPU memory. Only the 8 full-attention layers keep a key/value cache, about 32 KB per token (4 key/value heads of 256 dimensions), so about 0.6 GB for 6,200 image tokens and 12,384 generated ones. A single RTX 4090 (24 GB) is the target. The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`. On the RTX 4090 it was about 9.9–10.3 GB for book pages and 10.8 GB for newspaper pages at the full 6.3 million pixels (about 6,100 image tokens). A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues. Host memory is limited to 24 GB by `[resources]`.
- **Time.** Generation is token by token with batch size 1. On the RTX 4090 it runs at about 21 tokens/s with the PyTorch linear-attention implementation, so 12,384 tokens take about 580 s. Checking and loading the 10.6 GB of weights took about 20 s. `timeouts.sample = 1800` and `timeouts.startup = 600` leave a margin for slower disks or a raised `max_new_tokens`. `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample.

## Outputs

| File | Content |
| --- | --- |
| `output.html` | The generated text, decoded without special tokens and otherwise unchanged: the native output |
| `layout.json` | The size of the input image and the top-level blocks: `label`, `bbox` (four integers as written, or the attribute as written when it is not four integers), `bbox_px` (below), the inner HTML (`content`), the character offsets of the whole block in `output.html` (`span`) and whether it was closed (`complete`) |
| `page.html` | `output.html` without its top-level `Image` and `Figure` blocks, otherwise unchanged: the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `blocks`, `image_blocks`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `scaled_image`, `model_image`, the timings, `gpu_peak_memory_mb`, the transformers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (`verified_official_checkpoint`; `repository` and `revision` only when verified; `model_dir`, file checksums, the configured `model`, the prompt and its SHA-256) and the parameters |

**Blocks.** The adapter reads the top-level `<div>` elements of the output with Python's HTML parser. It counts only `<div>` tags, so another unclosed tag cannot hide a block. A block that is still open at the end of the output (cut off) runs to the end and is marked `complete: false`. Text outside any block stays in the output and is scored.

**Image and Figure blocks are not scored.** The prompt asks the model to describe every image inside its block, and to turn charts into data and diagrams into Mermaid. A description is not text on the page. The maintainer chose for [#53](https://github.com/michal-hradis/ocr_benchmarking/issues/53) to leave these blocks out of the scored text:

- `page.html` leaves out the whole top-level `Image` and `Figure` blocks. These are the labels that `chandra.output.parse_html(include_images=False)` drops too.
- Text that is printed inside an image, such as a caption within the picture, is dropped with it. A `Caption` block next to the image is kept.
- `Diagram`, `Chemical-Block` and the other labels are kept.
- The blocks stay in `output.html` and `layout.json`, and `extra.image_blocks` counts them.

**Boxes.** The model gives each box as `x0 y0 x1 y1` scaled to [0, 1000] of the image width and height. `bbox_px` is the same box in float pixel-edge coordinates of the original stored image ([PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)): x × width / 1000 and y × height / 1000, not clipped. The official `parse_layout` computes the same, rounded down and clipped to the image. A box that is not four integers gets no `bbox_px`; no box is invented.

**Plain text and the prediction.** The shared HTML converter turns `page.html` into the standardized prediction on the host ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), not in the container. HTML markup is therefore never part of the recognized text, and a conversion fix needs no new inference.

- The official Markdown output (`parse_markdown`) is not used. It drops `Page-Header` and `Page-Footer` blocks by default; the adapter keeps them, because they are part of the page's text.
- Headings become `heading` regions, paragraphs and loose text `text`, list items `list-item`, and HTML tables `table` regions with one row per line and cells separated by tabs. A `<br>` becomes a line break, and an `<img>` and its alt text add nothing.
- Block `<math display="block">` becomes a `formula` region with its LaTeX.
- **Inline `<math>` inside a `<p>` is split off.** The converter (version 1) closes an open `<p>` at every `<math>`, so the text after it becomes a separate region. The characters are all kept, but the line is split. This is a limitation of the shared converter, and a fix would change converter results for every engine ([#75](https://github.com/michal-hradis/ocr_benchmarking/issues/75)).
- The converted prediction is **text-only**: it has no positions and no physical lines. The block boxes are real model geometry, but they stay in `layout.json` and are not converted into prediction regions. That would need the shared converter to return positioned regions for HTML blocks, which is a possible follow-up. No text geometry is invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. An empty output, or one with only a `Blank-Page` block, is a genuine empty result: `success` with empty text. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make chandra-ocr2-model and run with --cache`);
- the model files do not match `models.sha256`.

## Tests

`tests/test_chandra_ocr2.py` runs offline in `make test`. It checks the following without the network:

- parameter validation and the prompt against the official SHA-256;
- the image sizes against `scale_to_fit` of `chandra-ocr` 0.2.0, including the upscaling of small images;
- block parsing (nested, unclosed and cut-off blocks, invalid boxes) and the box scaling;
- leaving out Image and Figure blocks;
- the model checksums and the download script;
- the engine protocol through the runner.

The engine runs with fake `torch` and `transformers` modules (`tests/fixtures/chandra-ocr2/fake`).

- The fake processor checks the following:
  - the conversation and chat-template arguments of `chandra.model.hf`;
  - left padding;
  - that the image arrives already scaled to 28-pixel units.
- The fake model checks greedy decoding with both end tokens. It returns these outputs:
  - a hand-written layout HTML page with a header, an image with a description, an HTML table and a footer (`tests/fixtures/chandra-ocr2/page.html`, not real model output);
  - an empty page;
  - a `Blank-Page` block ended by `<|endoftext|>`;
  - a sentence without blocks;
  - an out-of-memory error;
  - a newspaper page whose blocks never end.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

`make chandra-ocr2-smoke` (after `make chandra-ocr2-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page. It checks that blocks have pixel boxes. A second run with `max_new_tokens = 64` must mark the page `truncated` and score what was generated. A third run without `--cache` must fail explicitly.
