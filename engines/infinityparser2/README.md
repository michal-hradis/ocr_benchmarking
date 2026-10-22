# Infinity-Parser2-Flash engine (page-image layout parser, GPU)

Fourth direct page-image model ([#17](https://github.com/michal-hradis/ocr_benchmarking/issues/17)), after [LightOnOCR-2](../lightonocr/README.md), [FireRed-OCR](../fireredocr/README.md) and [OvisOCR2](../ovisocr2/README.md). [Infinity-Parser2-Flash](https://huggingface.co/infly/Infinity-Parser2-Flash) is a compact 2B document parser. It reads a whole page image and generates its **layout elements** in reading order as one JSON array. Each element has a box, a category and its text: Markdown for text, HTML for tables and LaTeX for formulas. There is no separate detection or line recognition. It runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-infinityparser2:flash-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `requirements.txt` | transformers and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files at the pinned revision |
| `prompt.txt` | The official `doc2json` instruction, byte for byte |
| `fetch_model.py` | Downloads the pinned model files into a host cache directory (`make infinityparser2-model`) |
| `infinityparser2_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions

- **Model:** [`infly/Infinity-Parser2-Flash`](https://huggingface.co/infly/Infinity-Parser2-Flash) at commit `cfddc4106b0abc4706f05575a164f7dd35d09c46` (2026-07-20), Apache-2.0 ([technical report](https://arxiv.org/abs/2607.07836), [code](https://github.com/infly-ai/INF-MLLM/tree/main/Infinity-Parser2)).
  - It is a Qwen3.5 model (`Qwen3_5ForConditionalGeneration`), the same architecture as OvisOCR2 but larger. Its 24-layer language model mixes 18 linear-attention (gated delta net) layers with 6 full-attention layers. The vision encoder has 24 layers.
  - The weights are 4.4 GB of bfloat16 (`model.safetensors`).
  - The model card says it primarily supports English and Chinese, and that accuracy degrades on other languages.
  - The larger sibling [Infinity-Parser2-Pro](https://huggingface.co/infly/Infinity-Parser2-Pro) is not part of this adapter.
- **Inference:** [transformers](https://github.com/huggingface/transformers) **5.19.0** (`Qwen3_5ForConditionalGeneration`), with torchvision 0.29.1 for the image processor. It runs on the base's torch **2.14.1** for CUDA 13.0 (NVIDIA driver ≥ 580).
  - The locked requirements are the same as OvisOCR2's. The image is separate, so each engine can change versions on its own.
  - The official pipeline is the `infinity_parser2` package (version 0.4.0 was checked). Its default backend is vLLM, and it also has a transformers backend. The package itself is not installed: it pulls in PDF libraries, an HTTP client and qwen-vl-utils, and it downloads models on its own. The adapter reproduces its transformers backend for images: the prompt, the image resizing, the decoding and the parsing of the JSON (below).
  - vLLM is not used, because it would need its own CUDA stack ([ADR 0007](../../docs/adr/0007-runner-and-engine-protocol.md)).
  - The optional `flash-linear-attention` and `causal-conv1d` kernels are not installed. transformers then runs the linear-attention layers with its PyTorch implementation. The results are the same up to floating-point differences.
- **Weights are not in the image.** They go to the `/cache` volume ([image conventions](../../docs/DOCKER.md#engine-image-conventions)) in an explicit download step:

  ```sh
  make infinityparser2-model                     # into ~/.cache/ocrbench (OCRBENCH_CACHE=<dir> to change)
  ```

  This writes `models/infinityparser2/Infinity-Parser2-Flash/<revision>/` under the cache directory and checks every file against `models.sha256`. Runs need `--cache <dir>`. At start-up the adapter checks the files against `models.sha256` again, and it records the SHA-256 of every file in the model directory with each sample. Inference is offline (`HF_HUB_OFFLINE=1`): files are read from that directory only.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-infinityparser2         # this image (no weights)
make infinityparser2-model          # the pinned weights, 4.4 GB, into ~/.cache/ocrbench
uv run ocrbench run start --engine engines/infinityparser2/engine.toml \
    --dataset datasets/dev-01.json --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/infinityparser2 --inspect
make infinityparser2-smoke          # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. With `device = "cpu"` set explicitly, the model runs in float32 on the CPU. That is very slow and is not the configuration recorded here.

## Task, prompt and parameters

The official pipeline has two page tasks. **`doc2json` is used**: it is the pipeline's default task, and it is the only one that locates what it reads. `doc2md`, which asks for Markdown directly, is not used, and there is no parameter for it. The maintainer chose this in [#17](https://github.com/michal-hradis/ocr_benchmarking/issues/17).

- `prompt.txt` holds the `doc2json` instruction byte for byte, including its leading and trailing newline. It is `PROMPT_DOC2JSON` of `infinity_parser2` 0.4.0, the same text as in the model card, and a test checks its SHA-256.
- The instruction asks for every layout element with its box `[x1, y1, x2, y2]`, its category and its text, in human reading order. The categories are `header`, `title`, `text`, `figure`, `table`, `formula`, `figure_caption`, `table_caption`, `formula_caption`, `figure_footnote`, `table_footnote`, `page_footnote` and `footer`. Figures get empty text, formulas LaTeX, tables HTML and everything else Markdown. The text must be the original, without translation or rewriting.
- The chat template puts the image and then the instruction into one user turn, with no system message. Thinking is turned off (`enable_thinking=False`, as in the official code).
- The same prompt is used for full pages and for paragraph crops. The sample `language` and other metadata are not used.
- The conversation and the SHA-256 of the instruction are recorded with every sample (`extra.models.prompt`, `extra.models.prompt_sha256`).

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters that are left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `model_dir` | `models/infinityparser2/Infinity-Parser2-Flash/<revision>` | Model directory, relative to `/cache`; another directory holds a custom checkpoint (below) |
| `max_new_tokens` | 32768 | Generated tokens before the output counts as truncated, as in the official code |
| `max_pixels` | 16,777,216 | The image is scaled down to at most this many pixels (4096 × 4096, as in the official code); at least 2048 |

**Decoding.** Decoding is greedy, as in the official code (temperature 0); there are no sampling parameters. The checkpoint has no `generation_config.json`. The language model's end token in `config.json` is `<|endoftext|>`, while the chat turn ends with the tokenizer's end token `<|im_end|>`. Generation stops at either of them. No repetition penalty is applied.

**Custom checkpoints.** Any directory under the cache that transformers can load as a Qwen3.5 model (for example a fine-tuned Infinity-Parser2) can be given as `model_dir`. Only the files are checked against `models.sha256`, never the directory name. The default directory must match exactly, or every sample fails. For another directory, the result is recorded:

- If the directory holds exactly the files in `models.sha256`, all matching, the official checkpoint is recorded: `extra.models.verified_official_checkpoint` is `true`, with `repository` and `revision`.
- Otherwise `verified_official_checkpoint` is `false` and no official repository or revision is recorded. Extra files count too, because transformers may load them (a `generation_config.json` would change the decoding).

Either way, `model_dir` and the SHA-256 of every file in it are recorded. For a custom checkpoint, also set `model` in a copy of `engine.toml` to name what is used, for example `"my-org/Infinity-Parser2-finetuned@<revision>"`. That name is what reports show as the model.

**Image size.** The adapter decodes the image as stored with Pillow (EXIF orientation not applied, modes other than RGB converted to RGB). Like the official pipeline, it resizes the image itself and passes it to the processor with `do_resize=False`. The model works in units of 32 × 32 pixels, one image token each (16-pixel patches, merged 2 × 2):

- The size is computed as `smart_resize` of qwen-vl-utils 0.0.14 computes it. The aspect ratio is kept and both sides are rounded to whole units.
- An image with more than `max_pixels` pixels is scaled down. One with fewer than 2048 pixels is scaled up, which affects only tiny crops: a paragraph crop is only rounded to whole units.
- The resizing is Pillow's bicubic `resize`, as in qwen-vl-utils.
- An image whose aspect ratio is above 200 fails, as it does in qwen-vl-utils.
- The default limit is 16,384 image tokens. An A4 page scanned at 300 dpi (2480 × 3508) is not scaled down: it becomes 2496 × 3520, about 8,600 image tokens.
- The size the model saw is recorded as `model_image`.

## Context, truncation and memory

The model's context is 262,144 tokens for the prompt (mostly image tokens) and the generated text together, far more than a page needs. The official vLLM backend limits it to 65,536; the adapter, like the official transformers backend, does not. In practice `max_new_tokens`, memory and time are the limits. Generation stops at the first end token, at `max_new_tokens`, or at the end of the context, whichever comes first.

- **JSON overhead.** Every element repeats its box and category, about 20–30 tokens. Dense pages with many short elements therefore need more tokens than the same text as Markdown.
- **Truncation.** Without an end token the sample is `truncated` (runner status `truncated`), with `message` naming the limit and `extra.stop_reason` set to `max_new_tokens` or `context` (`end-token` otherwise).
  - The array is then cut off inside an element. As in the official code, only the complete elements are kept. The text of the cut-off element is dropped and counted in `extra.dropped_characters`.
  - The kept text is scored as produced, including omissions ([EVALUATION.md](../../docs/EVALUATION.md#failed-partial-and-truncated-samples)). A model that repeats an element until the limit is also reported this way.
- **Dense and multi-column pages.** A newspaper page is read in one pass in the model's own reading order, scaled down to at most `max_pixels`.
  - A page with 2,000+ lines can need more than 32,768 tokens. It is then truncated and scored against every reference line, so the missing text counts as deletions in the strict and flexible metrics.
  - `max_new_tokens` can be raised, together with `timeouts.sample`.
  - The column order is the model's. The flexible metrics score the text independently of that order.
  - The model card itself names dense historical newspapers as a case where elements are often missed.
  - Tiling the page inside the engine is a possible follow-up, not part of this adapter.
- **Memory.** The bfloat16 weights take 4.4 GB of GPU memory.
  - Only the 6 full-attention layers keep a key/value cache, about 12 KB per token (2 key/value heads of 256 dimensions). That is about 0.6 GB for 16,384 image tokens and 32,768 generated ones. The 18 linear-attention layers keep a state of fixed size.
  - The vision encoder runs on all patches of the image at once, up to 65,536 at the default `max_pixels`.
  - A single RTX 4090 (24 GB) is the target. The peak allocated GPU memory of every sample is recorded as `gpu_peak_memory_mb`. On the RTX 4090 it was about 4.7–5.0 GB for 1600-pixel book pages (2,500–4,400 image tokens) and about 6.9 GB for newspaper pages at the default `max_pixels` (about 16,250 image tokens).
  - A CUDA out-of-memory error fails only that sample (`engine-error`) and the container continues. Lowering `max_pixels` reduces the memory needed. Host memory is limited to 16 GB by `[resources]`.
- **Time.** Generation is token by token with batch size 1. A truncated sample generates all `max_new_tokens` and is the slowest.
  - On the RTX 4090, generation runs at about 27–28 tokens/s with the PyTorch linear-attention implementation, so 32,768 tokens take about 1,160 s; loading the model takes about 16 s. `timeouts.sample = 2400` seconds allows for that with a margin.
  - `preprocess_seconds`, `generate_seconds` and `tokens_per_second` are recorded per sample.
- **Repetition.** On the dev-01 pages, one of seven outputs ran into a loop until `max_new_tokens`: the non-Latin manuscript page, which repeats two words inside its first element. That element never closes, so no complete element is kept: the sample is `truncated` with an empty prediction (`dropped_characters` 27,346). Both newspaper pages finished with an end token (33 and 55 elements).

## Outputs

| File | Content |
| --- | --- |
| `output.txt` | The generated text, decoded without special tokens and otherwise unchanged: the native output |
| `layout.json` | The size of the input image and the complete elements of the JSON array as the model wrote them. Each element with a valid box also has `bbox_px` (below) |
| `page.md` | The text of every element in model order, joined by blank lines: the primary output |
| `result.json` | Status, processing time and, in `extra`: `stop_reason`, `elements`, `dropped_characters`, `prompt_tokens`, `image_tokens`, `generated_tokens`, the effective `max_new_tokens`, `model_image`, the timings, `gpu_peak_memory_mb`, the transformers/torch/CUDA versions, the GPU name and memory, the model loading time, the model (`verified_official_checkpoint`; `repository` and `revision` only when verified; `model_dir`, file checksums, the configured `model`, the prompt and its SHA-256) and the parameters |

**Parsing the JSON.** The adapter parses the output as the official `postprocess_doc2json_result` does:

- A ```` ```json ```` fence is removed, also an unclosed one.
- The elements of the array are read one by one, and reading stops at the first incomplete or malformed element. All complete elements before it are kept, and the rest is counted in `dropped_characters`.
- Two things fail the sample (`engine-error`), with `output.txt` kept: output that is not a JSON array at all, such as a sentence, and an element that is not an object with a string text.
- An empty output or `[]` is a genuine empty result: `success` with empty text.
- If the model ends normally but the array is malformed, the sample is `partial`, and `message` says how much was dropped. The kept elements are scored.

**Boxes.** The model gives each box as `[x1, y1, x2, y2]` scaled to [0, 1000] of the image width and height. `bbox_px` is the same box in float pixel-edge coordinates of the original stored image ([PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)): x × width / 1000 and y × height / 1000, not clipped. The official code computes the same, rounded down to integers. The model's own `bbox` is kept next to it.

**Plain text and the prediction.** `page.md` joins the text of all elements, as the official `convert_json_to_markdown` does, with one difference: the official function drops `header`, `footer` and `page_footnote` elements, and the adapter keeps them, because they are part of the page's text. Figures have empty text and add nothing. The shared Markdown converter turns `page.md` into the standardized prediction on the host ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)), not in the container. Markdown and HTML markup is therefore never part of the recognized text:

- Every element becomes one or more typed regions with their text only. A title written as `# …` becomes a `heading`.
- HTML tables become `table` regions with one row per line and cells separated by tabs.
- A formula element's text is LaTeX. Without `$$` delimiters, it is converted as an ordinary Markdown paragraph, as the official Markdown output would also treat it, so Markdown escapes in it, such as `\{`, are decoded.
- The converter's limitations apply ([PREDICTIONS.md](../../docs/PREDICTIONS.md#host-side-conversion-of-native-output)).

The converted prediction is **text-only**: it has no positions and no physical lines. The element boxes are real model geometry, but they stay in `layout.json` and are not converted into prediction regions. Doing that would need the shared converter to turn each element's Markdown into a positioned region, which is a possible follow-up. The element categories are kept in `layout.json` only. No text geometry is invented.

**Status.** An error during generation (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. Every sample fails in these cases:

- the GPU is unusable;
- the parameters are invalid;
- the model directory is missing (`... does not exist; download the weights with make infinityparser2-model and run with --cache`);
- the model files do not match `models.sha256`.

## Tests

`tests/test_infinityparser2.py` runs offline in `make test`. It checks parameter validation, the prompt against the official SHA-256, the image sizes against qwen-vl-utils' rules, JSON parsing (fences, truncation, malformed and non-JSON output), the box scaling, the model checksums, the download script (without the network) and the engine protocol through the runner.

- The engine runs with fake `torch` and `transformers` modules (`tests/fixtures/infinityparser2/fake`). The fake processor checks that the image arrives already resized and that thinking is off. The fake model checks greedy decoding with both end tokens.
- The fake model returns these outputs:
  - a hand-written fenced layout JSON page with a header, a figure, an HTML table and a footer (`tests/fixtures/infinityparser2/page.json`, not real model output);
  - `[]` ended by `<|endoftext|>`;
  - an empty page;
  - a sentence instead of JSON;
  - an array cut off before the end token;
  - an out-of-memory error;
  - a newspaper page whose elements never end.
- That run is then evaluated and reported, including the truncated page against a 2,000-line reference.

`make infinityparser2-smoke` (after `make infinityparser2-model`) builds the image and runs the real model on the GPU. It goes through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page, and it checks that elements have pixel boxes. A second run with `max_new_tokens = 64` must mark the page `truncated` and score the complete elements. It also runs without `--cache`, which must fail explicitly.
