# Mistral OCR 4.1 engine (hosted API, page image to Markdown or paragraph blocks)

[Mistral OCR 4.1](https://docs.mistral.ai/models/ocr-4-1) ([#103](https://github.com/michal-hradis/ocr_benchmarking/issues/103)) is Mistral's hosted document OCR service. It reads a page image and returns the page as Markdown. With `include_blocks` it also returns paragraph-level **blocks** in reading order, each with a box, a label (`title`, `text`, `table`, `header`, …), the block's Markdown and confidence scores. The model is not available for download. This adapter sends each page to Mistral's [OCR API](https://docs.mistral.ai/capabilities/document_ai/basic_ocr) (`POST https://api.mistral.ai/v1/ocr`).

**This is the benchmark's first hosted-API engine.** The container needs outbound network access and an API key, so the runner starts it only with `--allow-network` ([ADR 0013](../../docs/adr/0013-opt-in-network-for-hosted-api-engines.md)). **Every page is a paid request**, and the page image is sent to Mistral. Mistral's [terms](https://legal.mistral.ai/terms) and data policy apply. Run it only on images you are allowed to send.

| Preset | File | Request | Output |
| --- | --- | --- | --- |
| `mistral-ocr-4.1-blocks` | `engine.toml` | `include_blocks: true`, block confidence | Positioned regions, one per block, with the block's Markdown as region markup |
| `mistral-ocr-4.1-markdown` | `presets/markdown.toml` | No blocks, page confidence | The page Markdown (`page.md`), converted to text on the host, without positions |

The two presets are separate requests and separate benchmark results. The blocks preset's page text comes from its blocks, the Markdown preset's from the page Markdown.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-mistral-ocr:1-v1` from `ocrbench/base-cpu:py3.12-v1`, with Pillow |
| `requirements.in`, `requirements.txt` | Pillow 12.3.0, hash-locked (`make docker-lock`), used only to read the image's format and size |
| `mistral_ocr_engine.py` | The adapter: Python standard library HTTP client (`urllib`), no Mistral SDK |
| `engine.toml`, `presets/markdown.toml` | The two configurations for `ocrbench run start` |

## Model version

- **Model ID `mistral-ocr-4-1`.** On 2026-10-30, the API's model list (`GET /v1/models`) gave `mistral-ocr-4-1`, `mistral-ocr-4` and `mistral-ocr-latest` as aliases of one model, "Official Mistral OCR 4.1". Mistral publishes **no dated snapshot** of 4.1, unlike OCR 3 (`mistral-ocr-2512`). `mistral-ocr-4-1` is therefore the most specific identifier available. Mistral may update the model behind it without changing the name, so results are not guaranteed to be reproducible later.
- The adapter refuses a job whose configured `model` is not `mistral-ocr-4-1`. It also fails a sample when the response names another model. Every sample records the model the response reports, Mistral's request ID (`mistral-correlation-id`) and the rate-limit headers.
- **Price:** Mistral's list price on 2026-10-30 was $4 per 1,000 pages. Each sample records `usage_info` (`pages_processed`, `doc_size_bytes`) and an *estimated* cost: pages × `usd_per_1000_pages`. The bill is what counts.
- The runs repeat only approximately. Two runs over the same seven pages a few minutes apart gave identical output on clean pages but different text on the newspapers and the Glagolitic page ([below](#development-set-results)).

## Build, run and evaluate

```sh
make docker-cpu                      # shared CPU base, once
make docker-mistral-ocr              # this image
export MISTRAL_API_KEY=...           # a Mistral API key; never written to the run
uv run ocrbench run start --engine engines/mistral-ocr/engine.toml --dataset datasets/dev-01.json \
    --allow-network                  # or --engine engines/mistral-ocr/presets/markdown.toml
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/mistral-ocr --inspect
make mistral-ocr-smoke               # opt-in, paid: both presets on three synthetic images
```

Without `--allow-network`, or with `MISTRAL_API_KEY` unset or empty, `run start` refuses the engine before it creates a run directory. The key reaches the container by name only (`docker run --env MISTRAL_API_KEY`), so it never appears in `run.json`, `container.json`, logs or outputs. No `--cache` and no GPU are needed.

## Request

For every sample, in job order, the adapter sends one request:

```json
{"model": "mistral-ocr-4-1",
 "document": {"type": "image_url", "image_url": "data:image/jpeg;base64,..."},
 "include_image_base64": false, "include_blocks": true,
 "extract_header": false, "extract_footer": false,
 "confidence_scores_granularity": "block"}
```

- **The image file is sent unchanged**, as a base64 data URL with its own media type. There is no PDF conversion, resizing or re-encoding ([ADR 0001](../../docs/adr/0001-image-only-samples-engine-owned-preprocessing.md)). Mistral documents raster image input, and JPEG and PNG pages were accepted at their full size, up to the 4194 × 6192 pixel (5.9 MB) newspaper page of dev-01. The adapter sends JPEG, PNG, WebP and AVIF files. Any other format, such as TIFF, fails the sample instead of being converted. Any resizing happens on Mistral's side and is not visible. The reported `dimensions` equalled the input image size on every page tried.
- **Nothing else from the job is sent:** no sample ID, file name, language or other metadata, and no ground truth.
- **Fixed options.** `table_format` is not set, so tables stay inline in the Markdown and in the blocks. Headers and footers are not extracted (`extract_header`/`extract_footer` false), so running heads and page numbers stay in the transcription, as they are in the reference. These three options are not parameters. With `table_format` `markdown` or `html`, Mistral moves tables out of the Markdown into a separate `tables` list and leaves a link in their place. The extract options move headers and footers into separate fields in the same way. The adapter does not put that text back, so it would go unscored.
- `include_image_base64` is false: the response refers to pictures but does not contain them.

### Parameters (`[params]`)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mode` | `blocks` | `blocks` (request blocks, positioned output) or `markdown` (page Markdown only) |
| `confidence_scores_granularity` | `block` | `page`, `block` or `word`; recorded, never scored |
| `request_timeout` | 300 | Seconds per HTTP request |
| `max_retries` | 4 | Retries after a throttled (`429`), timed-out (`408`), server-error (any `5xx`) or network-failed request |
| `retry_base_seconds` | 2 | Backoff: the response's `Retry-After` when valid (delay-seconds or an HTTP date, header name in any case; at most 120 s), else 2, 4, 8, 16 s |
| `min_interval_seconds` | 1 | Minimum spacing between request starts |
| `usd_per_1000_pages` | 4.0 | Only for the recorded cost estimate |

The sample timeout (1,800 s) covers a request's full timeout on every attempt.

## Outputs and statuses

Each sample directory holds:

- **`response.json`**: the API's response body, unchanged.
- **`page.md`**: the page's `markdown` field, the primary output of the Markdown preset.
- **`prediction.json`** (blocks preset only, primary): one region per block, in the response order, which is the reading order. The region's `polygon` is the block's box scaled from the page `dimensions` the API reports to the input image's pixels, and its `type` comes from the block label: `title` becomes `heading`, `header` becomes `page-header`, `footer` becomes `page-footer`, `table`, `caption` and `text` keep their names, `equation` becomes `formula`, `image` becomes `figure`, `list`, `code`, `references` and `aside_text` become `text`, and anything else becomes `other`. `native_type` keeps the label. The block's Markdown is the region `markup`, which the host converts with the shared rules ([ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)). An `image` block's content only references the picture (`![img-0.jpeg](img-0.jpeg)`), so such a region has a box but no markup. A region's `confidence` is the block's `average_content_confidence_score`, and all of the block's scores are kept in its `extra`. Confidence is never used by metrics.
- **`result.json`**: in `extra`, the image's media type, size and SHA-256; every HTTP attempt with its status and seconds; the latency of the successful request; the kept response headers; the response model; usage and estimated cost; the page dimensions and confidence; the block count; and the request settings.

| Case | Status |
| --- | --- |
| Valid one-page response | `ok` |
| Empty page Markdown | `ok`, an empty transcription. It is a result, not an error, and is recorded as `empty` |
| The page Markdown or any block ends in a **repetition loop** | `truncated`, scored as produced. The API reports no stop reason. A tail of one unit repeated in a row (8 times for a unit of up to 63 characters, 3 times for one of up to 4,096) means the service stopped generating inside a loop. This is the rule of the [LightOnOCR-3 adapter](../lightonocr3/README.md). The message and `tail_repetition` name the looping part |
| HTTP error, or throttling or network failure that outlasts the retries | `failed`, with the HTTP status and the API's error message. Authentication (`401`) and other request errors (`4xx` except `408` and `429`) are not retried |
| Response not JSON, not one page, no Markdown, another model, blocks missing or malformed in the blocks preset | `failed`, with `response.json` kept when the body was JSON |
| `MISTRAL_API_KEY` unset in the container, unknown parameter, wrong job model | Every sample `failed`, without a request |

Engine time is the time of the HTTP request: upload, Mistral's queueing and processing, and download. It depends on the network and the service load, and **it is not comparable with the on-device time of local engines**.

## Development-set results

On 2026-10-30, both presets were run on the seven pages of `datasets/dev-01.json` (run IDs `dev01-mistral-ocr-4.1-{blocks,markdown}-v1`; default evaluation configuration, micro-averaged %). All 28 requests (two runs of each preset) succeeded on the first attempt. One run of seven pages costs an estimated $0.028.

| Preset | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `mistral-ocr-4.1-blocks` | 68.76 | 83.53 | 57.41 | 72.84 | 44.98 | 29.29 | 73.7 s | 5 success, 2 truncated |
| `mistral-ocr-4.1-markdown` | 73.44 | 93.71 | 63.36 | 83.12 | 39.31 | 24.20 | 79.1 s | 6 success, 1 truncated |

- **Clean early-modern pages** score 0.8–8.2 % CER in both presets, with identical output in two runs. The output keeps the long s (`ſ`). The dta pages' library stamps and shelf marks are transcribed too, as text the reference may not contain.
- **Czech newspapers.** On `3e6309d0…`, reading order is off but the text is partly right (CER 67–69 %, fCER 37–39 %). On `bd497a62…`, the output is fluent Czech with many invented words, and it ends in a repetition loop after about 28,000–30,000 characters in both presets. The blocks preset returned only 13–18 blocks for this multi-column page, one of them over 14,000 characters long.
- **Glagolitic page** (`CkXII61-0017`, reference transcribed in Latin script): the output is Church Slavonic-like Cyrillic with loops, CER 186–261 %.
- **Lines:** the blocks and the page Markdown join a paragraph's lines into one line, so the flexible metrics see far fewer hypothesis lines than reference lines (for example 7 lines against 44 on `dta-steinen…`).
- **Run-to-run variation:** an earlier run of the same version gave 71.21 % (blocks) and 78.99 % (Markdown) CER overall. The difference came from the two newspapers and the Glagolitic page.
- Latency was 1–3 s per normal page, and 23–41 s for the two large newspaper pages (3509 × 5239 and 4194 × 6192 px).

## Tests

`tests/test_mistral_ocr.py` runs the adapter offline as a host process against a fake API on the loopback interface (`tests/fixtures/mistral-ocr/fake_api.py`). The fake API's hand-written response (`response.json`) has the shape of the real one, with boxes in half-size page dimensions to test the scaling. The tests cover the request content (only the image and the pinned options), the block conversion, empty pages, a repetition loop, a throttled request that succeeds on retry, an unlisted `520` with a lower-case `retry-after` date that succeeds on retry, `Retry-After` parsing, persistent `503` and immediate `401` failures, another model, two pages, a non-JSON body, a TIFF image that is not sent, and setup errors. They also check that the API key appears in no output. The opt-in `make mistral-ocr-smoke` (`pytest -m api --run-docker --run-api`) builds the image and runs both presets against the real API through `ocrbench run`, `evaluate` and `report`.
