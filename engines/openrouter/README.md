# OpenRouter engine (hosted vision-language models, page image to text)

[OpenRouter](https://openrouter.ai) ([#57](https://github.com/michal-hradis/ocr_benchmarking/issues/57), tracker [#38](https://github.com/michal-hradis/ocr_benchmarking/issues/38)) serves many hosted vision-language models through one OpenAI-style [chat completions API](https://openrouter.ai/docs/api-reference/chat-completion) (`POST https://openrouter.ai/api/v1/chat/completions`). This adapter is a generic transport: a preset names one OpenRouter model ID, one provider and a fixed prompt, and the adapter sends each page image with that prompt and records the answer.

**The container needs outbound network access and an API key**, so the runner starts it only with `--allow-network` ([ADR 0013](../../docs/adr/0013-opt-in-network-for-hosted-api-engines.md)). **Every page is a paid request**, and the page image is sent to OpenRouter and to the preset's provider. OpenRouter's [terms](https://openrouter.ai/terms) and [privacy policy](https://openrouter.ai/privacy) and the provider's terms apply. Run it only on images you are allowed to send.

| Preset | File | Model | Provider | Output |
| --- | --- | --- | --- | --- |
| `openrouter-gemini-3.8-flash-literal` | `engine.toml` | `google/gemini-3.8-flash` | `google-ai-studio` | Literal prompt, default image resolution; the answer as plain text (`page.txt`), scored as given, without positions |
| `openrouter-gemini-3.8-flash-layout` | `presets/layout.toml` | `google/gemini-3.8-flash` | `google-ai-studio` | Layout-aware prompt, default image resolution; plain text as above |
| `openrouter-gemini-3.8-flash-literal-original` | `presets/literal-original.toml` | `google/gemini-3.8-flash` | `google-ai-studio` | Literal prompt, `image_detail = "original"` (about twice the image tokens); plain text as above |

The presets share every setting except the prompt or the image detail ([#58](https://github.com/michal-hradis/ocr_benchmarking/issues/58)). The Qwen, OpenAI and Claude models are [#59](https://github.com/michal-hradis/ocr_benchmarking/issues/59); each further model is a further preset file of this engine, with the same prompts.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-openrouter:2-v1` from `ocrbench/base-cpu:py3.12-v1`, with Pillow |
| `requirements.in`, `requirements.txt` | Pillow 12.3.0, hash-locked (`make docker-lock`), used only to read the image's format and size |
| `openrouter_engine.py` | The adapter: Python standard library HTTP client (`urllib`), no SDK |
| `engine.toml` | The literal Gemini 3.8 Flash preset for `ocrbench run start` |
| `presets/layout.toml`, `presets/literal-original.toml` | The layout-aware and the high-resolution literal presets |

## Model and provider

- **Model ID `google/gemini-3.8-flash`.** On 2026-10-30, OpenRouter's model list gave it as Google's newest Flash model, with the snapshot `google/gemini-3.8-flash-20260902` behind it. The responses name `google/gemini-3.8-flash`. It accepts text and images; the model is proprietary and not downloadable.
- **Pinned model.** The adapter refuses a job whose `model` is not one exact `vendor/model` ID: routers (`openrouter/auto`), aliases (`~…`, `…latest`) and variants (`:free`, `:batch`, `:nitro`, …) are refused. A sample fails when the response names another model. A dated snapshot of the pinned model (`<model>-YYYYMMDD`) is accepted and recorded.
- **Pinned provider, no fallbacks.** The request sets `provider: {only: [<provider>], allow_fallbacks: false, require_parameters: true, data_collection: "deny"}`. OpenRouter then sends the request only to that provider, returns its error instead of trying another one, refuses it when the provider does not support every sent setting, and uses no provider that may store data for training. The preset uses `google-ai-studio`: on 2026-10-30, it was the only Google route for this model that accepts `temperature`, which is why it was chosen over Vertex. Its standard tier cost $0.75 per million input tokens (images included) and $3.75 per million output tokens. All responses reported provider `Google AI Studio` and `service_tier` `default`.
- **Deterministic settings:** `temperature` 0, `seed` 0, `reasoning.effort` `minimal`, `max_tokens` 32,768. The model's maximum output is 65,536 tokens. Six dev-01 pages repeated exactly in two runs.
- **Image resolution.** Google downsamples the image itself. By default, every dev-01 page, from 1600 × 2182 to 4194 × 6192 pixels, used about 1,100 prompt tokens including the prompt. OpenRouter passes the image's `detail` (parameter `image_detail`) to the model. A probe on 2026-10-30 with the 4194 × 6192 newspaper page and a two-word prompt gave 267 prompt tokens with `low`, 1,087 with no `detail`, `auto` or `high`, and 2,173 with `original`. OpenRouter silently ignores an unknown value such as `ultra_high` (1,087 tokens), so the adapter accepts only `auto`, `low`, `high` and `original`.
- Results depend on OpenRouter and Google at request time. The pinned model and provider, the recorded response model, provider, generation ID and timestamps are the provenance. Exact reproduction later is not guaranteed.

## Build, run and evaluate

```sh
make docker-cpu                      # shared CPU base, once
make docker-openrouter               # this image
export OPENROUTER_API_KEY=...        # an OpenRouter API key; never written to the run
uv run ocrbench run start --engine engines/openrouter/engine.toml --dataset datasets/dev-01.json \
    --allow-network
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/openrouter --inspect
make openrouter-smoke                # opt-in, paid: the preset on three synthetic images
```

Without `--allow-network`, or with `OPENROUTER_API_KEY` unset or empty, `run start` refuses the engine before it creates a run directory. The key reaches the container by name only (`docker run --env OPENROUTER_API_KEY`), so it never appears in `run.json`, `container.json`, logs or outputs. No `--cache` and no GPU are needed.

## Request

For every sample, in job order, the adapter sends one request:

```json
{"model": "google/gemini-3.8-flash",
 "messages": [{"role": "user", "content": [
   {"type": "text", "text": "<the preset's prompt>"},
   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}],
 "provider": {"only": ["google-ai-studio"], "allow_fallbacks": false,
              "require_parameters": true, "data_collection": "deny"},
 "max_tokens": 32768, "temperature": 0.0, "stream": false,
 "seed": 0, "reasoning": {"effort": "minimal"}}
```

- **The image file is sent unchanged**, as a base64 data URL with its own media type, with `"detail"` added when the preset sets `image_detail`. There is no resizing or re-encoding on our side ([ADR 0001](../../docs/adr/0001-image-only-samples-engine-owned-preprocessing.md)). The adapter sends JPEG, PNG and WebP files. Any other format, such as TIFF, fails the sample instead of being converted. JPEG pages up to 4194 × 6192 pixels (6.1 MB) were accepted.
- **Nothing else from the job is sent:** no sample ID, file name, language or other metadata, and no ground truth. There is no system message.
- **The prompt is the same for every page** of a preset; there is no page-specific prompt.
- **Redirects are refused.** Python's default redirect handling would resend the `Authorization` header to the new location, which may be another host. The adapter follows no redirect, so the key goes only to the configured API URL. A `3xx` answer fails the sample and is not retried.

### Parameters (`[params]`)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `provider` | (required) | The one OpenRouter provider slug, e.g. `google-ai-studio` |
| `prompt` | (required) | The fixed instruction sent before the image |
| `max_tokens` | 32768 | Output token limit, including reasoning; an answer that reaches it is `truncated` |
| `temperature` | 0 | Sampling temperature |
| `seed` | 0 | Sampling seed; `null` omits it |
| `reasoning_effort` | (omitted) | `none`, `minimal`, `low`, `medium` or `high`; sent as `reasoning.effort` |
| `image_detail` | (omitted) | `auto`, `low`, `high` or `original`; sent as `image_url.detail`. The model decides the resolution each value gives (see *Image resolution* above) |
| `answer_format` | `text` | `text`: the answer is `page.txt`, scored as given. `markdown`: the answer is `page.md`, converted on the host with the shared Markdown rules ([ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)), for a prompt that asks for Markdown |
| `request_timeout` | 300 | Seconds per HTTP request |
| `max_retries` | 4 | Retries after a throttled (`429`), timed-out (`408`), server-error (any `5xx`) or network-failed request, and after a `200` response that carries an error object instead of an answer |
| `retry_base_seconds` | 2 | Backoff: the response's `Retry-After` when valid, else 2, 4, 8, 16 s; any delay is at most 120 s |
| `min_interval_seconds` | 1 | Minimum spacing between request starts |

The sample timeout (2,100 s) covers the longest retry sequence: five attempts that each wait their full 300 s timeout after the 1 s spacing, and four delays of 120 s, 1,985 s in all. The test checks this bound against the preset. `request_timeout` limits each blocking socket operation, as in `urllib`; a server that trickles a response slowly could exceed it, and the runner's sample timeout then stops the container. The presets set `reasoning_effort = "minimal"` and `answer_format = "text"`, because their prompts ask for plain text: Markdown conversion would treat characters of the transcription, such as a leading `#` or `*`, as formatting.

## Prompts

The two prompts are fixed for every page and model ([#58](https://github.com/michal-hradis/ocr_benchmarking/issues/58)); the exact text is in the preset files, and `result.json` records its SHA-256. They were written once and not tuned on dev-01 scores; only the instruction not to describe the image follows a dev-01 observation, the remark on page orientation with which the version 1 prompt began the Glagolitic page. Both share the transcription rules: keep the original spelling, punctuation, capitalization, diacritics, abbreviations, line-end hyphens and historical letter forms such as the long s; do not modernize, correct, expand, translate or summarize; no description of the image and no commentary; plain text only.

- **Literal** (`engine.toml`, `presets/literal-original.toml`): a diplomatic transcription in reading order, one output line for each line of the page.
- **Layout-aware** (`presets/layout.toml`): names the layout elements a page may have (columns, blocks, headings, captions, tables, marginal notes, headers, footers) and asks for blocks in natural reading order: each column top to bottom before the next, a heading that spans columns before the columns below it. Blocks are separated by an empty line, lines kept within a block, and no labels are added. An empty line costs nothing in scoring ([EVALUATION.md](../../docs/EVALUATION.md)).

`engine_version` 2 replaced the provisional prompt of version 1 ([#57](https://github.com/michal-hradis/ocr_benchmarking/issues/57)) with the literal prompt and added `image_detail`. No multi-request tiled variant was added.

## Outputs and statuses

Each sample directory holds:

- **`request.json`**: the request as sent, with the image data replaced by `data:<type>;base64,<sha256 …>`. It never contains the API key.
- **`response.json`**: the API's response body, unchanged, including Gemini's encrypted reasoning details.
- **`attempt-<n>-response.json`**: the response body of every failed attempt *n*, unchanged, also when a later retry succeeds: HTTP error bodies and `200` bodies that carry an error, including any partial answer. A body that is not JSON is kept as `attempt-<n>-response.txt`. Error bodies do not contain the request, so they hold no image and no key.
- **`page.txt`** (primary): the answer, `choices[0].message.content`, unchanged. The host scores it as plain text. With `answer_format = "markdown"`, the answer is `page.md` instead, and the host converts it with the shared Markdown rules.
- **`result.json`**: in `extra`, the image's media type, size and SHA-256; the request settings (without the image) with their SHA-256 and the prompt's SHA-256, the same for every sample of a run; every HTTP attempt with its start time (UTC), status, seconds and any error; the latency of the successful request; the generation ID and `created` time; the response model, provider, service tier and system fingerprint; the usage, including prompt, completion and reasoning tokens; the cost in USD that OpenRouter reports (`usage.cost`); the finish reason and the provider's native one.

| Case | Status |
| --- | --- |
| Answer with `finish_reason` `stop` | `ok` |
| Empty answer | `ok`, an empty transcription. It is a result, not an error, and is recorded as `empty` |
| `finish_reason` `length` | `truncated`: the answer reached `max_tokens`. Scored as produced |
| Answer ends in a **repetition loop** | `truncated`, scored as produced. A tail of one unit repeated in a row (8 times for a unit of up to 63 characters, 3 times for one of up to 4,096) is the rule of the [LightOnOCR-3 adapter](../lightonocr3/README.md) |
| HTTP error, an error object in a `200` response (top-level, or a choice with `finish_reason` `error`), throttling or network failure that outlasts the retries | `failed`, with the status and the error message. Authentication (`401`), missing credits (`402`) and other request errors (`4xx` except `408` and `429`) are not retried |
| Any other finish reason, such as `content_filter` | `failed`, with the finish reasons and any refusal |
| Response not JSON, not one choice, no text message, another model | `failed`, with the body kept as `response.json`, or as `attempt-<n>-response.txt` when it is not JSON |
| A redirect (`3xx`) | `failed`, not followed and not retried |
| `OPENROUTER_API_KEY` unset in the container, unknown or invalid parameter, a router, alias or variant as the job model | Every sample `failed`, without a request |

Engine time is the time of the HTTP requests, including retries: upload, OpenRouter's and Google's queueing and processing, and download. It depends on the network and the service load, and **it is not comparable with the on-device time of local engines**.

## Development-set results

On 2026-10-30, each preset was run once on the seven pages of `datasets/dev-01.json` (runs `dev01-<preset>-v2`, engine image `sha256:aeac5db4…`; default evaluation configuration, micro-averaged %). Engine time is the sum of the HTTP requests; cost is what OpenRouter reported.

| Preset | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time | Cost | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `openrouter-gemini-3.8-flash-literal` | 33.22 | 40.85 | 4.53 | 8.18 | 93.47 | 90.06 | 229.2 s | $0.106 | 7 success |
| `openrouter-gemini-3.8-flash-layout` | 31.45 | 40.95 | 3.28 | 7.14 | 94.14 | 90.58 | 287.1 s | $0.139 | 7 success |
| `openrouter-gemini-3.8-flash-literal-original` | 46.91 | 58.26 | 1.88 | 4.06 | 96.67 | 94.60 | 157.4 s | $0.074 | 7 success |

Per page, CER (fCER) in %, and engine time:

| Page | Lines (ref) | literal | layout | literal-original |
| --- | ---: | --- | --- | --- |
| `3e6309d0…` (Czech newspaper) | 419 | 55.22 (1.72), 53.3 s | 63.78 (1.86), 52.5 s | 54.72 (1.07), 49.3 s |
| `bd497a62…` (Czech newspaper) | 484 | 25.28 (5.38), 71.5 s | 15.46 (2.61), 67.1 s | 53.46 (0.59), 63.4 s |
| `CkXII61-0017` (Glagolitic) | 19 | 89.56 (89.56), 81.9 s | 94.99 (94.99), 145.2 s | 86.64 (86.64), 20.2 s |
| `dta-346672…` | 34 | 3.14 (3.14), 5.4 s | 3.05 (3.05), 5.6 s | 1.83 (1.83), 6.1 s |
| `dta-laube…` | 25 | 4.44 (4.44), 4.4 s | 4.44 (4.44), 4.5 s | 4.86 (4.86), 5.0 s |
| `dta-martzi…` | 29 | 1.74 (1.74), 5.2 s | 1.74 (1.74), 5.3 s | 1.74 (1.74), 6.0 s |
| `dta-steinen…` | 44 | 0.96 (0.96), 7.6 s | 1.09 (1.09), 7.0 s | 1.09 (1.09), 7.5 s |

- **Higher image resolution reads the text best:** with `image_detail = "original"`, fCER falls from 4.53 % to 1.88 % (0.59 % on `bd497a62…`, 1.07 % on `3e6309d0…`), and WER on the Latin page `dta-346672…` from 22.7 % to 9.3 %. Prompt tokens rise from about 1,180 to about 2,270 per page. Its strict CER is the worst of the three only because it read the columns of `bd497a62…` in another order (53.5 % CER at 0.6 % fCER).
- **The layout prompt** changes mainly the newspaper reading order: better on `bd497a62…` (15.5 % against 25.3 % CER), worse on `3e6309d0…` (63.8 % against 55.2 %). Neither prompt matches the reference column order of `3e6309d0…`. On the early-modern pages, the two prompts give nearly the same text.
- **The Glagolitic page** now gets an answer in every preset, written in Glagolitic letters, while its reference is a Latin transliteration (`I DUHA Sveta, i učitej ih dar-`). Bag-of-words F1 is therefore 0, and CER is 87–95 %. The literal prompt asks for the original letter forms, so this is a convention mismatch, not a prompt to tune per page. The answers are long (11,073 and 19,864 output tokens at default resolution, 1,235 at `original`) and make up most of the cost of the two default-resolution runs. With the version 1 prompt, the page failed with a provider error ([#57](https://github.com/michal-hradis/ocr_benchmarking/issues/57)).
- **Literal prompt, version 1 against version 2:** the version 1 run (`…-literal-v1-r3`, CER 31.55 %, fCER 3.17 %, one failure) differs mostly on `bd497a62…` (15.4 % against 25.3 % CER, 2.4 % against 5.4 % fCER), where the version 2 answer has 498 instead of 484 lines. The early-modern pages are within 0.2 points.
- Every request succeeded on the first attempt; all responses came from Google AI Studio.

## Tests

`tests/test_openrouter.py` runs the adapter offline as a host process against a fake API on the loopback interface (`tests/fixtures/openrouter/fake_api.py`). The fake API's hand-written response (`response.json`) has the shape of a real Gemini 3.8 Flash response. The tests cover the request (only the prompt, the image and the pinned model, provider and settings), the refusal of routers, aliases and variants, the settings fingerprints, plain-text scoring of the answer and host conversion of a Markdown answer, an empty answer, a repetition loop, the token limit, a content filter, a dated snapshot and another model, a throttled request and two kinds of error in a `200` body that succeed on retry, persistent `503` and immediate `401` and `402` failures, two choices, a non-JSON body, a TIFF image that is not sent, and setup errors. They check that the bodies of failed attempts are kept, and that the API key appears in no output. The presets are checked to differ only in the prompt or the image detail, and `image_detail` to be sent as `image_url.detail`. A two-server test checks that a redirect to another server is refused and that server receives no request. The opt-in `make openrouter-smoke` (`pytest -m api --run-docker --run-api`) builds the image and runs every preset against the real API through `ocrbench run`, `evaluate` and `report`.
