# OpenRouter engine (hosted vision-language models, page image to text)

[OpenRouter](https://openrouter.ai) ([#57](https://github.com/michal-hradis/ocr_benchmarking/issues/57), tracker [#38](https://github.com/michal-hradis/ocr_benchmarking/issues/38)) serves many hosted vision-language models through one OpenAI-style [chat completions API](https://openrouter.ai/docs/api-reference/chat-completion) (`POST https://openrouter.ai/api/v1/chat/completions`). This adapter is a generic transport: a preset names one OpenRouter model ID, one provider and a fixed prompt, and the adapter sends each page image with that prompt and records the answer.

**The container needs outbound network access and an API key**, so the runner starts it only with `--allow-network` ([ADR 0013](../../docs/adr/0013-opt-in-network-for-hosted-api-engines.md)). **Every page is a paid request**, and the page image is sent to OpenRouter and to the preset's provider. OpenRouter's [terms](https://openrouter.ai/terms) and [privacy policy](https://openrouter.ai/privacy) and the provider's terms apply. Run it only on images you are allowed to send.

| Preset | File | Model | Provider | Output |
| --- | --- | --- | --- | --- |
| `openrouter-gemini-3.8-flash-literal` | `engine.toml` | `google/gemini-3.8-flash` | `google-ai-studio` | Literal prompt, default image resolution; the answer as plain text (`page.txt`), scored as given, without positions |
| `openrouter-gemini-3.8-flash-layout` | `presets/layout.toml` | `google/gemini-3.8-flash` | `google-ai-studio` | Layout-aware prompt, default image resolution; plain text as above |
| `openrouter-gemini-3.8-flash-literal-original` | `presets/literal-original.toml` | `google/gemini-3.8-flash` | `google-ai-studio` | Literal prompt, `image_detail = "original"` (about twice the image tokens); plain text as above |
| `openrouter-qwen3.8-flash-literal`, `-layout` | `presets/qwen3.8-flash-*.toml` | `qwen/qwen3.8-flash` | `alibaba` | Literal or layout-aware prompt, default image resolution; plain text as above |
| `openrouter-gpt-6.1-terra-literal`, `-layout` | `presets/gpt-6.1-terra-*.toml` | `openai/gpt-6.1-terra` | `openai` | As above |
| `openrouter-claude-sonnet-5.5-literal`, `-layout` | `presets/claude-sonnet-5.5-*.toml` | `anthropic/claude-sonnet-5.5` | `anthropic` | As above |
| `openrouter-gpt-6-luna-literal`, `-layout` | `presets/gpt-6-luna-*.toml` | `openai/gpt-6-luna` | `openai` | As above; low-cost |
| `openrouter-claude-haiku-5.5-literal`, `-layout` | `presets/claude-haiku-5.5-*.toml` | `anthropic/claude-haiku-5.5` | `anthropic` | As above; low-cost |

The three Gemini presets share every setting except the prompt or the image detail ([#58](https://github.com/michal-hradis/ocr_benchmarking/issues/58)). The model cohort of [#59](https://github.com/michal-hradis/ocr_benchmarking/issues/59) adds one current model each of the Qwen, OpenAI and Claude families and two low-cost models, each with the literal and the layout-aware prompt. Their presets have the Gemini presets' prompts and settings; only the model, the provider, and `temperature` and `seed` where the provider refuses them, differ (see *Model cohort* below).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-openrouter:2-v2` from `ocrbench/base-cpu:py3.12-v1`, with Pillow |
| `requirements.in`, `requirements.txt` | Pillow 12.3.0, hash-locked (`make docker-lock`), used only to read the image's format and size |
| `openrouter_engine.py` | The adapter: Python standard library HTTP client (`urllib`), no SDK |
| `engine.toml` | The literal Gemini 3.8 Flash preset for `ocrbench run start` |
| `presets/layout.toml`, `presets/literal-original.toml` | The layout-aware and the high-resolution literal Gemini presets |
| `presets/<model>-literal.toml`, `presets/<model>-layout.toml` | The literal and layout-aware presets of the other models |

## Gemini 3.8 Flash: model and provider

- **Model ID `google/gemini-3.8-flash`.** On 2026-10-30, OpenRouter's model list gave it as Google's newest Flash model, with the snapshot `google/gemini-3.8-flash-20260902` behind it. The responses name `google/gemini-3.8-flash`. It accepts text and images; the model is proprietary and not downloadable.
- **Pinned model.** The adapter refuses a job whose `model` is not one exact `vendor/model` ID: routers (`openrouter/auto`), aliases (`~…`, `…latest`) and variants (`:free`, `:batch`, `:nitro`, …) are refused. A sample fails when the response names another model. A dated snapshot of the pinned model (`<model>-YYYYMMDD`) is accepted and recorded.
- **Pinned provider, no fallbacks.** The request sets `provider: {only: [<provider>], allow_fallbacks: false, require_parameters: true, data_collection: "deny"}`. OpenRouter then sends the request only to that provider, returns its error instead of trying another one, refuses it when the provider does not support every sent setting, and uses no provider that may store data for training. The preset uses `google-ai-studio`: on 2026-10-30, it was the only Google route for this model that accepts `temperature`, which is why it was chosen over Vertex. Its standard tier cost $0.75 per million input tokens (images included) and $3.75 per million output tokens. All responses reported provider `Google AI Studio` and `service_tier` `default`.
- **Deterministic settings:** `temperature` 0, `seed` 0, `reasoning.effort` `minimal`, `max_tokens` 32,768. The model's maximum output is 65,536 tokens. Six dev-01 pages repeated exactly in two runs.
- **Image resolution.** Google downsamples the image itself. By default, every dev-01 page, from 1600 × 2182 to 4194 × 6192 pixels, used about 1,100 prompt tokens including the prompt. OpenRouter passes the image's `detail` (parameter `image_detail`) to the model. A probe on 2026-10-30 with the 4194 × 6192 newspaper page and a two-word prompt gave 267 prompt tokens with `low`, 1,087 with no `detail`, `auto` or `high`, and 2,173 with `original`. OpenRouter silently ignores an unknown value such as `ultra_high` (1,087 tokens), so the adapter accepts only `auto`, `low`, `high` and `original`.
- Results depend on OpenRouter and the provider at request time. The pinned model and provider, the recorded response model, provider, generation ID and timestamps are the provenance. Exact reproduction later is not guaranteed.

## Model cohort ([#59](https://github.com/michal-hradis/ocr_benchmarking/issues/59))

One current model each of the Qwen, OpenAI and Claude families, chosen with the maintainer from OpenRouter's model list on 2026-10-30, and two low-cost models. The Gemini Flash family is the Gemini 3.8 Flash presets above. Every model accepts images and is proprietary and hosted; none is downloadable.

| Model | Provider | Price per million tokens (input / output) | Sent settings | Image tokens of a page (dev-01) |
| --- | --- | --- | --- | --- |
| `qwen/qwen3.8-flash` (OpenRouter lists the snapshot `…-20260826`) | `alibaba` | $0.15 / $0.47 | `temperature` 0, `seed` 0 | about 2,600 |
| `openai/gpt-6.1-terra` | `openai` | $1 / $5 | `seed` 0 | 3,000 to 31,000 |
| `anthropic/claude-sonnet-5.5` | `anthropic` | $2 / $10 | none | 3,400 to 5,000 |
| `openai/gpt-6-luna` (low-cost) | `openai` | $0.10 / $0.50 | `seed` 0 | 3,000 to 31,000 |
| `anthropic/claude-haiku-5.5` (low-cost) | `anthropic` | $0.10 / $0.50 | none | 3,400 to 5,000 |

- **Same prompts and settings.** Each model has a literal and a layout-aware preset with the prompts of the Gemini presets, `reasoning_effort = "minimal"`, `max_tokens` 32,768, the default image resolution (no `image_detail`), the same timeouts and retries, and `answer_format = "text"`. A test checks that only the model, the provider, and `temperature` and `seed` differ.
- **Settings a provider refuses are not sent.** With `require_parameters`, OpenRouter refuses a request whose settings the provider does not support. On 2026-10-30, OpenAI accepted no `temperature` for the GPT-6 models, and Anthropic neither `temperature` nor `seed` for the Claude 5.5 models, so these presets leave them out. Their answers are therefore not guaranteed to repeat.
- **Reasoning.** `minimal` is the lowest effort every model accepts: Claude Sonnet 5.5 refuses `none` ("Reasoning is mandatory for this endpoint"). Qwen3.8 Flash still reasoned for up to about 13,800 tokens on a page with `minimal`; the GPT and Claude models used at most about 1,150.
- **Pinned provider.** Each model's own vendor, with no fallbacks and `data_collection: "deny"`. Every dev-01 response came from that provider and named the pinned model.
- **Image resolution.** Each provider scales the image itself. OpenAI's default kept far more of the large newspaper pages (21,754 and 30,835 prompt tokens) than Anthropic's (about 4,900) or Alibaba's (about 2,700).
- **Rate limits.** Alibaba's shared pool for Qwen3.8 Flash and Anthropic's for Claude Sonnet 5.5 answered `429` at times. The adapter's retries recovered all but one page, which was retried with `ocrbench run resume --retry-failed`.

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
| `temperature` | (omitted) | Sampling temperature, 0 to 2; not sent when left out. With `require_parameters`, a provider that does not accept it refuses the request |
| `seed` | (omitted) | Sampling seed; not sent when left out, as for `temperature` |
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
- **`response.json`**: the API's response body, unchanged, including any encrypted reasoning details.
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

Engine time is the time of the HTTP requests, including retries: upload, OpenRouter's and the provider's queueing and processing, and download. It depends on the network and the service load, and **it is not comparable with the on-device time of local engines**.

## Development-set results

### Gemini 3.8 Flash

On 2026-10-30, each Gemini preset was run once on the seven pages of `datasets/dev-01.json` (runs `dev01-<preset>-v2`, engine image `sha256:aeac5db4…`; default evaluation configuration, micro-averaged %). Engine time is the sum of the HTTP requests; cost is what OpenRouter reported.

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

### Model cohort

On 2026-10-30, each cohort preset was run once on the same seven pages (runs `dev01-<preset>-v2`, engine image `sha256:70eced5a…`, tag `2-v2`; default evaluation configuration, micro-averaged %). Engine time includes the waits of throttled requests.

| Preset | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time | Cost | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `openrouter-qwen3.8-flash-literal` | 41.42 | 54.73 | 26.82 | 37.28 | 79.97 | 66.41 | 903.8 s | $0.031 | 7 success |
| `openrouter-qwen3.8-flash-layout` | 35.69 | 51.90 | 16.77 | 29.94 | 77.43 | 64.04 | 707.1 s | $0.017 | 7 success |
| `openrouter-gpt-6.1-terra-literal` | 24.71 | 31.59 | 8.87 | 9.79 | 94.58 | 93.41 | 222.8 s | $0.167 | 7 success |
| `openrouter-gpt-6.1-terra-layout` | 25.94 | 31.52 | 16.70 | 18.38 | 89.62 | 88.31 | 207.4 s | $0.161 | 7 success |
| `openrouter-claude-sonnet-5.5-literal` | 81.59 | 83.57 | 81.10 | 81.61 | 31.23 | 28.99 | 64.2 s | $0.110 | 7 success |
| `openrouter-claude-sonnet-5.5-layout` | 82.00 | 84.09 | 81.65 | 82.46 | 30.08 | 27.43 | 74.3 s | $0.110 | 7 success |
| `openrouter-gpt-6-luna-literal` | 49.41 | 60.18 | 39.25 | 44.24 | 73.17 | 65.64 | 187.1 s | $0.018 | 7 success |
| `openrouter-gpt-6-luna-layout` | 43.87 | 50.62 | 25.12 | 32.80 | 79.97 | 73.93 | 197.1 s | $0.018 | 7 success |
| `openrouter-claude-haiku-5.5-literal` | 68.08 | 75.24 | 55.22 | 60.68 | 54.53 | 46.92 | 86.6 s | $0.009 | 7 success |
| `openrouter-claude-haiku-5.5-layout` | 38.59 | 49.63 | 25.64 | 34.78 | 75.94 | 67.00 | 109.6 s | $0.012 | 7 success |

Per page, CER (fCER) in %:

| Preset | `3e6309d0…` | `CkXII61-0017` | `bd497a62…` | `dta-346672…` | `dta-laube…` | `dta-martzi…` | `dta-steinen…` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen3.8-flash-literal` | 77.31 (40.89) | 85.59 (85.59) | 26.21 (22.63) | 2.79 (2.79) | 4.33 (4.33) | 2.03 (2.03) | 1.12 (1.12) |
| `qwen3.8-flash-layout` | 59.16 (8.56) | 106.26 (106.26) | 27.00 (24.82) | 2.70 (2.70) | 5.18 (5.18) | 2.99 (2.99) | 1.18 (1.18) |
| `gpt-6.1-terra-literal` | 65.74 (20.81) | 91.23 (91.23) | 0.52 (0.52) | 2.26 (2.26) | 4.44 (4.44) | 1.54 (1.54) | 1.18 (1.18) |
| `gpt-6.1-terra-layout` | 69.16 (42.95) | 93.53 (93.53) | 0.55 (0.55) | 1.92 (1.92) | 4.44 (4.44) | 3.96 (3.96) | 0.29 (0.29) |
| `claude-sonnet-5.5-literal` | 91.25 (89.89) | 76.62 (76.62) | 97.74 (97.71) | 1.66 (1.66) | 7.08 (7.08) | 1.74 (1.74) | 1.09 (1.09) |
| `claude-sonnet-5.5-layout` | 91.85 (90.88) | 82.05 (82.05) | 97.72 (97.69) | 1.92 (1.92) | 6.98 (6.98) | 7.92 (7.92) | 1.09 (1.09) |
| `gpt-6-luna-literal` | 66.27 (45.90) | 92.28 (92.28) | 49.65 (43.62) | 3.40 (3.40) | 7.40 (7.40) | 7.72 (7.72) | 1.22 (1.22) |
| `gpt-6-luna-layout` | 75.48 (56.83) | 92.28 (92.28) | 31.77 (7.13) | 5.57 (5.57) | 7.61 (7.61) | 6.85 (6.85) | 1.25 (1.25) |
| `claude-haiku-5.5-literal` | 58.93 (22.43) | 87.68 (87.68) | 92.72 (92.72) | 5.05 (5.05) | 6.98 (6.98) | 8.01 (8.01) | 1.25 (1.25) |
| `claude-haiku-5.5-layout` | 58.65 (21.90) | 77.04 (77.04) | 33.70 (33.70) | 5.23 (5.23) | 3.81 (3.81) | 5.60 (5.60) | 1.09 (1.09) |

- **The four early-modern pages** are read well by every model (0.3 to 8 % CER); the models differ on the newspapers.
- **GPT-6.1 Terra** reads the newspaper `bd497a62…` almost exactly (0.52 % CER in the reference column order), the best result of any OpenRouter preset on it, and has the lowest strict CER overall. On `3e6309d0…` it leaves out part of the page (fCER 20.8 % literal, 43.0 % layout).
- **Claude Sonnet 5.5 transcribes only the headlines of the two newspapers** (15 lines of `bd497a62…`, whose reference has 484), and so has the highest CER overall, although its early-modern pages are as good as the other models'. Anthropic scales the page to about 4,900 image tokens, probably too few for newspaper body text. Claude Haiku 5.5, at the same resolution, ended its literal answer for `bd497a62…` (40 lines) with a note, in Czech, that the resolution does not allow every word to be transcribed; with the layout prompt it wrote the body text in long lines (33.7 % CER). No higher-resolution or tiled variant was tried.
- **Qwen3.8 Flash** reads most of the newspapers, with many errors, and reasons for up to about 13,800 tokens on a page; with the waits of throttled requests it is the slowest; it is the cheapest of the mid-tier models.
- **The Glagolitic page:** the GPT and Claude models decline it in a sentence (Claude Sonnet's layout answer begins with a few Glagolitic letters), and Qwen answers in Armenian or Georgian letters. None gives the Latin transliteration of the reference.
- **The layout prompt** helps the smaller models most (Claude Haiku 5.5: CER 68.1 to 38.6 %; GPT-6 Luna: 49.4 to 43.9 %) and changes little for the others.

## Tests

`tests/test_openrouter.py` runs the adapter offline as a host process against a fake API on the loopback interface (`tests/fixtures/openrouter/fake_api.py`). The fake API's hand-written response (`response.json`) has the shape of a real Gemini 3.8 Flash response. The tests cover the request (only the prompt, the image and the pinned model, provider and settings), the refusal of routers, aliases and variants, the settings fingerprints, plain-text scoring of the answer and host conversion of a Markdown answer, an empty answer, a repetition loop, the token limit, a content filter, a dated snapshot and another model, a throttled request and two kinds of error in a `200` body that succeed on retry, persistent `503` and immediate `401` and `402` failures, two choices, a non-JSON body, a TIFF image that is not sent, and setup errors. They check that the bodies of failed attempts are kept, and that the API key appears in no output. The Gemini presets are checked to differ only in the prompt or the image detail, and each cohort preset to have the Gemini prompts and settings except the model, the provider, `temperature` and `seed`. `image_detail` is checked to be sent as `image_url.detail`, and a left-out `temperature`, `seed` or `reasoning_effort` not to be sent. A two-server test checks that a redirect to another server is refused and that server receives no request. The opt-in `make openrouter-smoke` (`pytest -m api --run-docker --run-api`) builds the image and runs every preset against the real API through `ocrbench run`, `evaluate` and `report`.
