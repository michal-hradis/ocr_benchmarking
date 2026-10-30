# OpenRouter engine (hosted vision-language models, page image to text)

[OpenRouter](https://openrouter.ai) ([#57](https://github.com/michal-hradis/ocr_benchmarking/issues/57), tracker [#38](https://github.com/michal-hradis/ocr_benchmarking/issues/38)) serves many hosted vision-language models through one OpenAI-style [chat completions API](https://openrouter.ai/docs/api-reference/chat-completion) (`POST https://openrouter.ai/api/v1/chat/completions`). This adapter is a generic transport: a preset names one OpenRouter model ID, one provider and a fixed prompt, and the adapter sends each page image with that prompt and records the answer.

**The container needs outbound network access and an API key**, so the runner starts it only with `--allow-network` ([ADR 0013](../../docs/adr/0013-opt-in-network-for-hosted-api-engines.md)). **Every page is a paid request**, and the page image is sent to OpenRouter and to the preset's provider. OpenRouter's [terms](https://openrouter.ai/terms) and [privacy policy](https://openrouter.ai/privacy) and the provider's terms apply. Run it only on images you are allowed to send.

| Preset | File | Model | Provider | Output |
| --- | --- | --- | --- | --- |
| `openrouter-gemini-3.8-flash-literal` | `engine.toml` | `google/gemini-3.8-flash` | `google-ai-studio` | The answer as plain text (`page.txt`), scored as given, without positions |

This first preset exercises the transport. Its prompt is **provisional**: the fixed literal and layout-aware prompt presets are [#58](https://github.com/michal-hradis/ocr_benchmarking/issues/58), and the Qwen, OpenAI and Claude models are [#59](https://github.com/michal-hradis/ocr_benchmarking/issues/59). Each further model or prompt is a further preset file of this engine.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-openrouter:1-v1` from `ocrbench/base-cpu:py3.12-v1`, with Pillow |
| `requirements.in`, `requirements.txt` | Pillow 12.3.0, hash-locked (`make docker-lock`), used only to read the image's format and size |
| `openrouter_engine.py` | The adapter: Python standard library HTTP client (`urllib`), no SDK |
| `engine.toml` | The Gemini 3.8 Flash preset for `ocrbench run start` |

## Model and provider

- **Model ID `google/gemini-3.8-flash`.** On 2026-10-30, OpenRouter's model list gave it as Google's newest Flash model, with the snapshot `google/gemini-3.8-flash-20260902` behind it. The responses name `google/gemini-3.8-flash`. It accepts text and images; the model is proprietary and not downloadable.
- **Pinned model.** The adapter refuses a job whose `model` is not one exact `vendor/model` ID: routers (`openrouter/auto`), aliases (`~…`, `…latest`) and variants (`:free`, `:batch`, `:nitro`, …) are refused. A sample fails when the response names another model. A dated snapshot of the pinned model (`<model>-YYYYMMDD`) is accepted and recorded.
- **Pinned provider, no fallbacks.** The request sets `provider: {only: [<provider>], allow_fallbacks: false, require_parameters: true, data_collection: "deny"}`. OpenRouter then sends the request only to that provider, returns its error instead of trying another one, refuses it when the provider does not support every sent setting, and uses no provider that may store data for training. The preset uses `google-ai-studio`: on 2026-10-30, it was the only Google route for this model that accepts `temperature`, which is why it was chosen over Vertex. Its standard tier cost $0.75 per million input tokens (images included) and $3.75 per million output tokens. All responses reported provider `Google AI Studio` and `service_tier` `default`.
- **Deterministic settings:** `temperature` 0, `seed` 0, `reasoning.effort` `minimal`, `max_tokens` 32,768. The model's maximum output is 65,536 tokens. Six dev-01 pages repeated exactly in two runs.
- **No resolution control.** OpenRouter exposes no image-resolution parameter for this model. Google downsamples the image itself: every dev-01 page, from 1600 × 2182 to 4194 × 6192 pixels, used about 1,100 prompt tokens including the prompt.
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

- **The image file is sent unchanged**, as a base64 data URL with its own media type. There is no resizing or re-encoding on our side ([ADR 0001](../../docs/adr/0001-image-only-samples-engine-owned-preprocessing.md)). The adapter sends JPEG, PNG and WebP files. Any other format, such as TIFF, fails the sample instead of being converted. JPEG pages up to 4194 × 6192 pixels (6.1 MB) were accepted.
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
| `answer_format` | `text` | `text`: the answer is `page.txt`, scored as given. `markdown`: the answer is `page.md`, converted on the host with the shared Markdown rules ([ADR 0011](../../docs/adr/0011-region-markup-converted-on-the-host.md)), for a prompt that asks for Markdown |
| `request_timeout` | 300 | Seconds per HTTP request |
| `max_retries` | 4 | Retries after a throttled (`429`), timed-out (`408`), server-error (any `5xx`) or network-failed request, and after a `200` response that carries an error object instead of an answer |
| `retry_base_seconds` | 2 | Backoff: the response's `Retry-After` when valid, else 2, 4, 8, 16 s; any delay is at most 120 s |
| `min_interval_seconds` | 1 | Minimum spacing between request starts |

The sample timeout (2,100 s) covers the longest retry sequence: five attempts that each wait their full 300 s timeout after the 1 s spacing, and four delays of 120 s, 1,985 s in all. The test checks this bound against the preset. `request_timeout` limits each blocking socket operation, as in `urllib`; a server that trickles a response slowly could exceed it, and the runner's sample timeout then stops the container. The preset sets `reasoning_effort = "minimal"` and `answer_format = "text"`, because its prompt asks for plain text: Markdown conversion would treat characters of the transcription, such as a leading `#` or `*`, as formatting.

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

On 2026-10-30, the preset was run three times on the seven pages of `datasets/dev-01.json`. The table is run `dev01-openrouter-gemini-3.8-flash-literal-v1-r3`, made with the final adapter, which scores the answer as plain text (default evaluation configuration, micro-averaged %). The earlier runs `…-v1` and `…-v1-r2` scored the same answers through Markdown conversion. All three runs gave byte-identical answers on the six successful pages. One run cost $0.063, as reported by OpenRouter.

| Preset | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `openrouter-gemini-3.8-flash-literal` | 31.55 | 40.39 | 3.17 | 6.41 | 95.13 | 91.51 | 190.3 s | 6 success, 1 failure |

- **The Glagolitic page** (`CkXII61-0017`) **fails** in every attempt of all three runs. The model starts with "The image is oriented upside-down. Transcribing the page in…", then Google AI Studio ends the generation with an internal error (`finish_reason` `error`, native `OTHER`, code 500, no charge). After four retries the sample fails and is scored as empty. The five attempts take 47 s of the engine time. Their bodies, with the partial answer, are kept as `attempt-<n>-response.json`.
- **Czech newspapers:** almost all text is right (fCER 1.7 % and 2.4 %), but the column reading order differs from the reference, so strict CER is 64.0 % and 15.4 %.
- **Plain text versus Markdown conversion:** with the same answers, the Markdown conversion of the earlier runs gave 4.76 % instead of 4.44 % CER on `dta-laube…` and changed its line count from 24 to 23.
- **Clean early-modern pages:** CER 1.1–4.4 %. The output keeps the long s (`ſ`) and line-end hyphens. On the Latin page `dta-346672…`, the model writes no space after commas (`felix,odioſa`), so WER is 25.6 % at 3.3 % CER.
- **Lines:** the answer keeps the page's lines (for example 418 hypothesis against 419 reference lines on `3e6309d0…`).
- Latency was 4–7 s per dta page and 53–68 s for the two large newspaper pages.

## Tests

`tests/test_openrouter.py` runs the adapter offline as a host process against a fake API on the loopback interface (`tests/fixtures/openrouter/fake_api.py`). The fake API's hand-written response (`response.json`) has the shape of a real Gemini 3.8 Flash response. The tests cover the request (only the prompt, the image and the pinned model, provider and settings), the refusal of routers, aliases and variants, the settings fingerprints, plain-text scoring of the answer and host conversion of a Markdown answer, an empty answer, a repetition loop, the token limit, a content filter, a dated snapshot and another model, a throttled request and two kinds of error in a `200` body that succeed on retry, persistent `503` and immediate `401` and `402` failures, two choices, a non-JSON body, a TIFF image that is not sent, and setup errors. They check that the bodies of failed attempts are kept, and that the API key appears in no output. A two-server test checks that a redirect to another server is refused and that server receives no request. The opt-in `make openrouter-smoke` (`pytest -m api --run-docker --run-api`) builds the image and runs the preset against the real API through `ocrbench run`, `evaluate` and `report`.
