Fakes for the offline protocol test in `tests/test_nemotron_parse.py`. They run the adapter as a
host process without vLLM, the nemotron-parse package, PyTorch, weights or a GPU.

- `bin/vllm` is a fake `vllm serve`: it checks the command line, serves `/health` and an
  OpenAI-compatible `/v1/chat/completions` on the loopback interface and checks every request (the
  served model name, the prompt before the PNG image, greedy decoding, `skip_special_tokens`
  false). A request with repetition penalty 1.1 is a full page, 1.0 a crop. It "generates" one
  token per character, chosen for a full page by the gray level near the top-left corner:

  | Gray level | Output |
  | --- | --- |
  | 200 | `../output.txt` (hand-written blocks, not real model output) and an end token |
  | 120 | the same; a crop of this page (the last full page requested) stops at `max_tokens` (stop reason `length`) |
  | 150 | text without blocks, and an end token |
  | 100 | 100 newspaper line blocks, then the last one repeated up to `max_tokens` (stop reason `length`) |
  | 80 | one character repeated up to `max_tokens` inside the first block (stop reason `length`) |
  | 50 | an HTTP 500 error (a request failure) |
  | 20 | the server exits |
  | other (70 in the test) | an empty output and an end token |

  A crop gets a table block with the corrected cell `2026` (the full page has `2O26`).
  `FAKE_VLLM_EXIT=1` makes it exit during start-up.
- `nemotron_parse` holds small stand-ins for the nemotron-parse 3.5.0.post7 functions the adapter
  uses (`PostprocessOptions`, `postprocess_document`, and `PROMPT`, `GenerationSettings`,
  `GenerationError` and `DocumentParser` with a simplified double mode), written for the test,
  not copied from NVIDIA's package. They do not render tables or clean up text. The GPU smoke test
  runs the real ones.
- `torch.py` reports one fake GPU. The `*.dist-info` directories provide the package versions.
