Fakes for the offline protocol test in `tests/test_unlimited_ocr.py`. They run the adapter as a
host process without vLLM, PyTorch, weights or a GPU.

- `bin/vllm` is a fake `vllm serve`: it checks the command line (the pinned model directory, the
  loopback host, the 32,768-token context and the n-gram logits processor), serves `/health` and an
  OpenAI-compatible `/v1/chat/completions` on the loopback interface and checks every request (the
  served model name, the prompt before the PNG image, greedy decoding, the n-gram arguments,
  `skip_special_tokens` false). It "generates" one token per character, up to `max_tokens` and at
  most, like vLLM 0.26, up to the end of the context (the image counts 1 token per 16 × 16
  pixels), chosen by the gray level near the top-left
  corner:

  | Gray level | Output |
  | --- | --- |
  | 200 | `../output.txt` (hand-written blocks, not real model output) and an end token |
  | 150 | text without layout tags, and an end token |
  | 100 | 100 newspaper line blocks, then the last line repeated up to the limit (stop reason `length`) |
  | 80 | one block and the stop reason `abort` |
  | 50 | an HTTP 500 error (a request failure) |
  | 20 | the server exits |
  | other (70 in the test) | an empty output and an end token |

  `FAKE_VLLM_EXIT=1` makes it exit during start-up.
- `torch.py` reports one fake GPU. The `*.dist-info` directories provide the package versions.
