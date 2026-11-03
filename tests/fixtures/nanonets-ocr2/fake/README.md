Fakes for the offline protocol test in `tests/test_nanonets_ocr2.py`. They run the adapter as a
host process without vLLM, PyTorch, weights or a GPU.

- `bin/vllm` is a fake `vllm serve`: it checks the command line (the pinned model directory, the
  loopback host, the 32,768-token context, one image per prompt, bfloat16, no remote code), serves
  `/health` and an OpenAI-compatible `/v1/chat/completions` on the loopback interface and checks
  every request (the served model name, one user message with the PNG image first and then the
  official prompt of `engines/nanonets-ocr2/prompt.txt`, greedy decoding, the repetition penalty
  1.05, seed 0 and nothing else). It "generates" one token per character, up to `max_tokens` and
  at most, like vLLM, up to the end of the context (the image counts 1 token per 28 × 28 pixels, at
  most 16,384, plus 150 for the prompt), chosen by the gray level near the top-left corner:

  | Gray level | Output |
  | --- | --- |
  | 200 | `../page.md` (hand-written in the model's output style, not real model output) and an end token |
  | 100 | 100 newspaper paragraphs, then the last line repeated up to the limit (stop reason `length`) |
  | 80 | a few characters and the stop reason `abort` |
  | 50 | an HTTP 500 error (a request failure) |
  | 20 | the server exits |
  | other (70 in the test) | an empty output and an end token |

  `FAKE_VLLM_EXIT=1` makes it exit during start-up.
- `torch.py` reports one fake GPU. The `*.dist-info` directories provide the package versions.
