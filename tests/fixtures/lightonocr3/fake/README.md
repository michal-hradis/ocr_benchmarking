Fakes for the offline protocol test in `tests/test_lightonocr3.py`. They run the adapter as a host
process without vLLM, PyTorch, weights or a GPU.

- `bin/vllm` is a fake `vllm serve`: it checks the command line (the pinned model directory of one
  of the three checkpoints, the loopback host, that checkpoint's context, one image per prompt, no
  prefix caching and the thinking flag for the Qwen3.5 checkpoints only), serves `/health` and an
  OpenAI-compatible `/v1/chat/completions` on the loopback interface and checks every request (the
  served model name, the PNG image first and, in grounding mode, the text `grounding`, LightOn's
  client sampling with seed 0, thinking disabled, and an image no larger than the checkpoint's
  longest edge). It "generates" one token per character, up to `max_tokens` and at most, like vLLM,
  up to the end of the context (the image counts 1 token per 32 × 32 pixels, 28 × 28 for the 1B
  checkpoint, plus 10), chosen by the gray level near the top-left corner:

  | Gray level | Output |
  | --- | --- |
  | 200 | `../output.md` in grounding mode, `../page.md` in plain mode (hand-written, not real model output), and an end token |
  | 150 | text without layout markers, and an end token |
  | 100 | 100 newspaper paragraphs (with markers in grounding mode), then the last line repeated up to the limit (stop reason `length`) |
  | 80 | one block and the stop reason `abort` |
  | 50 | an HTTP 500 error (a request failure) |
  | 20 | the server exits |
  | other (70 in the test) | an empty output and an end token |

  `FAKE_VLLM_EXIT=1` makes it exit during start-up.
- `torch.py` reports one fake GPU. The `*.dist-info` directories provide the package versions.
