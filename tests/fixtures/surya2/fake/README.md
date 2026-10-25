Fakes for the offline protocol test in `tests/test_surya2.py`. They run the adapter as a host
process without vLLM, Surya, PyTorch, weights or a GPU.

- `bin/vllm` is a fake `vllm serve`: it checks the command line, serves `/health` and an
  OpenAI-compatible `/v1/chat/completions` on the loopback interface, checks every request (the
  model name, Surya's full-page prompt after the PNG image, greedy decoding) and "generates" one
  token per character, chosen by the gray level near the top-left corner of the scaled page image:

  | Gray level | Output |
  | --- | --- |
  | 200 | `../output.html` (hand-written layout blocks, not real model output) and an end token |
  | 150 | text without layout blocks, and an end token |
  | 100 | 100 newspaper line blocks, then the last one repeated up to `max_tokens` (stop reason `length`) |
  | 50 | an HTTP 500 error (a request failure) |
  | 20 | the server exits |
  | other (70 in the test) | an empty output and an end token; a uniform page of that gray is blank, a checkerboard is not |

  `FAKE_VLLM_EXIT=1` makes it exit during start-up.
- `openai` is a minimal client for that endpoint.
- `surya` holds small stand-ins for the Surya 0.22.1 functions the adapter uses (the full-page
  prompt, `scale_to_fit`, `_build_messages`, `parse_full_page_html`, `is_blank_region`, the label
  tables, `SKIP_CANON_LABELS`, `_crop_block` and `_detect_repeat_loop`), written for the test, not copied from Surya.
  The GPU smoke test runs the real ones.
- `torch.py` is a copy of the dots.mocr fake. The `*.dist-info` directories provide the package
  versions.
