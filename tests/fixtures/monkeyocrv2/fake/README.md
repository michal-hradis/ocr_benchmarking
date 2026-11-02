Fakes for the offline protocol test in `tests/test_monkeyocrv2.py`. They run the adapter as a host
process without vLLM, upstream's source, PyTorch, weights or a GPU. The test points the adapter at
`source/` as upstream's source tree (`OCRBENCH_MONKEYOCRV2_SOURCE`).

- `source/parsing/serve.py` is a fake upstream `serve.py`: it checks the command line (the pinned
  target model directory, the loopback host, the served model name, the 16,384-token context, 128
  sequences and, with `--draft-model`, the pinned draft and 16 speculative tokens), serves `/health`,
  `/metrics` (vLLM's speculative decoding counters, with a draft) and an OpenAI-compatible
  `/v1/chat/completions` on the loopback interface, and checks every request (the served model
  name, the PNG image before the prompt, temperature 0, the token limit of the prompt, the layout
  image enlarged to about 1 MP, crops of at most 1 MP). It "generates" one token per character.
  A **layout** request is answered by the gray level near the top-left corner of the page:

  | Gray level | Layout output |
  | --- | --- |
  | 200 | `../layout.txt` (a hand-written layout with every block type and an invalid item, not real model output), stop |
  | 175 | a list whose only item has no valid box (no readable block), stop |
  | 160 | an item whose box cannot be mapped to pixels (`1e999`), stop |
  | 150 | 100 newspaper line blocks, cut off at `max_tokens` (stop reason `length`) |
  | 140 | three `Text` blocks; the recognition request of the second fails with HTTP 500 |
  | 120 | one whole-page `Text` block whose recognition repeats a word up to `max_tokens` (stop reason `length`) |
  | 100 | text without layout items, stop |
  | 87 | a blank output, stop |
  | 80 | an unfinished item and the stop reason `abort` |
  | 50 | an HTTP 500 error (a request failure) |
  | 20 | the server exits |
  | other (70 in the test) | `[]`, stop |

  A **recognition** request is answered by the gray level of the crop's pixel (2, 2): the test
  draws block number + 100 at the top-left corner of every block, and the server returns that
  block's fixed text (for a newspaper page, `Line <n> of a very long newspaper page.`). A crop
  without a known marker gets an empty output.
  `FAKE_VLLM_EXIT=1` makes it exit during start-up.
- `source/parsing/core_runner.py` holds small stand-ins for the upstream `core_runner` functions
  the adapter uses, written for the test, not copied from upstream: the prompts are upstream's
  strings, the tolerant parser, the formula and OTSL table formatting and the repetition check are
  simplified. The GPU smoke test runs the real ones.
- `torch.py` reports one fake GPU. The `*.dist-info` directories provide the package versions.
