Fake `torch` and `transformers` modules for the offline protocol test in `tests/test_glm_ocr.py`.
They run the adapter as a host process without PyTorch, transformers, weights or a GPU. The fake
processor checks that the prompt is the image followed by `Text Recognition:`, with no extra chat
template arguments, resizes like the real Glm46V image processor (whole 28-pixel units, the image
counted twice, as two temporal frames, against the `shortest_edge` to `longest_edge` pixel range)
and puts one image token per 28 × 28 pixels into the prompt; the fake model checks that generation
is greedy, with a repetition penalty, and stops at both end tokens, and "generates" one token per
character, chosen by the width of the input image:

| Width | Output |
| --- | --- |
| 900 | `../page.md` and `<\|endoftext\|>` |
| 901 | `x` and `<\|user\|>` (the start of the next turn) |
| 300 | an empty fenced block, as the real model wrote for a blank page, and an end token |
| 666 | raises a fake CUDA out-of-memory error |
| 2000 | numbered newspaper lines that never end (`newspaper_line` in `transformers/__init__.py`) |
| other | `x` and an end token |

`torch.py` is a copy of the OvisOCR2 fake. `transformers-5.19.0+fake.dist-info` provides the
package version. The processor and the model load only from a directory that contains
`config.json`.
