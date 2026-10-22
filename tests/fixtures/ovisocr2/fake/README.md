Fake `torch` and `transformers` modules for the offline protocol test in `tests/test_ovisocr2.py`.
They run the adapter as a host process without PyTorch, transformers, weights or a GPU. The fake
processor checks that the prompt is the image followed by `engines/ovisocr2/prompt.txt`, with
thinking turned off, resizes like the real Qwen2-VL image processor (whole 32-pixel units, scaled
into the `shortest_edge` to `longest_edge` pixel range) and puts one image token per 32 × 32
pixels into the prompt; the fake model checks that generation is greedy and stops at both end
tokens, and "generates" one token per character, chosen by the width of the input image:

| Width | Output |
| --- | --- |
| 900 | `../page.md` and the turn's end token `<\|im_end\|>` |
| 901 | `x` and the checkpoint's end token `<\|endoftext\|>` |
| 300 | only an end token (a blank page) |
| 666 | raises a fake CUDA out-of-memory error |
| 2000 | numbered newspaper lines that never end (`newspaper_line` in `transformers/__init__.py`) |
| other | `x` and an end token |

`torch.py` is a copy of the FireRed-OCR fake. `transformers-5.19.0+fake.dist-info` provides the
package version. The processor and the model load only from a directory that contains
`config.json`.
