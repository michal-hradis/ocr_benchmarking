Fake `torch` and `transformers` modules for the offline protocol test in `tests/test_fireredocr.py`.
They run the adapter as a host process without PyTorch, transformers, weights or a GPU. The fake
processor checks that the prompt is the image followed by `engines/fireredocr/prompt.txt`, resizes
like the real Qwen2-VL image processor (whole 32-pixel units, scaled into the `shortest_edge` to
`longest_edge` pixel range) and puts one image token per 32 × 32 pixels into the prompt; the fake
model "generates" one token per character, chosen by the width of the input image:

| Width | Output |
| --- | --- |
| 900 | `../page.md` and an end token |
| 300 | only an end token (a blank page) |
| 666 | raises a fake CUDA out-of-memory error |
| 2000 | numbered newspaper lines that never end (`newspaper_line` in `transformers/__init__.py`) |
| other | `x` and an end token |

`torch.py` is a copy of the LightOnOCR fake. `transformers-5.19.0+fake.dist-info` provides the
package version. The processor and the model load only from a directory that contains
`config.json`.
