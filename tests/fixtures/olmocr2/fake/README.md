Fake `torch` and `transformers` modules for the offline protocol test in `tests/test_olmocr2.py`.
They run the adapter as a host process without PyTorch, transformers, weights or a GPU. The fake
processor checks that the prompt is `engines/olmocr2/prompt.txt` followed by the image, resizes
like the real Qwen2-VL image processor with the checkpoint's settings (whole 28-pixel units,
scaled into the 3,136 to 12,845,056 pixel range) and puts one image token per 28 × 28 pixels into
the prompt; the fake model checks the decoding settings and "generates" one token per character,
chosen by the gray level of the (uniform) input image, which resizing does not change:

| Gray level | Output |
| --- | --- |
| 255 | `../output.md` (front matter and Markdown) and an end token |
| 250 | only a front matter without text (a blank page) and an end token |
| 200 | raises a fake CUDA out-of-memory error |
| 150 | Markdown without a front matter and an end token |
| 100 | a front matter and numbered newspaper lines that never end (`newspaper_line` in `transformers/__init__.py`) |
| other | `x` and an end token |

`torch.py` is a copy of the LightOnOCR fake. `transformers-5.19.0+fake.dist-info` provides the
package version. The processor and the model load only from a directory that contains
`config.json`.
