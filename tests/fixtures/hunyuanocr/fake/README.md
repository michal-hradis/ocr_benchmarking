Fake `torch` and `transformers` modules for the offline protocol test in `tests/test_hunyuanocr.py`.
They run the adapter as a host process without PyTorch, transformers, weights or a GPU. The fake
processor checks the official request (an empty system message, then the image before the task
prompt), resizes like the real HunYuanVL image processor with the checkpoint's settings (whole
32-pixel units, scaled into the 262,144 to 16,777,216 pixel range) and puts as many image tokens
into the prompt as the real one (rows × (columns + 1) + 2 for 32 × 32-pixel units). The fake model
checks the decoding settings (greedy, with a repetition penalty) and "generates" one token per
character, one at a time through the stopping criteria, chosen by the gray level of the (uniform)
input image, which resizing does not change:

| Gray level | Output |
| --- | --- |
| 255 | `../output.md` (Markdown with an HTML table) and an end token |
| 250 | nothing but an end token (a blank page) |
| 200 | raises a fake CUDA out-of-memory error |
| 150 | 100 numbered newspaper lines, then one line repeated without end |
| 100 | numbered newspaper lines that never end (`newspaper_line` in `transformers/__init__.py`) |
| other | `x` and an end token |

`torch.py` is a copy of the olmOCR-2 fake. `transformers-5.19.0+fake.dist-info` provides the
package version. The processor and the model load only from a directory that contains
`config.json`.
