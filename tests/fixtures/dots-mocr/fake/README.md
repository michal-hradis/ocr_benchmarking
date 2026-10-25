Fake `torch` and `transformers` modules for the offline protocol test in `tests/test_dots_mocr.py`.
They run the adapter as a host process without PyTorch, transformers, weights or a GPU. The fake
processor checks the official request (one user turn, the image before the prompt), resizes like
the real Qwen2-VL image processor with the checkpoint's settings (whole 28-pixel units, scaled into
the 3,136 to 11,289,600 pixel range) and puts as many image tokens into the prompt as the real one
(one per 28 × 28 pixels). The fake model is loaded with the checkpoint's own code and SDPA
attention, imports `flash_attn_varlen_func` as the real model code does, checks the decoding
settings (greedy, the three end tokens) and "generates" one token per character, chosen by the
gray level of the (uniform) input image, which resizing does not change:

| Gray level | Output |
| --- | --- |
| 255 | `../output.json` (a hand-written layout list, not real model output) and an end token |
| 250 | `[]` and an end token (a blank page) |
| 200 | raises a fake CUDA out-of-memory error |
| 150 | text that is not JSON, and an end token |
| 100 | 100 numbered newspaper lines as layout elements, then one element repeated without end |
| 50 | one complete element and the start of a second, then an end token |
| other | `[]` and an end token |

`torch.py` is a copy of the HunyuanOCR-1.5 fake. `transformers-4.57.6+fake.dist-info` provides
the package version. The processor and the model load only from a directory that contains
`config.json`.
