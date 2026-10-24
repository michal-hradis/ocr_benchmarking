Fake `torch`, `torchvision` and `transformers` modules for the offline protocol test in
`tests/test_deepseek_ocr2.py`. They run the adapter as a host process without PyTorch,
transformers, the model's remote code, weights or a GPU. The fake model checks the prompt (BOS,
the image tokens where the image mask is set, then the text after `<image>`), that the number of
image tokens matches the global view and the local tiles (1024 × 1024 and 768 × 768 pixels), and
the decoding settings; it "generates" one token per character, chosen by the gray level at the
center of the global view of the (uniform) input image, which resizing does not change:

| Gray level | Output |
| --- | --- |
| 255 | `../output.md` (Markdown with layout tags) and an end token |
| 250 | only an end token (a blank page) |
| 200 | raises a fake CUDA out-of-memory error |
| 150 | text without layout tags and an end token |
| 100 | numbered newspaper blocks that never end (`newspaper_line` in `transformers/__init__.py`) |
| other | `x` and an end token |

The tokenizer and the model load only from a directory that contains `config.json`, with
`trust_remote_code`. The `*.dist-info` directories provide the package versions.
