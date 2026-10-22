Fake `torch` and `transformers` modules for the offline protocol test in
`tests/test_infinityparser2.py`. They run the adapter as a host process without PyTorch,
transformers, weights or a GPU. The fake processor checks that the prompt is the image followed by
`engines/infinityparser2/prompt.txt`, with thinking turned off, that the adapter has already
resized the image to whole 32-pixel units (`do_resize=False`, as in the official pipeline) and puts
one image token per 32 × 32 pixels into the prompt; the fake model checks that generation is greedy
and stops at both end tokens, and "generates" one token per character, chosen by the width of the
resized image:

| Width | Output |
| --- | --- |
| 896 | `../page.json` (a fenced layout JSON array) and the turn's end token `<\|im_end\|>` |
| 960 | `[]` and the language model's end token `<\|endoftext\|>` |
| 288 | only an end token (a blank page) |
| 672 | raises a fake CUDA out-of-memory error |
| 1216 | a sentence instead of JSON |
| 1280 | one complete element and a cut-off one, then an end token (`MALFORMED`) |
| 1344 | one complete element in an array without its closing `]`, then an end token (`UNCLOSED`) |
| 1984 | numbered newspaper elements that never end (`newspaper_element` in `transformers/__init__.py`) |
| other | `[]` and an end token |

`torch.py` is a copy of the OvisOCR2 fake. `transformers-5.19.0+fake.dist-info` provides the
package version. The processor and the model load only from a directory that contains
`config.json`.
