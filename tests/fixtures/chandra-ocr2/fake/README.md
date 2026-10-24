Fake `torch` and `transformers` modules for the offline protocol test in `tests/test_chandra_ocr2.py`.
They run the adapter as a host process without PyTorch, transformers, weights or a GPU. The fake
processor checks the following:

- the conversation is one user turn with the image followed by `engines/chandra-ocr2/prompt.txt`;
- the chat template is applied as in `chandra.model.hf` (tokenized, batched, left padding);
- the adapter has already scaled the image to whole 28-pixel units, as `scale_to_fit` does.

It resizes like the Qwen2-VL image processor (32-pixel units, at least 65,536 pixels) and puts one
image token per 32 × 32 pixels into the prompt. The fake model checks that generation is greedy and
stops at both end tokens (`<|endoftext|>` from `generation_config.json`, then `<|im_end|>`). It
"generates" one token per character, chosen by the width of the image given to the processor:

| Width | Sample image | Output |
| --- | --- | --- |
| 3304 | 900 × 420 | `../page.html` (hand-written layout HTML) and `<\|im_end\|>` |
| 2296 | 300 × 200 | only an end token (a blank page) |
| 4480 | 1280 × 400 | a `Blank-Page` block and `<\|endoftext\|>` |
| 2548 | 666 × 400 | raises a fake CUDA out-of-memory error |
| 4312 | 1200 × 400 | a sentence without layout blocks |
| 1988 | 2000 × 3000 | numbered newspaper blocks that never end (`newspaper_block` in `transformers/__init__.py`) |
| other | | only an end token |

`torch.py` is a copy of the Infinity-Parser2 fake. `transformers-5.19.0+fake.dist-info` provides
the package version. The processor and the model load only from a directory that contains
`config.json`.
