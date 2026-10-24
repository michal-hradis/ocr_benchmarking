Fake `mineru_vl_utils`, `torch` and `transformers` modules for the offline tests in
`tests/test_mineru25_pro.py`. They run the adapter as a host process without mineru-vl-utils,
PyTorch, transformers, weights or a GPU.

`MinerUClient.two_step_extract` returns committed results, chosen by the image width: real
MinerU2.5-Pro-2605-1.2B output of mineru-vl-utils 2.0.7 (`mineru.json` of the engine) for the
synthetic pages of `tests/synthetic.py`:

| Width | Result |
| --- | --- |
| 900 | `../page.json` for `PAGE` |
| 520 | `../paragraph.json` for `PARAGRAPH` (the layout step labels the paragraph a title) |
| 1000 | `../columns.json` for `COLUMNS` |
| 1500 | `../newspaper.json` for `NEWSPAPER` |
| 2000 | one text block whose generation never ends (numbered newspaper lines) |
| 666 | raises a fake CUDA out-of-memory error |
| other | no blocks, as for the real blank page |

Like the real transformers backend, it calls the model's `generate` once for the layout (2
tokens per block, after 1,369 image tokens for the 1036 × 1036 page) and once per block it reads
(one token per word, one image token per 28 × 28 pixels of the crop), with MinerU's sampling
parameters: `max_new_tokens` when it is set, otherwise `max_length` of the 8,192-token context.
A generation that reaches its limit ends without `<|im_end|>`; its blocks or words are cut to the
tokens generated. `post_process.json2md` behaves as in mineru-vl-utils 2.0.7.

`torch.py` is a copy of the GLM-OCR fake. `transformers-5.19.0+fake.dist-info` provides the
package version. The model and processor load only from a directory that contains `config.json`.
