Fake `paddleocr` module (`PaddleOCRVL`) for the offline tests in `tests/test_paddleocr_vl.py`,
used together with the fake `paddle`, `paddlex` and `cv2` modules of
`tests/fixtures/paddleocr/fake`. It runs the adapter as a host process without PaddlePaddle or a
GPU. The fake pipeline chooses its result by the image width:

| Width | Result |
| --- | --- |
| 900 | `../page.json`, real PaddleOCR-VL-1.6 output for the synthetic `PAGE` |
| 1000 | `../columns.json`, real output for the synthetic two-column page |
| 1500 | `../newspaper.json`, real output for the synthetic three-column newspaper page |
| 666 | an error inside the pipeline |
| 777 | a Paddle CUDA out-of-memory error (`MemoryError` with a C++ traceback) |
| other | `../blank.json`, the real result for a blank page, with that image size |

For every block with text, `predict` calls the VL model's `generate` the way the real pipeline
does (one call per block): one token per word followed by the end token, or, when the block has
more words than `max_new_tokens`, exactly `max_new_tokens` tokens without the end token, with the
content cut to that many words.
