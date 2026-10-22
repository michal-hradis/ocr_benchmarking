Fake `paddle`, `paddlex`, `paddleocr` and `cv2` modules for the offline protocol test in
`tests/test_paddleocr.py`. They run the adapter as a host process without PaddlePaddle or a GPU.
The fake pipeline chooses its result by the image width:

| Width | Result |
| --- | --- |
| 900 | `../page.json`, real PaddleOCR 3.7.0 output for a synthetic page |
| 5400 | `../page-x6-limit<4000\|8000>.json` for the configured `max_side_limit`: real PaddleOCR 3.7.0 output for the synthetic page scaled 6x (5400 x 2520), detected at 4000 (downscaled) or 8000 (full resolution) |
| 666 | an error inside the pipeline |
| 777 | a Paddle CUDA out-of-memory error (`MemoryError` with a C++ traceback) |
| other | an empty result |

The fake reports the configured `max_side_limit` in `text_det_params` like the real pipeline;
with `FAKE_PADDLEOCR_IGNORE_LIMIT=1` it reports 4000 regardless, as a PaddleOCR that ignored the
setting would.
