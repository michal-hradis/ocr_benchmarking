Fake `paddle`, `paddlex`, `paddleocr` and `cv2` modules for the offline protocol test in
`tests/test_paddleocr.py`. They run the adapter as a host process without PaddlePaddle or a GPU:
a 900-pixel-wide image returns the committed result `../page.json` (real PaddleOCR 3.7.0
output for a synthetic page), any other width an empty result, and a 666-pixel-wide image
raises an error inside the pipeline.
