Fake `torch`, `cv2` and `pero_ocr` modules for the offline protocol test in `tests/test_pero.py`.
They run the adapter as a host process without PyTorch, OpenCV, pero-ocr or a GPU. The fake
`PageParser` takes its stage (layout analysis or OCR) from the INI it is given. On a 900-pixel-wide
image, layout analysis finds the regions and lines of the committed PERO output `../page.xml`, which
the fake `PageLayout` then serializes together with `../alto.xml`. That output is real pero-ocr
output for a synthetic page. On any other width layout analysis finds nothing. A
666-pixel-wide image raises an error during layout analysis, and on a 555-pixel-wide one the ALTO
export fails. `pero_ocr-0.7.0+fake.dist-info` provides the package version.
