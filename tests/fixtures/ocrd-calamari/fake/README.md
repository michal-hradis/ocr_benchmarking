Fake OCR-D modules (`ocrd`, `ocrd_utils`, the three processors) and `tesserocr` for the offline
protocol test in `tests/test_ocrd_calamari.py`. They run the adapter as a host process without
OCR-D, Tesseract, Calamari or TensorFlow. The fake `run_processor` writes the output PAGE of each
step; it picks the committed pipeline output by the width of the workspace image: `../page.xml` for
900 pixels, `../newspaper.xml` for 1500 and `../blank.xml` for anything else. Those files are real
output of the engine image for synthetic pages. A 666-pixel-wide image raises an error during
segmentation, and on a 555-pixel-wide one the recognition writes no PAGE file.
