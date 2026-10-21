Fake `torch` and `kraken` modules for the offline protocol test in `tests/test_kraken.py`. They run
the adapter as a host process without PyTorch, Kraken or a GPU: on a 900-pixel-wide image the
segmenter and recognizer return the committed result `../page.json` (real Kraken 7.1.1 output for
a synthetic page), on any other width nothing, and a 666-pixel-wide image raises an error inside
the segmenter. `kraken-7.1.1+fake.dist-info` provides the package version, and the model files
(`kraken/blla.mlmodel`, `models/`) only name the kind of model the fake loads.
