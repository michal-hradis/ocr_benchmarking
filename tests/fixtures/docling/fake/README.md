Fake `torch` and `docling` modules for the offline Docling adapter tests (`tests/test_docling.py`),
and `.dist-info` directories so that `importlib.metadata` reports the fake versions. They return
the committed real Docling outputs next to this directory; see `fake/docling/document_converter.py`.
