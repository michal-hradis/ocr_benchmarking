Fake `torch`, `numpy` and `doctr` modules for the offline protocol test in `tests/test_doctr.py`.
They run the adapter as a host process without PyTorch, docTR, weights or a GPU. `ocr_predictor`
loads only when both pinned checkpoint files are in `$DOCTR_CACHE_DIR/models/` and returns the
committed native results (real docTR 1.1.0 page exports of synthetic pages) chosen by the image
width:

| Width | Result |
| --- | --- |
| 900 | `../page.json` |
| 1000 | `../columns.json` |
| 1500 | `../newspaper.json` |
| 300 | `../blank.json` |
| 666 | raises a fake CUDA out-of-memory error |
| other | a page without blocks |

The fake `numpy.asarray` passes the PIL image through. The `*.dist-info` directory provides the
package version.
