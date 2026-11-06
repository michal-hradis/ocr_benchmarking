Fake `indra_ocr` and `cv2` modules for the offline protocol test in `tests/test_indra_ocr.py`.
They run the adapter as a host process without ONNX Runtime, OpenCV, numpy or the models. `cv2.imread` only
reads the image size; `indra_ocr.load` needs a model directory with `meta.json`, writes its
arguments to `$FAKE_INDRA_LOG` when set, and its converter returns the committed native pages (real Indra OCR 1.1.2
output of the engine image for synthetic pages, `indra-ocr --format json` fields) chosen by the image size:

| Size | Page |
| --- | --- |
| 900×420 | `../page.json` (tests/synthetic.py `PAGE`) |
| 1000×420 | `../columns.json` (`COLUMNS`) |
| 1500×620 | `../newspaper.json` (`NEWSPAPER`) |
| 300×200 | `../blank.json` (a blank page) |
| 420×900 | `../turned.json` (`PAGE` turned a quarter turn counter-clockwise; Indra turns it back, `rotation` 90) |
| 520×60 | `../line.json` (one line, read without layout blocks) |
| width 666 | raises a fake ONNX Runtime error |
| other | no lines and no blocks |

The `*.dist-info` directories provide the package versions.
