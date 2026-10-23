Fake `torch`, `transformers` and `surya` modules for the offline protocol test in
`tests/test_surya.py`. They run the adapter as a host process without PyTorch, Surya, weights or
a GPU. The predictors load only from a checkpoint directory that contains `config.json` and
return the committed native results (real Surya 0.17.1 output for synthetic pages, with the per-character boxes removed) chosen by the
image width:

| Width | Result |
| --- | --- |
| 900 | `../page.json` |
| 1000 | `../columns.json` |
| 1500 | `../newspaper.json` |
| 300 | `../blank.json` |
| 666 | raises a fake CUDA out-of-memory error |
| other | no lines and no layout blocks |

The recognizer checks it is given exactly the detected line polygons. The `*.dist-info`
directories provide the package versions.
