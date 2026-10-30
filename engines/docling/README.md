# Docling engine (standard document pipeline, GPU)

[Docling](https://github.com/docling-project/docling) as an end-to-end document conversion pipeline ([#77](https://github.com/michal-hradis/ocr_benchmarking/issues/77), RapidOCR presets [#78](https://github.com/michal-hradis/ocr_benchmarking/issues/78), tracker [#76](https://github.com/michal-hradis/ocr_benchmarking/issues/76)). Docling's standard pipeline detects layout regions, runs OCR on them, recognizes table structure and orders the result for reading, producing a DoclingDocument. The input is always one raster image, never a PDF or a text layer. The adapter runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-docling:2.145.0-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1`, without models |
| `requirements.in`, `requirements.txt` | Docling and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files; the adapter refuses to run unless all match |
| `fetch_model.py` | Downloads the models into the cache on the host (`make docling-model`) |
| `docling_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`, preset `docling` |
| `presets/*.toml` | The other presets ([Presets](#presets)) |

## Versions

- **Docling 2.145.0**, installed as `docling-slim` (the package the `docling` metapackage wraps) with only the extras the image pipeline needs: `models-local` (layout and TableFormer on torch), `feat-ocr-rapidocr-onnx` (RapidOCR 3.10.0 on onnxruntime 1.31.1) and `convert-core`. With docling-core 2.105.0 and docling-ibm-models 4.0.4, on the base's torch **2.14.1** for CUDA 13.0. No PDF, office or web format support is installed.
- **Models** (397 MB), pinned by revision and checked against `models.sha256`:

| Role | Model | Source and revision | License |
| --- | --- | --- | --- |
| Layout | Heron (RT-DETR, Docling's default layout model) | [`docling-project/docling-layout-heron`](https://huggingface.co/docling-project/docling-layout-heron) at `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8` (`main`, which Docling 2.145.0 asks for) | Apache-2.0 |
| Table structure | TableFormer, accurate mode | [`docling-project/docling-models`](https://huggingface.co/docling-project/docling-models) tag `v2.3.0` = `fc0f2d45e2218ea24bce5045f58a389aed16dc23` | CDLA-Permissive-2.0 |
| OCR | RapidOCR PP-OCRv5 mobile text detection, PP-OCRv5 Latin mobile recognition, v2.0 mobile line orientation | [RapidOCR's ModelScope release](https://www.modelscope.cn/models/RapidAI/RapidOCR) `v3.10.0` | Apache-2.0 |

- **Characters:** the Latin recognizer's character list (502 entries) is embedded in its ONNX file, so the checksum pins it too. It has all Czech letters (`áčďéěíňóřšťúůýž` and their capitals), the German umlauts and `ß`, `æ œ`, other accented Latin letters, `„ “ ” ‚ ‘ ’ « »`, dashes and `§ ¶ †`. It has no long s `ſ`, no r rotunda `ꝛ` and no non-Latin script such as Glagolitic. Fraktur is read with this Antiqua-trained model.

- **Licenses:** Docling, docling-core and docling-ibm-models are MIT and RapidOCR is Apache-2.0. The models' licenses are listed above.
- **Hardware:** an NVIDIA GPU with CUDA 13.0 (driver ≥ 580) for layout and TableFormer, tested on an RTX 4090. RapidOCR runs on the CPU, because the image has the CPU build of onnxruntime. Loading the pipeline takes about 16 s. Peak GPU memory allocated by torch was about 0.5 GB per page.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once
make docker-docling                 # this image (no models)
make docling-model                  # opt-in: the pinned models (397 MB) into OCRBENCH_CACHE
uv run ocrbench run start --engine engines/docling/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench   # or --engine engines/docling/presets/<preset>.toml
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/docling --inspect
make docling-smoke                  # opt-in: build the image, run it on synthetic samples on the GPU
```

`make docling-model` writes the models to `$(OCRBENCH_CACHE)/models/docling/`, the directory layout of Docling's `artifacts_path`. Files that are already present and correct are kept. The adapter checks every file against `models.sha256` before loading. It also checks that the configuration's `model` names exactly this pinned bundle (`MODEL` in the adapter), so a copied configuration cannot attribute results to other models than the ones that ran. Docling is given only this directory, and `HF_HUB_OFFLINE=1` is set, so inference needs no network.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | Docling's accelerator device for layout and TableFormer: `cuda:<n>` or `cpu` |
| `ocr_engine` | `rapidocr` | Docling's OCR engine. Only RapidOCR (onnxruntime backend, CPU) is supported so far; further engines come with their presets ([#79](https://github.com/michal-hradis/ocr_benchmarking/issues/79)–[#81](https://github.com/michal-hradis/ocr_benchmarking/issues/81)) |
| `ocr_lang` | `["latin"]` | RapidOCR language (Docling's `RapidOcrOptions.lang`). Only `["latin"]`, PP-OCRv5's Latin-script recognizer, is accepted, because it is the only recognizer whose model is pinned. The PP-OCRv5 Latin model covers Czech ([Characters](#versions)); other languages would need their own pinned models |
| `ocr_mode` | `default` | Docling's `OcrMode`: `default`, `layout_regions` or `full_page` ([OCR modes](#ocr-modes)) |
| `ocr_scale` | `3.0` | Docling's OCR render scale. It also sets the nominal resolution (below), so that Docling crops the input pixels 1:1 for OCR |
| `table_structure` | `true` | TableFormer (accurate mode) on table regions |
| `document_timeout` | `null` | Docling's own time limit per page in seconds. When it is reached, Docling returns what it has, and the sample is `partial`. The runner's `timeouts.sample` applies regardless |
| `num_threads` | `4` | Docling's accelerator threads, which also set RapidOCR's onnxruntime threads |

Fixed: Docling's automatic OCR engine selection is not used. RapidOCR runs with Docling 2.145.0's settings for it and RapidOCR 3.10.0's defaults otherwise; `result.json` records the ones that matter (`extra.rapidocr`): `text_score` 0.5 (lines recognized with less confidence are dropped), the resizing below, detection thresholds, batch sizes of 6, and the model files. The adapter checks that the detection, orientation and recognition models RapidOCR loaded are the pinned files and that the recognizer uses its embedded character list; anything else fails every sample, so no fallback model can run. Docling's page batch sizes do not matter, because every sample is one page. Remote services, external plugins, picture classification and description, code, formula and chart enrichment, and form-field extraction are off (Docling's defaults). Docling does not get the sample `language`.

## OCR modes

Docling 2.145.0's standard pipeline runs the layout model first and OCR after it (`StandardPdfPipeline`: preprocessing, layout, OCR, layout post-processing, tables, assembly, reading order). What the OCR engine reads depends on `OcrMode` (`BaseOcrModel.get_ocr_rects`), checked in the installed source:

- **`layout_regions`:** the bounding boxes of all layout clusters, drawn on a page mask in points, dilated by 20 pt (60 input pixels at 216 DPI) and merged into the bounding boxes of the connected areas (`_deduplicate_rects`). RapidOCR reads each merged box as one crop. Clusters less than about 20 pt apart therefore become one crop: on a dense page, nearly the whole page. Text outside every crop is not read.
- **`default`:** for a PDF, the layout regions that have no usable text layer (`pdf_aware_layout_regions`). An image has no text layer, and Docling's image backend reports the whole page as a bitmap, so every cluster qualifies: the same crops as `layout_regions`. The two modes then differ only in how OCR and PDF cells are merged, and an image has no PDF cells.
- **`full_page`:** one crop, the whole page (`force_full_page_ocr`, deprecated upstream, sets this mode). Layout post-processing then assigns the OCR cells to the layout clusters. A cell in no cluster becomes its own `text` cluster (`create_orphan_clusters`, on by default), so text outside the layout regions is kept as one-line regions.

**Resolution.** Docling crops each OCR box from the page rendered at `ocr_scale`, which the nominal DPI makes the input pixels 1:1. RapidOCR then shrinks any crop whose longer side exceeds `max_side_len` = 2000 px to that size, for both detection and recognition (`Global.use_preprocess_img`); crops shorter than 30 px are enlarged. Detection additionally scales its input so that the shorter side is at least 736 px (`Det.limit_type` `min`). The adapter does not change this upstream behavior, but records every crop in input pixels with RapidOCR's resize factor in `result.json` (`extra.ocr_crops`: `count`, `shrunk`, `min_resize` and `crops`). On dev-01 the layout modes read each newspaper as one or two crops that cover nearly the whole page, shrunk to 0.35 and 0.40, and the larger crop of two DTA pages is shrunk to 0.76 and 0.93. In `full_page` mode the page is the crop: 0.32 and 0.38 for the newspapers, 0.71–0.92 for the DTA pages.

## Presets

A preset is a frozen engine configuration; its `name` identifies it as a separate system in reports. All presets use the same image, models and settings except where the table says otherwise.

| Preset (file) | `name` | Settings |
| --- | --- | --- |
| `docling` (`engine.toml`) | `docling` | Docling's standard pipeline at its defaults: Heron layout, RapidOCR PP-OCRv5 Latin in the default OCR mode (for an image, the layout regions), TableFormer accurate, the rule-based reading order |
| `rapidocr-regions` (`presets/rapidocr-regions.toml`) | `docling-rapidocr-regions` | `ocr_mode = "layout_regions"`: RapidOCR on the layout regions, named explicitly instead of through Docling's PDF-aware default ([#78](https://github.com/michal-hradis/ocr_benchmarking/issues/78)) |
| `rapidocr-fullpage` (`presets/rapidocr-fullpage.toml`) | `docling-rapidocr-fullpage` | `ocr_mode = "full_page"`: RapidOCR on the whole page, cells assigned to the layout regions ([#78](https://github.com/michal-hradis/ocr_benchmarking/issues/78)) |

A new preset is a new TOML file with its own `name`; a changed setting is a new preset, not an edit of an existing one. Layout and OCR-scope variants are [#80](https://github.com/michal-hradis/ocr_benchmarking/issues/80), and the DocTags VLMs are [#82](https://github.com/michal-hradis/ocr_benchmarking/issues/82)–[#84](https://github.com/michal-hradis/ocr_benchmarking/issues/84).

## Image input and coordinates

Docling sizes an image page in points from the file's DPI metadata, assuming 72 DPI when there is none. It renders the OCR input at 72 DPI × `ocr_scale` and applies the EXIF orientation. Left alone, it would upsample a 4194 × 6192 newspaper without a DPI tag three times for OCR, and its positions would follow the EXIF view.

The adapter decodes the image as stored with Pillow, without applying the EXIF orientation ([PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)). It passes the pixels to Docling as an uncompressed TIFF with no EXIF data, at a **nominal resolution of 72 × `ocr_scale` DPI** (216 DPI by default), whatever the file's metadata says. The TIFF stores the resolution as an exact rational. A page of W × H pixels therefore becomes W/3 × H/3 points: Docling crops the original pixels 1:1 for OCR (RapidOCR may then shrink a crop, [OCR modes](#ocr-modes)), and the layout model gets the page at a third of its size, which it resizes to its own input anyway. Before converting, the adapter checks that Docling's page is exactly the input image at that resolution. A different size, for example a page turned by 90 degrees, fails the sample. Positions are mapped back to input pixels by width / page width and height / page height, and from Docling's bottom-left origin (provenance boxes) or top-left origin (layout clusters, OCR cells) to the top-left origin.

## Outputs

For every sample the adapter writes the following files into the sample directory:

| File | Content |
| --- | --- |
| `docling.json` | The DoclingDocument (`export_to_dict()`), unchanged |
| `docling.md` | Its Markdown export (`export_to_markdown()`), unchanged. Not scored |
| `docling_layout.json` | Docling's layout clusters of the page with their OCR cells and child clusters (`page.predictions.layout`), unchanged |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output, converted from the two JSON files above |
| `result.json` | Status, processing time and, in `extra`: Docling's conversion status and errors, the warnings Docling and RapidOCR logged (up to 50), the nominal DPI, peak GPU memory, region/line/cell counts, the versions, GPU and load time, the checksums of the model files, the parameters, Docling's full pipeline options and RapidOCR's settings in effect |

**Mapping** (decided in [#77](https://github.com/michal-hradis/ocr_benchmarking/issues/77#issuecomment-6532444998)):

- **Regions:** the items of the DoclingDocument are read in document order. This is a pre-order walk from the body, as `DoclingDocument.iterate_items` does with all content layers and with the children of pictures. Docling's reading-order model builds this order. Captions are children of their table or picture and follow it. Page headers and footers keep their place. Text that Docling recognized inside a picture is kept: these are child items of the picture that the Markdown export leaves out. Every provenance box of an item becomes one region `r<n>`, with the box as its polygon and the item's label (for example `section_header`) as `native_type`. An item that Docling merged from several layout clusters (a paragraph continued in another column) therefore becomes several regions, in order. The region `type` follows the label: titles and section headers are `heading`; text, paragraphs and references are `text`; then `list-item`, `caption`, `footnote`, `page-header` and `page-footer`; tables and indexes are `table`; formulas are `formula`; pictures and charts are `figure`; anything else is `other`. `extra` holds the item's `self_ref`, the provenance index, the cluster ID and the layout confidence.
- **Lines:** every provenance box is matched to the layout cluster it came from. The box and the cluster bounding box are the same rectangle, so they are matched to within 0.001 pt. If several clusters match, the one with the item's label is preferred. The region's lines are that cluster's OCR cells with non-blank text, in Docling's cell order. Each line keeps its raw OCR text, its 4-point box and its confidence. Its ID is `l<cell index>`. Cells that belong to a child cluster that is itself an item (text inside a picture) go only to that child's region. An OCR cell is one RapidOCR text box: usually a whole printed line, sometimes part of one. The adapter does not merge cells. A table's lines are its cells' OCR text, not TableFormer's structure, which stays in `docling.json`. Regions without cells, such as pictures, have no text.
- **Text:** the page text is the lines joined by line breaks, with regions separated by blank lines. Docling's own item text stays in the native files and is not scored. Docling builds it from the same cells but joins the lines with spaces, removes a hyphen at a line end and replaces `’ ‘ “ ”` with ASCII quotes.
- **Cells outside items:** non-blank OCR cells in clusters that no DoclingDocument item uses are left out, not invented into the prediction. Their number is `counts.cells_outside_regions` in `result.json`, and a non-zero number is reported in the sample's `message`, for example "3 of 120 non-blank OCR cells are in no DoclingDocument item and are left out". The cells stay in `docling_layout.json`.
- **Checks:** a box that matches no unused cluster, a cell that would go to two regions, a document that is not exactly one page, and an unknown coordinate origin all fail the sample. These indicate a Docling result that the mapping does not understand.

**Status.** An image that cannot be decoded, a conversion that Docling reports as failed, or an error such as CUDA out of memory fails only that sample (`status: failed`, runner `engine-error`), and the container continues. A partial conversion (for example after `document_timeout`) keeps its outputs and is `partial`. An unusable GPU, missing or changed models, a `model` other than the pinned bundle, or invalid parameters fail every sample. A page on which nothing is detected is a **genuine empty result**: `success` with empty text and no regions.

## Measured on dev-01

All presets on the seven pages of dev-01, RTX 4090, default evaluation configuration (`strict`, empty-output failure policy). Runs (not in Git): `docling-dev01` (image `sha256:993ecbd6…`, #77) and `docling-rapidocr-regions-dev01`, `docling-rapidocr-fullpage-dev01` (image `sha256:d13605b8…`, which adds only the recorded settings and crops). All 7 samples of every run were `success`. Rates are micro averages over the pages. fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)); the evaluator marks them as not proven optimal on 6–7 of the 7 pages. Engine time is the sum of the adapter's per-sample times. Loading the pipeline took another 15–16 s.

| Preset | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time (7 pages) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `docling` | 64.8 % | 82.7 % | 37.0 % | 55.7 % | 60.5 | 40.0 | 73 s |
| `docling-rapidocr-regions` | 64.8 % | 82.7 % | 37.0 % | 55.7 % | 60.5 | 40.0 | 71 s |
| `docling-rapidocr-fullpage` | 66.2 % | 85.0 % | 37.4 % | 57.0 % | 60.3 | 39.0 | 70 s |

CER per page in % (fCER in parentheses where it differs). Lines are the OCR cells in the prediction against the reference text lines. `docling-rapidocr-regions` is identical to `docling` on every page, so it shares its columns.

| Page | Content | Lines (`docling` and regions / full page / GT) | `docling` and regions CER (fCER) | Full page CER (fCER) |
| --- | --- | ---: | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 338 / 313 / 419 | 75.3 (39.5) | 75.5 (45.8) |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 365 / 382 / 484 | 69.3 (40.1) | 72.2 (36.6) |
| `CkXII61-0017` | Glagolitic print | 15 / 12 / 19 | 90.2 | 90.6 |
| `dta-346672…` | Latin, Antiqua and italic | 31 / 31 / 34 | 43.6 (23.9) | 43.6 (23.5) |
| `dta-laube…` | German Fraktur, 1833 | 29 / 24 / 25 | 11.1 | 10.4 |
| `dta-martzi…` | German Fraktur, 1692 | 34 / 29 / 29 | 17.5 | 17.6 |
| `dta-steinen…` | German Antiqua, 1894 | 46 / 41 / 44 | 17.3 (14.1) | 17.1 (13.7) |

Observations (measured, unless marked):

- **`default` and `layout_regions` are the same for an image.** The two runs produced byte-identical predictions on all seven pages, as the source reading in [OCR modes](#ocr-modes) predicts. The explicit preset only removes the dependence on Docling's PDF-aware default.
- **Region OCR reads the newspapers almost as full pages.** Docling merges the layout clusters of a newspaper into one or two crops that cover nearly the whole page, and RapidOCR shrinks them to 0.35 and 0.40 of the input resolution. The full-page preset shrinks them to 0.32 and 0.38. Both lose text: the predictions are 21–28 % shorter than the reference in region mode and 23–24 % in full-page mode, and every OCR cell is in a region (`cells_outside_regions` is 0 in all runs). That the shrinking causes these deletions is likely but was not tested (speculation); `max_side_len` is RapidOCR's default and no preset changes it.
- **Full-page OCR changes the newspapers in both directions.** fCER is 6.3 points worse on `3e6309d0…` (25 fewer cells) and 3.5 points better on `bd497a62…` (17 more cells). On the DTA pages it returns up to 5 fewer cells, and CER changes by less than 1 point. The order of the newspaper paragraphs is Docling's in both modes: strict CER is about twice fCER.
- **The Latin recognizer has no Glagolitic**, and the Fraktur pages are read with a Latin Antiqua model without `ſ` ([Characters](#versions)).
- The development set is small and not representative; these numbers are not a ranking of Docling or its OCR modes.

## Tests

`tests/test_docling.py` checks the mapping on committed native results, all offline in `make test`. The fixtures are in `tests/fixtures/docling/<name>/` (`docling.json` and `docling_layout.json`). They are real Docling 2.145.0 output of this preset for synthetic pages rendered by `tests/synthetic.py`:

- `page`
- `paragraph`: a crop that Docling's layout takes for a picture with text inside
- `columns`
- `newspaper`: three columns
- `table`: a ruled table, a picture and a caption
- `blank`

The tests check the regions in document order with their OCR cells, the page text and line positions against the rendered page, the provenance boxes in input pixels, tables and pictures, text inside a picture read once, Docling's newspaper order and a blank page. On small hand-made results, they also check raw cells instead of Docling's joined text, both coordinate origins, merged items, label preference, blank cells and rejected inconsistent results. Further offline tests cover the page-size check, the TIFF input (stored pixels, no EXIF, nominal DPI), parameters, model sources and `fetch_model.py`. The engine protocol is tested through the runner with fake `torch` and `docling` modules (`tests/fixtures/docling/fake`), including a CUDA error, a failed and a partial conversion, a turned page and OCR cells outside every item between successful samples, a `model` that names other models, the CPU device, the OCR mode of each preset and setup errors. Further tests check that the presets differ from `docling` only in name and OCR mode, the recorded OCR crops (in input pixels, both origins, with RapidOCR's resize factor) and that RapidOCR settings with any model other than the pinned files are rejected.

`make docling-smoke` builds the image and runs real Docling, with each of the three presets, through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report`. The samples are the synthetic full page, a paragraph crop, a blank page (an empty `success`), the two-column, newspaper and table pages, and the full page saved as a 300 DPI JPEG with EXIF orientation 6. Ordered CER is below 10 % on all except the newspaper, where fCER is below 10 %, and the JPEG's positions match the stored pixels. Each run must record the OCR mode, RapidOCR's settings and its crops (one per page in full-page mode, none shrunk on these small pages). A run with a GPU that is not visible must fail explicitly.
