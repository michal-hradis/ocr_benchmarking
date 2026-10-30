# Docling engine (standard document pipeline, GPU)

[Docling](https://github.com/docling-project/docling) as an end-to-end document conversion pipeline ([#77](https://github.com/michal-hradis/ocr_benchmarking/issues/77), tracker [#76](https://github.com/michal-hradis/ocr_benchmarking/issues/76)). Docling's standard pipeline detects layout regions, runs OCR on them, recognizes table structure and orders the result for reading, producing a DoclingDocument. The input is always one raster image, never a PDF or a text layer. The adapter runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-docling:2.145.0-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1`, without models |
| `requirements.in`, `requirements.txt` | Docling and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files; the adapter refuses to run unless all match |
| `fetch_model.py` | Downloads the models into the cache on the host (`make docling-model`) |
| `docling_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`, preset `docling` |

## Versions

- **Docling 2.145.0**, installed as `docling-slim` (the package the `docling` metapackage wraps) with only the extras the image pipeline needs: `models-local` (layout and TableFormer on torch), `feat-ocr-rapidocr-onnx` (RapidOCR 3.10.0 on onnxruntime 1.31.1) and `convert-core`. With docling-core 2.105.0 and docling-ibm-models 4.0.4, on the base's torch **2.14.1** for CUDA 13.0. No PDF, office or web format support is installed.
- **Models** (397 MB), pinned by revision and checked against `models.sha256`:

| Role | Model | Source and revision | License |
| --- | --- | --- | --- |
| Layout | Heron (RT-DETR, Docling's default layout model) | [`docling-project/docling-layout-heron`](https://huggingface.co/docling-project/docling-layout-heron) at `8f39ad3c0b4c58e9c2d2c84a38465abf757272d8` (`main`, which Docling 2.145.0 asks for) | Apache-2.0 |
| Table structure | TableFormer, accurate mode | [`docling-project/docling-models`](https://huggingface.co/docling-project/docling-models) tag `v2.3.0` = `fc0f2d45e2218ea24bce5045f58a389aed16dc23` | CDLA-Permissive-2.0 |
| OCR | RapidOCR PP-OCRv5 mobile text detection, PP-OCRv5 Latin mobile recognition, v2.0 mobile line orientation | [RapidOCR's ModelScope release](https://www.modelscope.cn/models/RapidAI/RapidOCR) `v3.10.0` | Apache-2.0 |

- **Licenses:** Docling, docling-core and docling-ibm-models are MIT and RapidOCR is Apache-2.0. The models' licenses are listed above.
- **Hardware:** an NVIDIA GPU with CUDA 13.0 (driver ≥ 580) for layout and TableFormer, tested on an RTX 4090. RapidOCR runs on the CPU, because the image has the CPU build of onnxruntime. Loading the pipeline takes about 16 s. Peak GPU memory allocated by torch was about 0.5 GB per page.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once
make docker-docling                 # this image (no models)
make docling-model                  # opt-in: the pinned models (397 MB) into OCRBENCH_CACHE
uv run ocrbench run start --engine engines/docling/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/docling --inspect
make docling-smoke                  # opt-in: build the image, run it on synthetic samples on the GPU
```

`make docling-model` writes the models to `$(OCRBENCH_CACHE)/models/docling/`, the directory layout of Docling's `artifacts_path`. Files that are already present and correct are kept. The adapter checks every file against `models.sha256` before loading. Docling is given only this directory, and `HF_HUB_OFFLINE=1` is set, so inference needs no network.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | Docling's accelerator device for layout and TableFormer: `cuda:<n>` or `cpu` |
| `ocr_engine` | `rapidocr` | Docling's OCR engine. Only RapidOCR (onnxruntime backend, CPU) is supported so far; further engines come with their presets ([#78](https://github.com/michal-hradis/ocr_benchmarking/issues/78)–[#81](https://github.com/michal-hradis/ocr_benchmarking/issues/81)) |
| `ocr_lang` | `["latin"]` | One RapidOCR language: a PP-OCR code or an `iso:` BCP-47 tag (Docling's `RapidOcrOptions.lang`). `latin` is PP-OCRv5's Latin-script recognizer. Other languages need their models in `models.sha256` |
| `ocr_mode` | `default` | Docling's `OcrMode`: `default` (for an image: OCR of each layout region, Docling's default), `layout_regions` or `full_page` |
| `ocr_scale` | `3.0` | Docling's OCR render scale. It also sets the nominal resolution (below), so that OCR always sees the input pixels 1:1 |
| `table_structure` | `true` | TableFormer (accurate mode) on table regions |
| `document_timeout` | `null` | Docling's own time limit per page in seconds. When it is reached, Docling returns what it has, and the sample is `partial`. The runner's `timeouts.sample` applies regardless |
| `num_threads` | `4` | Docling's accelerator threads, which also set RapidOCR's onnxruntime threads |

Fixed: Docling's automatic OCR engine selection is not used. Remote services, external plugins, picture classification and description, code, formula and chart enrichment, and form-field extraction are off (Docling's defaults). Docling does not get the sample `language`.

## Preset

A preset is a frozen engine configuration; its `name` identifies it as a separate system in reports.

| Preset (file) | `name` | Settings |
| --- | --- | --- |
| `docling` (`engine.toml`) | `docling` | Docling's standard pipeline at its defaults: Heron layout, RapidOCR PP-OCRv5 Latin in the default (layout-region) OCR mode, TableFormer accurate, the rule-based reading order |

A new preset is a new TOML file with its own `name`; a changed setting is a new preset, not an edit of an existing one. RapidOCR variants are [#78](https://github.com/michal-hradis/ocr_benchmarking/issues/78), layout and OCR-scope variants are [#80](https://github.com/michal-hradis/ocr_benchmarking/issues/80), and the DocTags VLMs are [#82](https://github.com/michal-hradis/ocr_benchmarking/issues/82)–[#84](https://github.com/michal-hradis/ocr_benchmarking/issues/84).

## Image input and coordinates

Docling sizes an image page in points from the file's DPI metadata, assuming 72 DPI when there is none. It renders the OCR input at 72 DPI × `ocr_scale` and applies the EXIF orientation. Left alone, it would upsample a 4194 × 6192 newspaper without a DPI tag three times for OCR, and its positions would follow the EXIF view.

The adapter decodes the image as stored with Pillow, without applying the EXIF orientation ([PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)). It passes the pixels to Docling as an uncompressed TIFF with no EXIF data, at a **nominal resolution of 72 × `ocr_scale` DPI** (216 DPI by default), whatever the file's metadata says. The TIFF stores the resolution as an exact rational. A page of W × H pixels therefore becomes W/3 × H/3 points: OCR sees the original pixels 1:1, and the layout model gets the page at a third of its size, which it resizes to its own input anyway. Before converting, the adapter checks that Docling's page is exactly the input image at that resolution. A different size, for example a page turned by 90 degrees, fails the sample. Positions are mapped back to input pixels by width / page width and height / page height, and from Docling's bottom-left origin (provenance boxes) or top-left origin (layout clusters, OCR cells) to the top-left origin.

## Outputs

For every sample the adapter writes the following files into the sample directory:

| File | Content |
| --- | --- |
| `docling.json` | The DoclingDocument (`export_to_dict()`), unchanged |
| `docling.md` | Its Markdown export (`export_to_markdown()`), unchanged. Not scored |
| `docling_layout.json` | Docling's layout clusters of the page with their OCR cells and child clusters (`page.predictions.layout`), unchanged |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output, converted from the two JSON files above |
| `result.json` | Status, processing time and, in `extra`: Docling's conversion status and errors, the warnings Docling and RapidOCR logged (up to 50), the nominal DPI, peak GPU memory, region/line/cell counts, the versions, GPU and load time, the checksums of the model files, the parameters and Docling's full pipeline options |

**Mapping** (decided in [#77](https://github.com/michal-hradis/ocr_benchmarking/issues/77#issuecomment-6532444998)):

- **Regions:** the items of the DoclingDocument are read in document order. This is a pre-order walk from the body, as `DoclingDocument.iterate_items` does with all content layers and with the children of pictures. Docling's reading-order model builds this order. Captions are children of their table or picture and follow it. Page headers and footers keep their place. Text that Docling recognized inside a picture is kept: these are child items of the picture that the Markdown export leaves out. Every provenance box of an item becomes one region `r<n>`, with the box as its polygon and the item's label (for example `section_header`) as `native_type`. An item that Docling merged from several layout clusters (a paragraph continued in another column) therefore becomes several regions, in order. The region `type` follows the label: titles and section headers are `heading`; text, paragraphs and references are `text`; then `list-item`, `caption`, `footnote`, `page-header` and `page-footer`; tables and indexes are `table`; formulas are `formula`; pictures and charts are `figure`; anything else is `other`. `extra` holds the item's `self_ref`, the provenance index, the cluster ID and the layout confidence.
- **Lines:** every provenance box is matched to the layout cluster it came from. The box and the cluster bounding box are the same rectangle, so they are matched to within 0.001 pt. If several clusters match, the one with the item's label is preferred. The region's lines are that cluster's OCR cells with non-blank text, in Docling's cell order. Each line keeps its raw OCR text, its 4-point box and its confidence. Its ID is `l<cell index>`. Cells that belong to a child cluster that is itself an item (text inside a picture) go only to that child's region. An OCR cell is one RapidOCR text box: usually a whole printed line, sometimes part of one. The adapter does not merge cells. A table's lines are its cells' OCR text, not TableFormer's structure, which stays in `docling.json`. Regions without cells, such as pictures, have no text.
- **Text:** the page text is the lines joined by line breaks, with regions separated by blank lines. Docling's own item text stays in the native files and is not scored. Docling builds it from the same cells but joins the lines with spaces, removes a hyphen at a line end and replaces `’ ‘ “ ”` with ASCII quotes.
- **Checks:** a box that matches no unused cluster, a cell that would go to two regions, a document that is not exactly one page, and an unknown coordinate origin all fail the sample. These indicate a Docling result that the mapping does not understand.

**Status.** An image that cannot be decoded, a conversion that Docling reports as failed, or an error such as CUDA out of memory fails only that sample (`status: failed`, runner `engine-error`), and the container continues. A partial conversion (for example after `document_timeout`) keeps its outputs and is `partial`. An unusable GPU, missing or changed models, or invalid parameters fail every sample. A page on which nothing is detected is a **genuine empty result**: `success` with empty text and no regions.

## Measured on dev-01

Preset `docling` on the seven pages of dev-01: run `docling-dev01`, not in Git; image `sha256:993ecbd6…`; RTX 4090; default evaluation configuration (`strict`, empty-output failure policy). All 7 samples were `success`. Rates are micro averages over the pages. fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)); the evaluator marks them as not proven optimal on all 7 pages. Engine time is the sum of the adapter's per-sample times. Loading the pipeline took another 16 s.

| Preset | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time (7 pages) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `docling` | 64.8 % | 82.7 % | 37.0 % | 55.7 % | 60.5 | 40.0 | 73 s |

CER per page in % (fCER in parentheses where it differs). Lines are the OCR cells in the prediction against the reference text lines.

| Page | Content | Lines (pred / GT) | Regions | CER (fCER) |
| --- | --- | ---: | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 338 / 419 | 32 | 75.3 (39.5) |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 365 / 484 | 54 | 69.3 (40.1) |
| `CkXII61-0017` | Glagolitic print | 15 / 19 | 5 | 90.2 |
| `dta-346672…` | Latin, Antiqua and italic | 31 / 34 | 18 | 43.6 (23.9) |
| `dta-laube…` | German Fraktur, 1833 | 29 / 25 | 10 | 11.1 |
| `dta-martzi…` | German Fraktur, 1692 | 34 / 29 | 6 | 17.5 |
| `dta-steinen…` | German Antiqua, 1894 | 46 / 44 | 10 | 17.3 (14.1) |

Observations (measured, unless marked):

- **Text is lost on the newspapers before conversion.** The layout clusters cover 85–89 % of the newspaper pages, and every OCR cell is in a region (`cells_outside_regions` is 0). Docling's own Markdown export has slightly fewer words than the prediction, which also contains page headers and footers. Yet RapidOCR returns only 338 and 365 cells for 419 and 484 reference lines. The predictions are 21 % and 28 % shorter than the reference, and deletions are 62 % and 76 % of the flexible character errors. The likely cause is RapidOCR's default detection resolution on large region crops, but this was not analysed (speculation).
- **The order of the newspaper paragraphs is Docling's.** Strict CER is about twice fCER on the newspapers. On the synthetic three-column page, Docling reads the paragraphs row by row across the columns, and the adapter keeps that order.
- **The Latin recognizer has no Glagolitic**, and the Fraktur pages are read with a Latin Antiqua model.
- The development set is small and not representative; these numbers are not a ranking of Docling.

## Tests

`tests/test_docling.py` checks the mapping on committed native results, all offline in `make test`. The fixtures are in `tests/fixtures/docling/<name>/` (`docling.json` and `docling_layout.json`). They are real Docling 2.145.0 output of this preset for synthetic pages rendered by `tests/synthetic.py`:

- `page`
- `paragraph`: a crop that Docling's layout takes for a picture with text inside
- `columns`
- `newspaper`: three columns
- `table`: a ruled table, a picture and a caption
- `blank`

The tests check the regions in document order with their OCR cells, the page text and line positions against the rendered page, the provenance boxes in input pixels, tables and pictures, text inside a picture read once, Docling's newspaper order and a blank page. On small hand-made results, they also check raw cells instead of Docling's joined text, both coordinate origins, merged items, label preference, blank cells and rejected inconsistent results. Further offline tests cover the page-size check, the TIFF input (stored pixels, no EXIF, nominal DPI), parameters, model sources and `fetch_model.py`. The engine protocol is tested through the runner with fake `torch` and `docling` modules (`tests/fixtures/docling/fake`), including a CUDA error, a failed and a partial conversion and a turned page between successful samples, the CPU device and setup errors.

`make docling-smoke` builds the image and runs real Docling through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report`. The samples are the synthetic full page, a paragraph crop, a blank page (an empty `success`), the two-column, newspaper and table pages, and the full page saved as a 300 DPI JPEG with EXIF orientation 6. Ordered CER is below 10 % on all except the newspaper, where fCER is below 10 %, and the JPEG's positions match the stored pixels. A run with a GPU that is not visible must fail explicitly.
