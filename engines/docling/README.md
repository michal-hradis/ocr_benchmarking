# Docling engine (standard document pipeline, GPU)

[Docling](https://github.com/docling-project/docling) as an end-to-end document conversion pipeline ([#77](https://github.com/michal-hradis/ocr_benchmarking/issues/77), RapidOCR presets [#78](https://github.com/michal-hradis/ocr_benchmarking/issues/78), Tesseract presets [#79](https://github.com/michal-hradis/ocr_benchmarking/issues/79), tracker [#76](https://github.com/michal-hradis/ocr_benchmarking/issues/76)). Docling's standard pipeline detects layout regions, runs OCR on them, recognizes table structure and orders the result for reading, producing a DoclingDocument. The input is always one raster image, never a PDF or a text layer. The adapter runs in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-docling:2.145.0-v2` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1`, without models; a build stage compiles tesserocr |
| `requirements.in`, `requirements.txt` | Docling and its dependencies, hash-locked against the base (`make docker-lock`) |
| `tesserocr-build.in`, `tesserocr-build.txt` | Hash-locked build requirements of tesserocr, used only in the build stage (`make docker-lock`) |
| `models.sha256` | Checksums of the model files; the adapter refuses to run unless all match |
| `tessdata_best.sha256` | Checksums of the `tessdata_best` models for the Tesseract presets, a copy of [the Tesseract engine's](../tesseract/README.md#versions) (a test keeps them equal) |
| `tessdata.sha256` | Checksum of the `osd` model in the image, checked when the image is built |
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

- **Tesseract** (Tesseract presets, [#79](https://github.com/michal-hradis/ocr_benchmarking/issues/79)): **Tesseract 5.3.4** with Leptonica 1.82.0 from the Ubuntu 24.04 packages `tesseract-ocr=5.3.4-1build5` and `libtesseract5=5.3.4-1build5`, for Docling's command-line backend. **tesserocr 2.12.0** is used for its bindings backend. It is compiled in a build stage from its PyPI source release (SHA-256 checked) against `libtesseract-dev=5.3.4-1build5`, with Cython 3.3.0, setuptools 80.9.0 and wheel 0.45.1 (`tesserocr-build.txt`). The PyPI wheels would bundle a different Tesseract, so with this build both backends run the same library. Both read the models given below; the command-line run also uses Tesseract's output configs (`tsv`) from the image. The packages' own `tessdata_fast` `eng` model is installed but not used.
- **Tesseract models:** **tessdata_best** `ces`, `deu`, `eng` and `deu_latf` at commit `e12c65a915945e4c28e237a9b52bc4a8f39a0cec`. These are the files of the standalone [Tesseract engine's best presets](../tesseract/README.md#versions) (`make tesseract-best`, 48 MB), read from the same place in the cache and checked against `tessdata_best.sha256`. They are float LSTM models with no legacy engine. `deu_latf` is German Fraktur, formerly `frk`. `osd` (orientation and script detection), which Docling runs on every crop, comes from the Ubuntu package `tesseract-ocr-osd=1:4.1.0-2`. It is the same file as the standalone engine's `osd` (`tessdata.sha256`). `tessdata_fast` is not used: `deu_latf` exists only in `tessdata_best`, and these presets should be comparable to the standalone `tesseract-best-*` presets.
- **Characters:** the Latin recognizer's character list (502 entries) is embedded in its ONNX file, so the checksum pins it too. It has all Czech letters (`áčďéěíňóřšťúůýž` and their capitals), the German umlauts and `ß`, `æ œ`, other accented Latin letters, `„ “ ” ‚ ‘ ’ « »`, dashes and `§ ¶ †`. It has no long s `ſ`, no r rotunda `ꝛ` and no non-Latin script such as Glagolitic. Fraktur is read with this Antiqua-trained model.

- **Licenses:** Docling, docling-core and docling-ibm-models are MIT, RapidOCR is Apache-2.0, Tesseract and its models are Apache-2.0, and tesserocr is MIT. The other models' licenses are listed above.
- **Hardware:** an NVIDIA GPU with CUDA 13.0 (driver ≥ 580) for layout and TableFormer, tested on an RTX 4090. RapidOCR and Tesseract run on the CPU, because the image has the CPU build of onnxruntime and Tesseract has no GPU support. Loading the pipeline takes about 16 s. Peak GPU memory allocated by torch was about 0.5 GB per page.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once
make docker-docling                 # this image (no models)
make docling-model                  # opt-in: the pinned models (397 MB) into OCRBENCH_CACHE
make tesseract-best                 # opt-in, Tesseract presets only: tessdata_best (48 MB), shared with engines/tesseract
uv run ocrbench run start --engine engines/docling/engine.toml --dataset datasets/dev-01.json \
    --cache ~/.cache/ocrbench   # or --engine engines/docling/presets/<preset>.toml
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/docling --inspect
make docling-smoke                  # opt-in: build the image, run it on synthetic samples on the GPU
```

`make docling-model` writes the models to `$(OCRBENCH_CACHE)/models/docling/`, the directory layout of Docling's `artifacts_path`. Files that are already present and correct are kept. The adapter checks every file the preset uses against `models.sha256` before loading. The Tesseract presets do not need the RapidOCR files. Their tessdata_best files are checked against `tessdata_best.sha256`. It also checks that the configuration's `model` names exactly the pinned bundle of its OCR engine (`MODELS` in the adapter), so a copied configuration cannot attribute results to other models than the ones that ran. Docling is given only this directory, and `HF_HUB_OFFLINE=1` is set, so inference needs no network.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | Docling's accelerator device for layout and TableFormer: `cuda:<n>` or `cpu` |
| `ocr_engine` | `rapidocr` | Docling's OCR engine, by the `kind` of its options: `rapidocr` (onnxruntime backend, CPU), `tesseract` (the Tesseract command line, `TesseractCliOcrOptions`) or `tesserocr` (the tesserocr bindings, `TesseractOcrOptions`) ([Tesseract](#tesseract)). Further engines come with their presets ([#81](https://github.com/michal-hradis/ocr_benchmarking/issues/81)) |
| `ocr_lang` | `["latin"]` | The OCR languages (Docling's `lang`). RapidOCR accepts only `["latin"]`, PP-OCRv5's Latin-script recognizer, because it is the only recognizer whose model is pinned. The PP-OCRv5 Latin model covers Czech ([Characters](#versions)); other languages would need their own pinned models. With Tesseract, it is a non-empty list of distinct tessdata_best models out of `ces`, `deu`, `deu_latf` and `eng`, joined by `+` in the given order, which Tesseract treats as preference order. Docling hands these codes to Tesseract unchanged. An empty list, with which Docling would let Tesseract detect the script, is rejected |
| `psm` | `null` | Tesseract's page segmentation mode for each crop: 1 or 3–13 (0 and 2 recognize no text). Required with Tesseract and must be absent with RapidOCR |
| `ocr_mode` | `default` | Docling's `OcrMode`: `default`, `layout_regions` or `full_page` ([OCR modes](#ocr-modes)) |
| `ocr_scale` | `3.0` | Docling's OCR render scale. It also sets the nominal resolution (below), so that Docling crops the input pixels 1:1 for OCR |
| `table_structure` | `true` | TableFormer (accurate mode) on table regions |
| `document_timeout` | `null` | Docling's own time limit per page in seconds. When it is reached, Docling returns what it has, and the sample is `partial`. The runner's `timeouts.sample` applies regardless |
| `num_threads` | `4` | Docling's accelerator threads, which also set RapidOCR's onnxruntime threads |

Fixed: Docling's automatic OCR engine selection is not used. For Tesseract's fixed settings, see [Tesseract](#tesseract). RapidOCR runs with Docling 2.145.0's settings for it and RapidOCR 3.10.0's defaults otherwise; `result.json` records the ones that matter (`extra.rapidocr`): `text_score` 0.5 (lines recognized with less confidence are dropped), the resizing below, detection thresholds, batch sizes of 6, and the model files. The adapter checks that the detection, orientation and recognition models RapidOCR loaded are the pinned files and that the recognizer uses its embedded character list; anything else fails every sample, so no fallback model can run. Docling's page batch sizes do not matter, because every sample is one page. Remote services, external plugins, picture classification and description, code, formula and chart enrichment, and form-field extraction are off (Docling's defaults). Docling does not get the sample `language`.

## OCR modes

Docling 2.145.0's standard pipeline runs the layout model first and OCR after it (`StandardPdfPipeline`: preprocessing, layout, OCR, layout post-processing, tables, assembly, reading order). What the OCR engine reads depends on `OcrMode` (`BaseOcrModel.get_ocr_rects`), checked in the installed source:

- **`layout_regions`:** the bounding boxes of all layout clusters, drawn on a page mask in points, dilated by 20 pt (60 input pixels at 216 DPI) and merged into the bounding boxes of the connected areas (`_deduplicate_rects`). RapidOCR reads each merged box as one crop. Clusters less than about 20 pt apart therefore become one crop: on a dense page, nearly the whole page. Text outside every crop is not read.
- **`default`:** for a PDF, the layout regions that have no usable text layer (`pdf_aware_layout_regions`). An image has no text layer, and Docling's image backend reports the whole page as a bitmap, so every cluster qualifies: the same crops as `layout_regions`. The two modes then differ only in how OCR and PDF cells are merged, and an image has no PDF cells.
- **`full_page`:** one crop, the whole page (`force_full_page_ocr`, deprecated upstream, sets this mode). Layout post-processing then assigns the OCR cells to the layout clusters. A cell in no cluster becomes its own `text` cluster (`create_orphan_clusters`, on by default), so text outside the layout regions is kept as one-line regions.

**Resolution.** Docling crops each OCR box from the page rendered at `ocr_scale`, which the nominal DPI makes the input pixels 1:1. RapidOCR then shrinks any crop whose longer side exceeds `max_side_len` = 2000 px to about that size, for both detection and recognition (`Global.use_preprocess_img`), and enlarges a crop whose shorter side is below 30 px; either way it rounds each side to a multiple of 32, so the two sides are resized by slightly different factors. Detection additionally scales its input so that the shorter side is at least 736 px (`Det.limit_type` `min`). The adapter does not change this upstream behavior. It records it as it happens in `result.json` (`extra.ocr_crops`), by wrapping Docling's `get_ocr_rects` and RapidOCR's `preprocess_img` on the loaded pipeline without changing their results:

- `crops`: every box Docling asked to OCR, in input pixels. A box of zero area is `skipped`, as Docling never reads it. Every other box has RapidOCR's input size (`crop`), the size after preprocessing (`resized`) and the factors per side (`resize`, below 1 when shrunk).
- `boxes`, `skipped`, `rapidocr_calls`, `shrunk`, `enlarged` and `min_resize` (the smallest factor) summarize them. With orientation detection off (Docling's default), RapidOCR is called once per box that is read, in order (`paired`). Otherwise the calls are listed apart in `calls`.

On dev-01, the layout modes read each newspaper as one or two crops that cover nearly the whole page, shrunk to 0.35 and 0.40. The larger crop of two DTA pages is shrunk to 0.75 and 0.92. In `full_page` mode the page is the crop: 0.32 and 0.38 for the newspapers and 0.70–0.91 for the DTA pages; the 1878-pixel Glagolitic page is not resized.

## Tesseract

Docling 2.145.0 has two Tesseract backends ([#79](https://github.com/michal-hradis/ocr_benchmarking/issues/79)). Both drive the same Tesseract 5.3.4 with the same models but use it differently. Their behavior below was checked in the installed source (`tesseract_ocr_cli_model.py`, `tesseract_ocr_model.py`):

- **Command line** (`ocr_engine = "tesseract"`): Docling saves each OCR crop as a PNG at the nominal DPI and runs `tesseract <crop> stdout tsv -l <lang> --tessdata-dir <dir> --psm <psm>`. It passes no `--oem`: Tesseract's default selects LSTM for these models. It keeps the TSV rows with text, which are **words**, and makes one OCR cell of each.
- **Bindings** (`ocr_engine = "tesserocr"`): Docling sets the crop on a `PyTessBaseAPI` (OEM default, the same languages and PSM) and asks for its **text lines** (`GetComponentImages(TEXTLINE)`). It then **recognizes each line again on its own**, by setting the line's rectangle and reading its text and mean confidence. Each OCR cell is one line. Re-reading the lines one by one is what makes this backend different from the command line, beyond the cell granularity.
- **Both** run Tesseract's orientation and script detection (OSD, `osd` model) on every crop before OCR and turn the crop when OSD reports an orientation other than 0. Docling maps the positions back to the page. When OSD fails (for example "Too few characters" on a small crop), Docling logs an error and reads the crop unturned. Docling has no option to switch OSD off, and the adapter does not change this.
- **Tessdata.** The adapter creates a tessdata directory for each run with links to exactly the preset's verified tessdata_best models, the image's `osd` model and Tesseract's `configs`. It passes this directory as Docling's `path` and also sets `TESSDATA_PREFIX` to it, because Docling lists the languages and runs the command line's OSD without `--tessdata-dir`. Before the first sample, the adapter checks that Docling's model runs Tesseract 5.3.4 (and tesserocr 2.12.0) with exactly the preset's languages, PSM and directory and without automatic script detection. Otherwise every sample fails.
- **Resolution.** Tesseract reads the crops as Docling cuts them, 1:1 input pixels at the nominal 216 DPI ([Image input](#image-input-and-coordinates)), without resizing them.
- **Recorded** in `result.json`: `extra.tesseract` (Tesseract and tesserocr versions, data directory, languages, PSM, OEM, the languages Tesseract lists) and, in `extra.ocr_crops`, each crop's box with the image Tesseract read (turned if `osd_orientation` is not 0, `null` when OSD failed), its number of OCR cells and whether the call failed. The summary counts are `tesseract_calls`, `failed`, `osd_failed` and `rotated`.
- **Lines from words.** The command line's word cells are grouped into Tesseract's own lines. The adapter records the TSV rows of every call (`tesseract.json`) by wrapping Docling's `_run_tesseract` without changing its result. Docling numbers the page's cells in the order of these rows. The adapter checks that every cell is the recorded word of its number; otherwise the sample fails. Within a region, the words of one Tesseract line (crop, block, paragraph and line number) become one prediction line: the words joined by single spaces in Docling's cell order, the bounding box of the words' boxes, their mean confidence, and in `extra` the number of words and the Tesseract line. When Docling assigns the words of one Tesseract line to two layout regions, each region gets its part. The word boxes stay in `docling_layout.json`. The bindings already give lines, and each cell is a line as with RapidOCR.
- **A failed Tesseract call** (non-zero exit) is logged by Docling, which goes on without that crop's text. The adapter counts it, and the sample is `partial` with a message, not a silent success.

## Presets

A preset is a frozen engine configuration; its `name` identifies it as a separate system in reports. All presets use the same image, layout and TableFormer models and settings except where the table says otherwise.

| Preset (file) | `name` | Settings |
| --- | --- | --- |
| `docling` (`engine.toml`) | `docling` | Docling's standard pipeline at its defaults: Heron layout, RapidOCR PP-OCRv5 Latin in the default OCR mode (for an image, the layout regions), TableFormer accurate, the rule-based reading order |
| `rapidocr-regions` (`presets/rapidocr-regions.toml`) | `docling-rapidocr-regions` | `ocr_mode = "layout_regions"`: RapidOCR on the layout regions, named explicitly instead of through Docling's PDF-aware default ([#78](https://github.com/michal-hradis/ocr_benchmarking/issues/78)) |
| `rapidocr-fullpage` (`presets/rapidocr-fullpage.toml`) | `docling-rapidocr-fullpage` | `ocr_mode = "full_page"`: RapidOCR on the whole page, cells assigned to the layout regions ([#78](https://github.com/michal-hradis/ocr_benchmarking/issues/78)) |
| `tesseract-cli` (`presets/tesseract-cli.toml`) | `docling-tesseract-cli` | `ocr_engine = "tesseract"`: the Tesseract command line on the layout regions (`ocr_mode = "layout_regions"`) with tessdata_best `ces+deu+eng` and `psm = 3`, instead of RapidOCR ([#79](https://github.com/michal-hradis/ocr_benchmarking/issues/79)). It differs from `docling-rapidocr-regions` only in the OCR engine, its models and its settings. Its languages, models and PSM are those of the standalone `tesseract-best-psm3` |
| `tesseract-cli-deu-latf` (`presets/tesseract-cli-deu-latf.toml`) | `docling-tesseract-cli-deu-latf` | `ocr_lang = ["deu_latf"]`: German Fraktur for historical German prints. Otherwise `docling-tesseract-cli` (compare the standalone `tesseract-best-deu-latf-psm3`) |
| `tesserocr` (`presets/tesserocr.toml`) | `docling-tesserocr` | `ocr_engine = "tesserocr"`: the bindings backend with the same Tesseract, models and settings as `docling-tesseract-cli` |

A new preset is a new TOML file with its own `name`; a changed setting is a new preset, not an edit of an existing one. Tesseract's sample `language` is not used: the languages are part of the frozen configuration, and the Fraktur preset is run on every page like the others, not chosen per page. Layout and OCR-scope variants are [#80](https://github.com/michal-hradis/ocr_benchmarking/issues/80), and the DocTags VLMs are [#82](https://github.com/michal-hradis/ocr_benchmarking/issues/82)–[#84](https://github.com/michal-hradis/ocr_benchmarking/issues/84).

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
| `tesseract.json` | Tesseract command line only: for every Tesseract call, the image it read, the OSD orientation and the TSV rows it returned as Docling read them (words with text; all TSV columns, unchanged) |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output, converted from the two JSON files above |
| `result.json` | Status, processing time and, in `extra`: Docling's conversion status and errors, the warnings Docling and RapidOCR logged (up to 50), the nominal DPI, peak GPU memory, region/line/cell counts, the versions, GPU and load time, the checksums of the model files (`models.files`, and `models.tessdata_best` for Tesseract), the parameters, Docling's full pipeline options, the OCR crops and RapidOCR's or Tesseract's settings in effect (`rapidocr` or `tesseract`) |

**Mapping** (decided in [#77](https://github.com/michal-hradis/ocr_benchmarking/issues/77#issuecomment-6532444998)):

- **Regions:** the items of the DoclingDocument are read in document order. This is a pre-order walk from the body, as `DoclingDocument.iterate_items` does with all content layers and with the children of pictures. Docling's reading-order model builds this order. Captions are children of their table or picture and follow it. Page headers and footers keep their place. Text that Docling recognized inside a picture is kept: these are child items of the picture that the Markdown export leaves out. Every provenance box of an item becomes one region `r<n>`, with the box as its polygon and the item's label (for example `section_header`) as `native_type`. An item that Docling merged from several layout clusters (a paragraph continued in another column) therefore becomes several regions, in order. The region `type` follows the label: titles and section headers are `heading`; text, paragraphs and references are `text`; then `list-item`, `caption`, `footnote`, `page-header` and `page-footer`; tables and indexes are `table`; formulas are `formula`; pictures and charts are `figure`; anything else is `other`. `extra` holds the item's `self_ref`, the provenance index, the cluster ID and the layout confidence.
- **Lines:** every provenance box is matched to the layout cluster it came from. The box and the cluster bounding box are the same rectangle, so they are matched to within 0.001 pt. If several clusters match, the one with the item's label is preferred. The region's lines are that cluster's OCR cells with non-blank text, in Docling's cell order. Each line keeps its raw OCR text, its 4-point box and its confidence. Its ID is `l<cell index>`. Cells that belong to a child cluster that is itself an item (text inside a picture) go only to that child's region. An OCR cell is one RapidOCR text box (usually a whole printed line, sometimes part of one) or one line of the tesserocr bindings, and the adapter does not merge these cells. The command line's word cells are grouped into Tesseract's lines ([Tesseract](#tesseract)). A table's lines are its cells' OCR text, not TableFormer's structure, which stays in `docling.json`. Regions without cells, such as pictures, have no text.
- **Text:** the page text is the lines joined by line breaks, with regions separated by blank lines. Docling's own item text stays in the native files and is not scored. Docling builds it from the same cells but joins the lines with spaces, removes a hyphen at a line end and replaces `’ ‘ “ ”` with ASCII quotes.
- **Cells outside items:** non-blank OCR cells in clusters that no DoclingDocument item uses are left out, not invented into the prediction. Their number is `counts.cells_outside_regions` in `result.json`, and a non-zero number is reported in the sample's `message`, for example "3 of 120 non-blank OCR cells are in no DoclingDocument item and are left out". The cells stay in `docling_layout.json`.
- **Checks:** a box that matches no unused cluster, a cell that would go to two regions, a document that is not exactly one page, and an unknown coordinate origin all fail the sample. These indicate a Docling result that the mapping does not understand.

**Status.** An image that cannot be decoded, a conversion that Docling reports as failed, or an error such as CUDA out of memory fails only that sample (`status: failed`, runner `engine-error`), and the container continues. A partial conversion (for example after `document_timeout`) or a failed Tesseract call keeps its outputs and is `partial`. An unusable GPU, missing or changed models or tessdata, another Tesseract than the pinned one, a `model` other than the pinned bundle, or invalid parameters fail every sample. A page on which nothing is detected is a **genuine empty result**: `success` with empty text and no regions.

## Measured on dev-01

All presets on the seven pages of dev-01, RTX 4090, default evaluation configuration (`strict`, empty-output failure policy). Runs are not in Git. The `*-dev01-v2` runs used image `-v2` (`sha256:ae4b0a7c…`). All 7 samples of every run were `success`. Rates are micro averages over the pages. fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)); the evaluator marks them as not proven optimal on most pages. Engine time is the sum of the adapter's per-sample times. Loading the pipeline took another 15–17 s.

**RapidOCR presets** (#77, #78; first runs `docling-dev01`, `docling-rapidocr-regions-dev01b` and `docling-rapidocr-fullpage-dev01b` on image `-v1`). Rerun on `-v2` (`docling-dev01-v2`, `docling-rapidocr-regions-dev01-v2`, `docling-rapidocr-fullpage-dev01-v2`), their predictions are byte-identical to the `-v1` runs on all seven pages, so the scores below hold for both images.

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
- **Region OCR reads the newspapers almost as full pages.** Docling merges the layout clusters of a newspaper into one or two crops that cover nearly the whole page, and RapidOCR shrinks them to 0.35 and 0.40 of the input resolution (the smaller of the two factors per side). The full-page preset shrinks them to 0.32 and 0.38. Both lose text: the predictions are 21–28 % shorter than the reference in region mode and 23–24 % in full-page mode, and every OCR cell is in a region (`cells_outside_regions` is 0 in all runs). That the shrinking causes these deletions is likely but was not tested (speculation); `max_side_len` is RapidOCR's default and no preset changes it.
- **Full-page OCR changes the newspapers in both directions.** fCER is 6.3 points worse on `3e6309d0…` (25 fewer cells) and 3.5 points better on `bd497a62…` (17 more cells). On the DTA pages it returns up to 5 fewer cells, and CER changes by less than 1 point. The order of the newspaper paragraphs is Docling's in both modes: strict CER is about twice fCER.
- **The Latin recognizer has no Glagolitic**, and the Fraktur pages are read with a Latin Antiqua model without `ſ` ([Characters](#versions)).

**Tesseract presets** (#79; runs `docling-tesseract-cli-dev01-v2`, `docling-tesseract-cli-deu-latf-dev01-v2` and `docling-tesserocr-dev01-v2`). The standalone [Tesseract presets](../tesseract/README.md#presets) with the same models, languages and PSM (runs `tesseract-best-psm3-dev01` and `tesseract-best-deu-latf-psm3-dev01`, single-threaded) are given for comparison; they read each page as a whole.

| Preset | CER | WER | fCER | fWER | BoW F1 | ROUGE-2 F1 | Engine time (7 pages) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `docling-tesseract-cli` | 41.4 % | 55.1 % | 6.0 % | 19.3 % | 83.9 | 73.3 | 170 s |
| `docling-tesseract-cli-deu-latf` | 49.8 % | 81.2 % | 19.8 % | 63.8 % | 40.4 | 18.5 | 110 s |
| `docling-tesserocr` | 44.5 % | 63.0 % | 31.4 % | 48.9 % | 77.1 | 63.9 | 391 s |
| standalone `tesseract-best-psm3` | 48.2 % | 62.3 % | 27.4 % | 41.1 % | 81.8 | 69.2 | 239 s |
| standalone `tesseract-best-deu-latf-psm3` | 54.6 % | 84.0 % | 35.9 % | 74.7 % | 39.6 | 18.2 | 141 s |

CER per page in % (fCER in parentheses where it differs). Lines are the prediction lines (for the command line, Tesseract's lines grouped from the word cells) against the reference lines.

| Page | Content | Lines (CLI / CLI `deu_latf` / tesserocr / GT) | `docling-tesseract-cli` | `docling-tesseract-cli-deu-latf` | `docling-tesserocr` | Standalone `best-psm3` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 479 / 480 / 352 / 419 | 72.1 (7.2) | 79.2 (24.5) | 87.9 (52.2) | 83.6 (62.8) |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 516 / 511 / 483 / 484 | 25.9 (2.3) | 38.2 (17.5) | 18.9 | 32.9 (5.9) |
| `CkXII61-0017` | Glagolitic print | 20 / 20 / 18 / 19 | 84.6 | 93.3 | 87.3 | 81.0 |
| `dta-346672…` | Latin, Antiqua and italic | 35 / 34 / 30 / 34 | 50.3 (20.3) | 44.3 (25.0) | 59.1 (37.8) | 21.6 (16.0) |
| `dta-laube…` | German Fraktur, 1833 | 27 / 27 / 21 / 25 | 8.1 | 5.0 | 16.4 | 11.3 |
| `dta-martzi…` | German Fraktur, 1692 | 39 / 43 / 26 / 29 | 26.2 | 21.3 | 42.2 | 30.0 |
| `dta-steinen…` | German Antiqua, 1894 | 43 / 43 / 43 / 44 | 1.5 | 2.4 | 4.8 | 1.3 |

Observations (measured, unless marked):

- **Docling's layout regions help Tesseract on the newspapers.** With the same models, languages and PSM as the standalone `tesseract-best-psm3`, the command line inside Docling reaches fCER 7.2 % and 2.3 % on the two newspapers against 62.8 % and 5.9 %, and 6.0 % overall against 27.4 %. Tesseract reads one or two crops per newspaper here too, as Docling merges the dense layout clusters (as with RapidOCR). That Docling's crop boundaries explain the difference, rather than the newer Tesseract (5.3.4 against 5.3.0) or the nominal 216 DPI, is likely but was not tested (speculation). Strict CER stays high on `3e6309d0…` (72.1 %), because Docling's reading order across the columns differs from the reference.
- **Docling's reading order and crops hurt the Latin page** `dta-346672…`: CER 50.3 % against the standalone 21.6 %, with fCER 20.3 % against 16.0 %.
- **The bindings are worse than the command line** on every page except `bd497a62…` (18.9 % against 25.9 % strict, but 18.9 % against 2.3 % fCER) and are 2.3 times slower. They return fewer lines (352 against 479 on `3e6309d0…`, 26 against 39 on `dta-martzi…`). That re-reading each line rectangle on its own ([Tesseract](#tesseract)) loses or merges text is likely, but the cause was not isolated (speculation).
- **`deu_latf` helps on the Fraktur pages** (5.0 % and 21.3 % against 8.1 % and 26.2 %), as it does standalone, and hurts everything else, with BoW F1 40.4. This matches the standalone presets.
- **OSD:** on 5 of the 7 pages, Docling's orientation detection failed on 1–2 crops (logged in `warnings`, `osd_orientation` null); those crops were read unturned. On the Glagolitic page, OSD turned one of its three crops by 180°. No Tesseract call failed.
- **Speed:** Tesseract 5.3.4 from Ubuntu uses OpenMP threads, up to the container's 4 CPUs, while the standalone presets run single-threaded. Engine times are therefore not comparable across the two engines.
- The development set is small and not representative; these numbers are not a ranking of Docling, its OCR engines or its OCR modes.

## Tests

`tests/test_docling.py` checks the mapping on committed native results, all offline in `make test`. The fixtures are in `tests/fixtures/docling/<name>/` (`docling.json` and `docling_layout.json`). They are real Docling 2.145.0 output of this preset for synthetic pages rendered by `tests/synthetic.py`. The real output of `docling-tesseract-cli` (with `tesseract.json`) and `docling-tesserocr` for `page`, `paragraph`, `newspaper`, `table` and `blank` is in `tests/fixtures/docling/tesseract-cli/<name>/` and `tests/fixtures/docling/tesserocr/<name>/`:

- `page`
- `paragraph`: a crop that Docling's layout takes for a picture with text inside
- `columns`
- `newspaper`: three columns
- `table`: a ruled table, a picture and a caption
- `blank`

The tests check the regions in document order with their OCR cells, the page text and line positions against the rendered page, the provenance boxes in input pixels, tables and pictures, text inside a picture read once, Docling's newspaper order and a blank page. On small hand-made results, they also check raw cells instead of Docling's joined text, both coordinate origins, merged items, label preference, blank cells and rejected inconsistent results. Further offline tests cover the page-size check, the TIFF input (stored pixels, no EXIF, nominal DPI), parameters, model sources and `fetch_model.py`. The engine protocol is tested through the runner with fake `torch` and `docling` modules (`tests/fixtures/docling/fake`), including a CUDA error, a failed and a partial conversion, a turned page and OCR cells outside every item between successful samples, a `model` that names other models, the CPU device, the OCR mode of each preset and setup errors. Further tests check that the presets differ from `docling` only in name and OCR mode, the recorded OCR crops (in input pixels, both origins, a skipped zero-area box, RapidOCR's actual sizes when shrinking and enlarging, calls that do not pair up with the boxes) and that RapidOCR settings with any model other than the pinned files are rejected.

For Tesseract, the tests check:

- On the committed Tesseract results: the regions in document order, the command line's words grouped into Tesseract's lines (the page text and line boxes against the rendered page; a line box is the union of its word boxes), and a Tesseract line split between two regions. A TSV that does not match the cells is rejected.
- The presets: each differs from the preset it is compared with in one thing only.
- The tessdata: it is the standalone Tesseract engine's, and the adapter's tessdata directory holds only the preset's models, `osd` and `configs`. Missing or changed models are rejected.
- The Tesseract settings check, and the recorded crops of both backends, including OSD orientation, a failed OSD and a failed call.
- The engine protocol through the runner with a fake `tesserocr`, for both backends. This covers a failed Tesseract call (`partial`), `tesseract.json`, the tessdata checksums, and setup errors for another Tesseract version, a changed model and a wrong `model`.

`make docling-smoke` builds the image and runs real Docling, with each of the three presets, through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report`. The samples are the synthetic full page, a paragraph crop, a blank page (an empty `success`), the two-column, newspaper and table pages, and the full page saved as a 300 DPI JPEG with EXIF orientation 6. Ordered CER is below 10 % on all except the newspaper, where fCER is below 10 %, and the JPEG's positions match the stored pixels. Each run must record the OCR mode, RapidOCR's settings and its crops (one per page in full-page mode, none shrunk on these small pages). A run with a GPU that is not visible must fail explicitly. The three Tesseract presets run on the same samples, which needs `make tesseract-best`. Every sample must be `success`, with one Tesseract call per crop, no failed call and no turned crop. The command line must write `tesseract.json` and the bindings must not. The command line's lines must have several words. Ordered CER must be below 10 % on the page, paragraph, two-column and JPEG pages and below 25 % on the table page, where the bindings read each table line again on its own. fCER must be below 10 % on the newspaper.
