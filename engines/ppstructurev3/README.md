# PP-StructureV3 engine (PaddleOCR document layout pipeline, GPU)

Page-image layout pipeline ([#45](https://github.com/michal-hradis/ocr_benchmarking/issues/45)): PaddleOCR's PP-StructureV3, which detects layout blocks and layout regions, orders the blocks for reading and recognizes the text lines of the page. It is a separate engine from the [PaddleOCR PP-OCRv6 adapter](../paddleocr/README.md), which detects and recognizes lines without any layout. It runs in a GPU Docker image on the shared Paddle base ([DOCKER.md](../../docs/DOCKER.md)) and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol). The input is always the full page image; PDFs are not used.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-ppstructurev3:3.7.0-v1` from `ocrbench/base-cu130-paddle:paddle3.4.0-py3.12-v1` |
| `requirements.in`, `requirements.txt` | PaddleOCR with the PaddleX `ocr` extra that PP-StructureV3 requires, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the model files; the build fails if they differ |
| `ppstructurev3_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`, preset `latin` |
| `presets/ocrv6.toml` | Preset `ocrv6`: the same pipeline with the PP-OCRv6 models of the `paddleocr` engine ([Presets](#presets)) |

## Versions

- PaddleOCR **3.7.0** and PaddleX **3.7.2** with the `ocr` extra, on PaddlePaddle **3.4.0** for CUDA 13.0 (the `base-cu130-paddle` family). It needs an NVIDIA driver for CUDA 13.0 (≥ 580) and has been run on an RTX 4090. The pipeline configuration is PaddleX's `PP-StructureV3.yaml` of that version, changed only by the arguments under [Parameters](#parameters).
- Models from Hugging Face `PaddlePaddle/<name>` at pinned commits, all Apache-2.0, 503 MB in total:

| Role | Model | Commit | Used by |
| --- | --- | --- | --- |
| Layout detection | [`PP-DocLayout_plus-L`](https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L) | `aa52b8528c84f9b1a34ac3a88fe0e576edb9d11d` | both presets |
| Layout region detection | [`PP-DocBlockLayout`](https://huggingface.co/PaddlePaddle/PP-DocBlockLayout) | `f270657da59d956ab69ffc2c4722fe1f557bf17b` | both presets |
| Text line orientation | [`PP-LCNet_x1_0_textline_ori`](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori) | `cd237a44b0e359d4fe38310a416203cf7403faa5` | both presets |
| Text detection | [`PP-OCRv5_server_det`](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det) | `ca867c897ecbca8873081573a802ad70d499cb94` | `latin` |
| Text recognition | [`latin_PP-OCRv5_mobile_rec`](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec) | `ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57` | `latin` |
| Text detection | [`PP-OCRv6_medium_det`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det) | `8e0f56fb2ef86b461d99cfc7ac5c137738985f61` | `ocrv6` |
| Text recognition | [`PP-OCRv6_medium_rec`](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec) | `e5a92bcbc5cc1b494628e458d267778f0704fd7c` | `ocrv6` |

- The models are downloaded **when the image is built** and checked against `models.sha256`, so inference needs no `--cache` and no network. The table, formula, seal, chart, document orientation and unwarping models of the full PP-StructureV3 are not in the image (see [Parameters](#parameters)).

The Ubuntu packages (`libgomp1` in the base, `libgl1` and `libglib2.0-0t64` for OpenCV here) are pinned by version, not by an archive snapshot. If Ubuntu removes a version, the build fails rather than silently changing it; then bump the pin and the image tag (`-v2`) and `engine_version`.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-paddle            # shared Paddle base, once (about 6 GB)
make docker-ppstructurev3           # this image (PaddleOCR, PaddleX and the models)
uv run ocrbench run start --engine engines/ppstructurev3/engine.toml --dataset datasets/dev-01.json
uv run ocrbench run start --engine engines/ppstructurev3/presets/ocrv6.toml --dataset datasets/dev-01.json
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/ppstructurev3 --inspect
make ppstructurev3-smoke            # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. The image is a CUDA build: without a visible GPU every sample fails with `engine-error` and the reason; there is no CPU fallback (`PADDLE_PDX_DISABLE_DEVICE_FALLBACK`).

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the PP-StructureV3 defaults, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `gpu:0` | Paddle device, `gpu:<n>` |
| `ocr_models` | `PP-OCRv5-latin` | Text detection and recognition models: `PP-OCRv5-latin` (`PP-OCRv5_server_det`, `latin_PP-OCRv5_mobile_rec`) or `PP-OCRv6-medium` (`PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec`) |
| `layout_threshold` | per class, 0.3–0.5 | Layout detection score threshold for all classes, `0 < value <= 1` |
| `text_det_limit_side_len`, `text_det_limit_type` | 736, `min` | Detection input scaling: the shorter (`min`) or longer (`max`) side is scaled to at least / at most this many pixels; PaddleX then caps the longer side at 4000 pixels |
| `text_det_thresh`, `text_det_box_thresh` | 0.3, 0.6 | DB detector pixel threshold and box score threshold, `0 < value <= 1` |
| `text_det_unclip_ratio` | 1.5 | DB box expansion ratio, a finite number `> 0` |
| `text_rec_score_thresh` | 0.0 | Lines with a lower recognition score are dropped, `0 <= value <= 1` |
| `text_recognition_batch_size` | 8 | Lines recognized per batch |

Fixed: layout region detection (`PP-DocBlockLayout`, which groups blocks into regions for the reading order) and the text line orientation classifier are **on**, as in the PP-StructureV3 defaults. Document orientation classification and unwarping are **off**, because they would transform the image so that positions no longer refer to the input. Table, formula, seal and chart recognition are **off** and their models are not in the image: the benchmark scores page text, and the full-page OCR already covers the text in those areas (see Mapping). The sample `language` is not used.

## Presets

A preset is a frozen engine configuration; its `name` identifies it as a separate system in reports. Both presets use the same image, layout models, pipeline settings and resources, and differ **only in `ocr_models`**.

| Preset (file) | `name` | `ocr_models` | Purpose |
| --- | --- | --- | --- |
| `latin` (`engine.toml`) | `ppstructurev3` | `PP-OCRv5-latin` | The OCR models that PaddleOCR 3.7 itself selects for PP-StructureV3 with a Latin-script language (`PPStructureV3(lang="cs")` or `"de"`). The PP-StructureV3 default without a language, `PP-OCRv5_server_rec`, recognizes only Chinese, English and Japanese characters and drops diacritics and long s, so it is not used. |
| `ocrv6` (`presets/ocrv6.toml`) | `ppstructurev3-ocrv6` | `PP-OCRv6-medium` | The detection and recognition models of the [`paddleocr` engine](../paddleocr/README.md), so that the two engines differ in the layout analysis (and PP-StructureV3's detection settings) only. PaddleOCR 3.7 does not offer PP-OCRv6 through `PPStructureV3(ocr_version=…)`, so the models are given by name and directory. |

A new preset is a new TOML file with its own `name`; a changed setting is a new preset, not an edit of an existing one.

## Outputs

For every sample the adapter decodes the image as stored (EXIF orientation not applied, as in [PREDICTIONS.md](../../docs/PREDICTIONS.md#positions)), runs the pipeline and writes into the sample directory:

| File | Content |
| --- | --- |
| `ppstructurev3.json` | PP-StructureV3's native result (`res.json`: the blocks of `parsing_res_list` with their labels, boxes and own text, the layout detections and the overall OCR result with every line), unchanged |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output |
| `result.json` | Status, processing time and, in `extra`, the model settings, the effective detection settings, the number of blocks, the line counts (below), the PaddleOCR/PaddleX/Paddle/CUDA/cuDNN versions, the GPU name, the model loading time, the models and the parameters |

**Mapping.** Every block of `parsing_res_list` becomes a region `b1`, `b2`, … in that order, which is PP-StructureV3's reading order (`reading_order` lists them explicitly). Its box is the region polygon (pixels of the input image), its label the `native_type` (for example `paragraph_title`), and the region `type` follows the label: titles `heading`, `text`/`abstract`/`content`/`reference`/`algorithm`/`aside_text` `text`, the figure, table and chart titles `caption`, `header`/`footer`/`number` `page-header`/`page-footer`/`page-number`, `footnote`, `table`, `formula`, images, charts and seals `figure`, anything else `other`. `extra` holds the native `block_id` and `block_order`. Blocks without lines, such as a picture, remain regions with a polygon and no text.

**Lines.** The native result does not record which OCR lines a block was built from, so the adapter assigns them: every line of the overall OCR result (`overall_ocr_res`: `rec_texts`, `rec_polys`, `rec_scores`, `rec_boxes`) with non-blank text goes into the block whose box its box overlaps by more than 3 pixels on both axes (the rule by which PaddleX matches OCR boxes to blocks) with the largest intersection, ties to the earlier block. A line keeps its detected polygon and its recognition score (0–1) as confidence, and its ID `l<n>` is its position in `overall_ocr_res`. Within a block, lines are ordered top to bottom, and boxes that share at least 80 % of the smaller height form one visual line, read left to right, as PP-StructureV3 orders a horizontal block. The region and page text come from these lines; PP-StructureV3's own `block_content`, which joins the lines into paragraphs and removes hyphens at line ends, stays in the native file. Lines outside every block are left out, as they are left out of PP-StructureV3's blocks; their number is reported as `extra.lines.outside_blocks` in `result.json`. With table recognition off, PaddleX leaves a table block's own text empty; the adapter gives it the OCR lines inside it like any other block.

**Status.** An image that cannot be decoded or an error in the pipeline fails only that sample (`status: failed`, runner `engine-error`) and the container continues; for Paddle's CUDA out-of-memory errors the message keeps only Paddle's summary. A result for an image of another size, a result whose OCR arrays differ in length and a result from a document preprocessor that ran with orientation classification or unwarping on fail the sample (PaddleOCR 3.7 reports the preprocessor as used with both steps off; that is accepted). An unusable GPU or invalid parameters fail every sample. A page on which nothing is detected is a **genuine empty result**: `success` with empty text and no regions.

## Measured on dev-01

Both presets on the seven pages of dev-01 (runs `ppstructurev3-latin-dev01` and `ppstructurev3-ocrv6-dev01`, not in Git; image `sha256:54dad966…`, whose adapter differs from the committed one only in a code comment; RTX 4090), next to the `paddleocr` engine's `det4000` preset (run `paddleocr-det4000-dev01`). All 14 PP-StructureV3 samples were `success`. Rates are micro averages over the pages; fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)). Engine time is the sum of the adapter's per-sample times; loading the models took another 14 s per run. GPU memory was not measured.

| Engine, preset | CER | WER | fCER | fWER | BoW F1 | Engine time (7 pages) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppstructurev3` (`latin`) | 47.7 % | 61.6 % | 7.5 % | 21.6 % | 83.0 | 10 s |
| `ppstructurev3-ocrv6` | 47.9 % | 62.6 % | 7.4 % | 20.6 % | 83.8 | 17 s |
| `paddleocr` (`det4000`) | 72.0 % | 89.2 % | 12.1 % | 25.1 % | 85.2 | 11 s |

CER per page in % (fCER in parentheses where it differs).

| Page | Content | Blocks | `latin` | `ocrv6` | `paddleocr` |
| --- | --- | ---: | ---: | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 27 | 64.6 (6.3) | 63.6 (5.2) | 82.2 (12.4) |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 44 | 43.2 (3.4) | 43.5 (3.3) | 81.3 (9.8) |
| `CkXII61-0017` | Glagolitic print | 4 | 91.6 | 83.7 | 82.3 |
| `dta-346672…` | Latin, Antiqua and italic | 5 | 5.6 | 12.5 | 12.0 |
| `dta-laube…` | German Fraktur, 1833 | 8 | 4.2 | 4.0 | 4.0 |
| `dta-martzi…` | German Fraktur, 1692 | 6 | 8.7 | 16.1 | 13.1 |
| `dta-steinen…` | German Antiqua, 1894 | 7 | 30.6 | 31.2 | 17.5 |

Observations:

- **The layout reading order helps the newspapers:** strict CER falls from 81–82 % to 43–65 %, as columns are no longer interleaved line by line. The order of the 27–44 blocks still differs from the reference in places, so the strict rates stay high.
- **The flexible rates improve as well** on the newspapers, from 9.8–12.4 % to 3.3–6.3 % fCER with either preset, although `ocrv6` uses the same recognizer as `paddleocr`. The differences are in the line boxes the recognizer gets (PP-StructureV3's detection settings, its splitting at block boundaries) and in the prediction units; they were not analysed further.
- **Lines can be lost at block boundaries.** On `dta-steinen…`, a page with curved lines, PP-StructureV3 splits OCR boxes that reach into two blocks and re-recognizes the parts, and several lines next to these boxes disappear from its overall OCR result and from its own `block_content` ("Der Grund kann nur der sein …"); the adapter faithfully has them missing too, and CER rises to 31 % against 17.5 % without layout. On the newspapers 1 and 8 (`latin`) or 0 and 6 (`ocrv6`) of 423–520 lines lie outside every block.
- **The Latin PP-OCRv5 recognizer is better on most pages** than PP-OCRv6 medium inside the same pipeline (`dta-346672…`, `dta-martzi…`) and faster; PP-OCRv6 is better on the Glagolitic page, which neither covers.

## Tests

`tests/test_ppstructurev3.py` checks the mapping on committed native results (`tests/fixtures/ppstructurev3/`: real PP-StructureV3 output of preset `latin` for synthetic pages rendered by `tests/synthetic.py`: `page.json`, a two-column page `columns.json`, a three-column newspaper page `newspaper.json`, whose overall OCR lines interleave the columns, and the blank page `blank.json`). The tests check the page text in the rendered reading order, the line polygons against the rendered lines in input-image pixels, the region types, the line assignment (largest overlap, the 3-pixel rule, ties, lines outside blocks), the line order within a block, empty pages, rejected results (preprocessing, size, inconsistent arrays), parameter validation, the pipeline arguments and model checksums, and the engine protocol through the runner with a fake `paddleocr` (`tests/fixtures/ppstructurev3/fake`) and the fake `paddle`, `paddlex` and `cv2` modules of `tests/fixtures/paddleocr/fake`, including a pipeline error and a simulated CUDA out-of-memory error between successful samples, all offline in `make test`. `make ppstructurev3-smoke` builds the image and runs real PP-StructureV3 (preset `latin`) on the GPU through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on the synthetic full page, a paragraph crop, a blank page (an empty `success`) and the two-column and newspaper pages (ordered CER < 10 % on all, so the columns are read one after another), plus a run with a GPU that is not visible, which must fail explicitly. The `ocrv6` preset was run by hand on the synthetic pages and on dev-01 (above); it is not part of the smoke test.
