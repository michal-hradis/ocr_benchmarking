# Kraken engine (historical OCR/HTR, GPU)

Historical OCR/HTR pipeline baseline ([#13](https://github.com/michal-hradis/ocr_benchmarking/issues/13)): [Kraken](https://kraken.re) baseline layout analysis (regions and text line baselines with boundary polygons) followed by line recognition, in a GPU Docker image on the shared torch base ([DOCKER.md](../../docs/DOCKER.md)), following the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol).

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-kraken:7.1.1-v1` from `ocrbench/base-cu130-torch:torch2.14.1-py3.12-v1` |
| `requirements.in`, `overrides.txt`, `requirements.txt` | Kraken and its dependencies, hash-locked against the base (`make docker-lock`) |
| `models.sha256` | Checksums of the recognition models; the build fails if they differ |
| `kraken_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start` |

## Versions

- Kraken **7.1.1** (Apache-2.0) on the base's torch **2.14.1** for CUDA 13.0, with torchvision 0.29.1. Kraken 7.1.1 declares `torch<=2.14`, which by version ordering excludes the 2.14.1 patch release of the base; `overrides.txt` overrides that bound when locking instead of adding a torch family. The adapter has been run with it on an RTX 4090 (CUDA 13.0 needs an NVIDIA driver ≥ 580).
- **Segmentation:** Kraken's default baseline and region model `blla.mlmodel` (Apache-2.0), shipped inside the hash-locked `kraken` wheel; its SHA-256 is recorded for every sample. Its only region class is `text`, and it has no neural reading order model.
- **Recognition models in the image**, from Zenodo; CATMuS models are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (attribution: cite the records below when publishing results), `german_print` is CC0:

  | `recognition_model` | Record | Use |
  | --- | --- | --- |
  | `catmus-print-fondue-large.mlmodel` (default, 23 MB) | [CATMuS-Print (Large)](https://doi.org/10.5281/zenodo.10592716), S. Gabay, 2024-01-30 | Printed books and newspapers, 16th–21st c., French, Spanish, German, English, Latin, Italian and other Latin-script languages |
  | `McCATMuS_nfd_nofix_V1.mlmodel` (16 MB) | [McCATMuS](https://doi.org/10.5281/zenodo.13788177), 2024-09-19 | Handwritten, printed and typewritten documents, 16th–21st c., the same languages; for handwriting crops |
| `german_print.mlmodel` (16 MB) | [german_print](https://doi.org/10.5281/zenodo.10519596), UB Mannheim, 2023-05-16 | German and Latin prints from incunabula to the 20th c., Fraktur (`Latf`); for German Fraktur pages |

  They are downloaded **when the image is built** and checked against `models.sha256` (the "small weights" case of the image conventions), so inference needs no `--cache` and no network.

**Transcription conventions.** The CATMuS models follow the [CATMuS guidelines](https://catmus-guidelines.github.io/): graphematic transcription that keeps the long s `ſ`, historical u/v and i/j and some abbreviation marks, and outputs decomposed characters (NFD). `german_print` also keeps `ſ`. The evaluator applies NFC by default, so decomposition alone does not count as an error; `ſ` versus `s` does unless the evaluation config uses NFKC ([EVALUATION.md](../../docs/EVALUATION.md)). The CATMuS character sets contain no háček (no `č`, `ř`, `ž`), so they cannot transcribe Czech correctly; `german_print` has these letters but was trained on German and Latin. CATMuS-Print is the default because it covers the widest range of Latin-script prints; it confuses many Fraktur letters (for example `h` and `b`), for which `german_print` is much better.

## Choosing and supplying models

Kraken models are specific to scripts, languages, periods and transcription guidelines; no single model covers every collection. Choose a recognition model whose training data match the material, and keep the segmentation model unless the layout is unusual.

1. **Find a model.** The [Kraken model repository on Zenodo](https://zenodo.org/communities/ocr_models) lists the openly licensed models (on a machine with network access, `kraken list` and `kraken show <DOI>` print them). Check the record's license, script, languages and training data. Only use weights whose license allows your use; do not use weights obtained from projects that do not permit it.
2. **Put it into a cache directory** on the host, for example `models/kraken/my-model.mlmodel` under a directory passed as `--cache`, and record its DOI and SHA-256. Recognition models must be trained on baseline segmentation; legacy bounding-box models are rejected.
3. **Reference it** in a copy of `engine.toml`: `recognition_model = "/cache/models/kraken/my-model.mlmodel"` (and `segmentation_model` for a custom segmentation model, which may also include a reading order model), and change `model` to name what is used, for example `"blla+my-model@zenodo.<record>"`. Then run with `ocrbench run start --cache <dir> ...`. The adapter records the path and SHA-256 of both models it actually loaded with every sample.

A missing file, a file that is not a Kraken model, or a model for the wrong task (for example a segmentation model given as `recognition_model`) makes every sample fail with `engine-error` and a message starting with `unsupported model` or naming the missing file.

## Build, run and evaluate

```sh
make gpu-preflight                  # host driver and NVIDIA Container Toolkit
make docker-cu130-torch             # shared torch base, once (about 6 GB)
make docker-kraken                  # this image (Kraken and the three recognition models)
uv run ocrbench run start --engine engines/kraken/engine.toml --dataset datasets/dev-01.json
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/kraken --inspect
make kraken-smoke                   # opt-in: build the image, run it on synthetic samples on the GPU
```

The container runs with `--gpus all` (`[resources]` in `engine.toml`) and uses `device`. If the requested GPU is not visible, every sample fails with `engine-error` (`cuda:0 requested but 0 CUDA devices are visible`); there is no silent CPU fallback. Kraken also runs on the CPU when `device = "cpu"` is set explicitly; that is much slower and not the configuration recorded here.

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message. Parameters left out (or `null`) keep the defaults below, which are recorded per sample.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `cuda:0` | `cuda:<n>`, or `cpu` |
| `recognition_model` | `catmus-print-fondue-large.mlmodel` | A model in the image (table above) or a path under `/cache/` |
| `segmentation_model` | `null` | `null` for Kraken's default `blla.mlmodel`, or a path under `/cache/` |
| `batch_size` | 1 | Lines recognized per batch (Kraken's default) |
| `segmentation_min_height` | 1200 | Images less than this many pixels high are segmented on a taller canvas (below); `0` turns this off |

Kraken's other inference settings keep their defaults: 32-bit precision, horizontal left-to-right page direction, Unicode BiDi reordering of the output, and the polygonal reading order heuristic. The sample `language` is not used; the model decides the script.

**Segmentation canvas.** The default segmentation model scales every input to a fixed height. A short image, such as a paragraph crop of a few lines, is then enlarged until the text no longer looks like page text, and Kraken finds only fragments with empty text. The adapter therefore centres an image less than `segmentation_min_height` pixels high on a canvas of that height, filled with the median colour of the image's top and bottom rows, and segments the canvas. It moves the segmentation back by the vertical offset (clipped to the image) **before** recognition, which runs on the original image. Every output therefore refers to input-image pixels. The offset is recorded as `canvas_offset`. On the synthetic samples this turns a two-line paragraph crop from seven empty lines into the correct transcription, and taller pages are not changed.

## Outputs

For every sample the adapter decodes the image as stored with Pillow (EXIF orientation not applied, as in [PREDICTIONS.md](../../docs/PREDICTIONS.md#positions); modes other than grayscale and RGB are converted to RGB), segments and recognizes it, and writes into the sample directory:

| File | Content |
| --- | --- |
| `kraken.xml` | Kraken's own PAGE XML serialization (2019 schema): regions with their type in `custom`, lines with boundary polygons, baselines and recognized text, words and glyphs with confidences |
| `kraken.json` | Kraken's segmentation container (`Segmentation`, loadable with `kraken.containers.Segmentation(**data["segmentation"])`) with each line's recognized text in `text`, and per-character `confidences` per line; the source of the prediction |
| `prediction.json` | Standardized prediction ([PREDICTIONS.md](../../docs/PREDICTIONS.md)), the primary output |
| `result.json` | Status, processing time and, in `extra`, `canvas_offset`, the numbers of Kraken lines and regions and of predicted lines and regions, the Kraken/torch/CUDA versions, the GPU name, the model loading time, the model files with SHA-256 and the parameters |

**Mapping.** The prediction is built from `kraken.json`, not by parsing `kraken.xml`: Kraken's PAGE template writes lines outside any region into an invented full-page region.

- **Lines:** every line with non-blank text becomes a line with its boundary polygon, baseline, mean character confidence (0–1, as in the PAGE line `TextEquiv`) and its Kraken line type when that is not `default`. `extra.kraken_id` is the `TextLine` ID in `kraken.xml`.
- **Order:** lines are read in Kraken's order, which is its reading order (the polygonal heuristic, or the last of `line_orders` when a custom segmentation model with a reading order model adds one).
- **Regions:** each run of consecutive lines in the same Kraken region (the first one listed for the line) becomes one region, with its boundary polygon as `polygon`, the Kraken class as `native_type` and a [recommended type](../../docs/PREDICTIONS.md#region-types): `text` for `text`, `paragraph` and SegmOnto `MainZone`/`MarginTextZone`; `heading` for `title`, `heading` and `TitlePageZone`; `figure`, `table`, `caption`, `footnote`, `page-header`, `page-footer` and `page-number` for the corresponding classes; anything else is `other` (`native_type` keeps the class). Consecutive lines outside any region form a region of type `text` without a polygon (`extra.grouping: "unassigned"`); this is not a claim of layout structure.
- **Reading order:** regions are in Kraken's line order and `reading_order` lists them in that order, so the page text keeps Kraken's line order exactly. If Kraken's order alternates between regions, a Kraken region whose lines are not consecutive becomes several regions, one per run of lines, each with the same polygon, type and `extra.kraken_id`; no text is repeated.
- **Not in the prediction:** regions without any recognized line (for example image regions, or the empty and zero-area `text` regions the default model reports on blank pages) and lines with blank text stay only in the native outputs. A region or line polygon with fewer than 3 points, or a baseline with fewer than 2, is omitted rather than invented.
- **Page text:** the lines of a region joined by newlines, regions joined by a blank line.

**Status.** An error in segmentation or recognition (for example CUDA out of memory) fails only that sample (`status: failed`, runner `engine-error`) and the container continues. An unusable GPU, invalid parameters, or a missing or unsupported model fail every sample. A page on which Kraken finds no lines, or only lines with blank text, is a **genuine empty result**: `success` with empty text and no regions; `kraken_lines` and `kraken_regions` in `result.json` show what was found.

## Tests

`tests/test_kraken.py` checks the mapping on a committed native result (`tests/fixtures/kraken/page.json` and `page.xml`, Kraken 7.1.1 output for a synthetic page; the PAGE XML is read with the benchmark's own PAGE reader and must contain the same lines), grouping, region types and reading order on hand-written results, parameter validation, the segmentation canvas, and the engine protocol through the runner with fake `torch` and `kraken` modules (`tests/fixtures/kraken/fake`), all offline in `make test`. `make kraken-smoke` builds the image and runs real Kraken on the GPU through `ocrbench run start`, `evaluate` (twice, without rerunning OCR) and `report` on a synthetic full page, a paragraph crop and a blank page (an empty `success`), plus a run with a GPU that is not visible, which must fail explicitly.
