# OCR-D/Calamari engine (historical OCR pipeline, CPU)

Historical OCR baseline ([#46](https://github.com/michal-hradis/ocr_benchmarking/issues/46)). A fixed, minimal [OCR-D](https://ocr-d.de) pipeline:

1. `ocrd-cis-ocropy-binarize` ([ocrd_cis](https://github.com/cisocrgroup/ocrd_cis)) binarizes the page with ocropy's nlbin method.
2. `ocrd-tesserocr-segment` ([ocrd_tesserocr](https://github.com/OCR-D/ocrd_tesserocr)) segments the page into regions and text lines with Tesseract's layout analysis and writes their reading order.
3. `ocrd-calamari-recognize` ([ocrd_calamari](https://github.com/OCR-D/ocrd_calamari)) recognizes every text line with a pretrained [Calamari](https://github.com/Calamari-OCR/calamari) model ensemble.

The adapter runs the three processors in its own process. All processor parameters keep their OCR-D defaults except the Calamari model and the CPU device. The pipeline cannot be configured beyond the choice of model ([Presets](#presets)); it is a baseline, not a workflow-design tool.

It runs in a **standalone CPU Docker image** and follows the engine protocol in [RUNNER.md](../../docs/RUNNER.md#engine-protocol). ocrd_calamari 2.1.0 requires TensorFlow below 2.16, which has no Python 3.12 wheels, so the image builds on `python:3.11-slim-bookworm` pinned by digest. No base family fits, which [ADR 0004](../../docs/adr/0004-docker-base-image-families.md) allows. The image follows the base conventions of [DOCKER.md](../../docs/DOCKER.md): `/opt/venv`, user `ocr`, `/cache`.

| File | Purpose |
| --- | --- |
| `Dockerfile` | Engine image `ocrbench/engine-ocrd-calamari:2.1.0-v1` |
| `requirements.in`, `requirements.txt` | OCR-D core, ocrd_cis, ocrd_tesserocr, tesserocr, Calamari and TensorFlow, hash-locked for Python 3.11 (`make docker-lock`) |
| `models.sha256` | Checksums of the three model archives; the build fails if they differ |
| `models.files.sha256` | Checksums of every file unpacked from the archives; checked by the build and by the adapter before it loads a model |
| `ocrd_calamari_engine.py` | The adapter |
| `engine.toml` | Engine configuration for `ocrbench run start`, preset `gt4histocr` |
| `presets/fraktur19.toml`, `presets/antiqua.toml` | Presets with the other two model ensembles ([Presets](#presets)) |

## Versions and licences

| Component | Version | Licence |
| --- | --- | --- |
| OCR-D core (`ocrd`) | 3.13.3 | Apache-2.0 |
| ocrd_cis | 0.2.0 | MIT |
| ocrd_tesserocr | 0.21.2 | MIT |
| tesserocr, with Tesseract 5.5.3 and Leptonica 1.87.0 bundled in its wheel | 2.12.0 | MIT; Tesseract Apache-2.0 |
| Tesseract's English model for the segmentation (`tesseract-ocr-eng`, tessdata_fast) | Debian `1:4.1.0-2` | Apache-2.0 |
| ocrd_calamari | 2.1.0, commit `4d66fa1b809abe124a8e632a0eda6a2c133d9f2d`, source archive checked by SHA-256 in the `Dockerfile` | GPL-3.0 |
| calamari-ocr | 2.3.1 | GPL-3.0 |
| TensorFlow (CPU) | 2.15.1 | Apache-2.0 |
| Calamari models | [calamari_models](https://github.com/Calamari-OCR/calamari_models) release 2.2 | MIT |

ocrd_calamari 2.1.0 is not on PyPI, so it is installed from its GitHub source archive with `--no-deps`. Its declared dependencies are in `requirements.txt`. The Debian packages (`libmagic1` for OCR-D, `tesseract-ocr-eng`) are pinned by version, not by an archive snapshot. If Debian removes a version, the build fails rather than silently changing it; then bump the pin, the image tag (`-v2`) and `engine_version`.

**Models.** Each archive holds a 5-fold voting ensemble that Calamari combines (`confidence_voter_default_ctc`). The three archives are downloaded **when the image is built** (about 90 MB together) and checked against `models.sha256`, so inference needs no `--cache` and no network. Every unpacked file is checked against `models.files.sha256` at build time.

Before loading a model, the adapter checks that its directory holds exactly the files in `models.files.sha256`, with those SHA-256 digests. Otherwise every sample fails with `engine-error` and a message naming the missing, unexpected or modified files. This also applies when `OCRBENCH_OCRD_CALAMARI_MODELS` points the adapter at another model directory. A recorded archive SHA-256 therefore always describes the checkpoints that were loaded. The two checksum files sit next to the adapter and cannot be overridden.

| Model | Archive SHA-256 | Training data (calamari_models README) |
| --- | --- | --- |
| `gt4histocr` | `d0b063fb…bc5047` | The whole [GT4HistOCR](https://zenodo.org/record/1344132) corpus: German and Latin prints, 15th–19th century, Fraktur and Antiqua (grayscale nlbin images) |
| `fraktur_19th_century` | `7e4460e5…842684` | The DTA19 part of GT4HistOCR and other 19th-century Fraktur |
| `antiqua_historical` | `d3326df9…f090b7` | The Antiqua parts of GT4HistOCR |

**Czech.** No model was trained on Czech, and their alphabets cannot produce most Czech letters. The models use NFC text, and none has the combining caron (U+030C) or the precomposed letters č, ď, ě, ň, š, ť and ž. `gt4histocr` has á, é, í, ó, ř, ú and ů, `antiqua_historical` has á, é, í, ó and ú, and `fraktur_19th_century` has only é. Czech text is therefore misread at every one of these letters, whatever the image quality.

The calamari_models `idiotikon` ensemble (NFD, with a combining caron, but no ring above for ů) would cover more of Czech. It is not one of the agreed presets and would be a follow-up.

## Build, run and evaluate

```sh
make docker-ocrd-calamari           # this image (OCR-D, Calamari, TensorFlow and the models)
uv run ocrbench run start --engine engines/ocrd-calamari/engine.toml --dataset datasets/dev-01.json
uv run ocrbench run start --engine engines/ocrd-calamari/presets/fraktur19.toml --dataset datasets/dev-01.json
uv run ocrbench run start --engine engines/ocrd-calamari/presets/antiqua.toml --dataset datasets/dev-01.json
uv run ocrbench evaluate runs/<run-id>
uv run ocrbench report runs/<run-id> -o reports/ocrd-calamari --inspect
make ocrd-calamari-smoke            # opt-in: build the image, run it on synthetic samples on the CPU
```

No GPU and no base image are needed. The container gets 4 CPUs and 8 GB of memory (`[resources]` in `engine.toml`). TensorFlow is the CPU build path (`CUDA_VISIBLE_DEVICES` is empty, Calamari `device` is `-1`). Tesseract runs single-threaded (`OMP_THREAD_LIMIT=1`).

## Parameters

`[params]` in `engine.toml` are passed to the adapter in `job.json`. Unknown or invalid values make every sample fail with `engine-error` and a message.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model` | `gt4histocr` | Calamari ensemble: `gt4histocr`, `fraktur_19th_century` or `antiqua_historical` |

The parameters each processor actually used are recorded by OCR-D in the `Metadata` of `page.xml`. The defaults that matter most are:
- binarization: method `ocropy` with threshold 0.5 and DPI from the image metadata;
- segmentation: bounding-box regions and lines, tables found (`find_tables`), DPI from the image metadata or, failing that, estimated by Tesseract;
- recognition: line-level text and the default voter.

## Presets

Each preset is a separate engine configuration with its own name; they differ only in `model`.

| Preset (file) | Engine name | `model` |
| --- | --- | --- |
| `gt4histocr` (`engine.toml`) | `ocrd-calamari` | `gt4histocr` |
| `fraktur19` (`presets/fraktur19.toml`) | `ocrd-calamari-fraktur19` | `fraktur_19th_century` |
| `antiqua` (`presets/antiqua.toml`) | `ocrd-calamari-antiqua` | `antiqua_historical` |

## Outputs

Each sample directory contains:

| File | Content |
| --- | --- |
| `page.xml` | The PAGE XML written by `ocrd-calamari-recognize`, unchanged. It holds the regions, lines and reading order from Tesseract, Calamari's line text with confidences, and OCR-D's processing metadata. Its `AlternativeImage` entries name derived images from the temporary workspace, which are not kept. |
| `prediction.json` | Standardized prediction built from `page.xml` ([PREDICTIONS.md](../../docs/PREDICTIONS.md)) |
| `result.json` | Status and `extra`: region and line counts, the seconds per step (`step_seconds`), the package versions and Tesseract version (`environment`), the Calamari model with its checkpoints, archive SHA-256 and the number of verified files (`models`), and `params` |

**Mapping.** Every TextRegion with at least one non-blank line becomes a region `r1`, `r2`, … in the order of the PAGE `ReadingOrder`. Indexed groups are read by `@index`, and nested groups are read in place, as in the benchmark's PAGE reader. Tesseract writes a table as a group that references the TableRegion and its cells, so the cells, which are TextRegions inside the TableRegion, keep the table's position and order. Their `extra.parent` names the table. Regions that the reading order does not reference follow in file order.

Each region keeps:
- its bounding-box polygon;
- the Tesseract block type as `native_type`: `heading` and `caption` map to the same types, `paragraph` to `text`, `floating` (pull-out text) to `other`, and regions without a type to `text`;
- its OCR-D id in `extra`.

Lines keep their file order, their polygon (a box, since Tesseract gives no baseline) and the Calamari confidence. Regions without text, such as images and separators, stay only in `page.xml`. The coordinates are those of the original image: OCR-D maps everything back from its derived images.

**Failures.** An error in any processor on a page fails that sample with the processor's exception as the message (`OCRD_MISSING_OUTPUT=ABORT`); the adapter continues with the next sample. A page without text is a valid empty result.

## Observed behaviour and restrictions

- **Runtime.** The ensembles are loaded once per container, which takes about 15 s. Within a page, binarization and segmentation take well under a second on the synthetic pages; recognition dominates, at about 1 s per line on real pages ([dev-01](#measured-on-dev-01)). ocrd_calamari predicts in a background process with fixed batches of 12 lines and pads the last partial batch of each page by waiting 1.1 s for every missing line. This adds up to about 11 s per page, whatever its size: 10 s for a two-line paragraph and 14 s for the 19-line newspaper page.
- **Not bit-reproducible.** Lines are batched as they arrive and padded to the widest line of their batch, so the batch composition depends on timing. Two runs of the same image gave different readings of a few characters on the synthetic newspaper page.
- **Segmentation.** Tesseract finds the three columns of the synthetic newspaper page and orders them correctly. On the two-column page, whose gutter is narrow, it makes one block per row band and reads each line across both columns.
- **Image input.** Calamari reads the latest derived image, which is the binarized and clipped page from the first two steps. The `gt4histocr` ensemble was trained on grayscale images, the other two on binarized ones. EXIF orientation is not applied (OCR-D reads the stored pixels). The binarization takes the resolution from the image metadata; a wrong declared DPI degrades it ([dev-01](#measured-on-dev-01)).

## Measured on dev-01

The three presets on the seven pages of dev-01 (runs `ocrd-calamari-dev01`, `ocrd-calamari-fraktur19-dev01` and `ocrd-calamari-antiqua-dev01`, not in Git; image `sha256:31d96174…` built from the committed files; AMD Ryzen 5 2600X, the runs one after another with 4 CPUs each). The Tesseract preset `best-deu-latf-psm3` (run `tesseract-best-deu-latf-psm3-dev01`) is shown for comparison. All 21 samples were `success`. Rates are micro averages over the pages; fCER and fWER are the flexible, order-independent rates ([EVALUATION.md](../../docs/EVALUATION.md)). Engine time is the sum of the adapter's per-sample times.

| Engine, preset | CER | WER | fCER | fWER | BoW F1 | Engine time (7 pages) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ocrd-calamari` (`gt4histocr`) | 55.2 % | 85.1 % | 46.8 % | 82.5 % | 30.4 | 922 s |
| `ocrd-calamari-fraktur19` | 57.2 % | 84.9 % | 46.7 % | 80.3 % | 28.2 | 916 s |
| `ocrd-calamari-antiqua` | 62.0 % | 93.0 % | 57.4 % | 92.4 % | 15.9 | 914 s |
| `tesseract-best-deu-latf-psm3` | 54.6 % | 84.0 % | 35.9 % | 74.7 % | 39.6 | 141 s |

CER per page in % (fCER in parentheses where it differs). Lines are those of the `gt4histocr` run.

| Page | Content | Lines | `gt4histocr` | `fraktur19` | `antiqua` | Tesseract `deu_latf` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `3e6309d0…` | Czech newspaper, 4194 × 6192 | 343 | 74.7 (58.9) | 74.6 (55.9) | 75.6 (67.7) | 85.1 (67.3) |
| `bd497a62…` | Czech newspaper, 3509 × 5239 | 403 | 49.5 (43.9) | 50.3 (42.4) | 57.1 (53.5) | 44.9 (20.1) |
| `CkXII61-0017` | Glagolitic print | 18 | 76.6 | 78.7 | 76.8 | 85.2 |
| `dta-346672…` | Latin, Antiqua and italic | 31 | 52.0 | 75.2 | 61.3 | 23.1 (17.3) |
| `dta-laube…` | German Fraktur, 1833 | 25 | 7.4 | 4.4 | 34.9 | 7.2 |
| `dta-martzi…` | German Fraktur, 1692 | 24 | 47.9 | 72.2 | 83.9 | 23.6 |
| `dta-steinen…` | German Antiqua, 1894 | 43 | 12.1 | 19.8 | 27.3 | 2.1 |

- **Czech newspapers.** None of the models can write most Czech letters ([Czech](#versions-and-licences)), and the large pages are read in many small Tesseract blocks (64 and 43 regions), so both the ordered and the flexible rates stay high.
- **Fraktur.** `fraktur19` is the best preset on the 1833 Fraktur page (4.4 %), `gt4histocr` is close behind (7.4 %), and `antiqua` fails on it, as expected.
- **Declared resolution.** The four DTA images declare 72 DPI, which is much lower than their real resolution. ocropy-binarize takes the DPI from the image and scales its background estimate by 300/DPI. At 72 DPI the background window is about four times too small for the strokes, so thick strokes are hollowed out to their outlines and Calamari drops letters. Binarization also takes 24–34 s per page instead of 1–3 s. This hurts most on the bold 1692 Fraktur page. The newspapers declare no DPI and are binarized at zoom 1. Rerunning `dta-martzi…` and `dta-346672…` outside the benchmark with the binarization parameter `dpi=300` gave 10.8 % instead of 47.9 % and 41.1 % instead of 52.0 % CER (`gt4histocr`). The remaining errors on `dta-346672…` are mostly in its italic lines and small text fragments. The committed configuration keeps the OCR-D default (DPI from the image).
- **Runtime.** Recognition takes about 1 s per line with the five-model ensemble on 4 CPUs: 356 s for the 343-line newspaper page. At this rate, a page with more than about 800 lines would exceed the 900 s sample timeout.

## Tests

`tests/test_ocrd_calamari.py` checks the mapping on committed pipeline output. The files in `tests/fixtures/ocrd-calamari/` (`page.xml`, `newspaper.xml` and `blank.xml`) are real output of this image with preset `gt4histocr` for synthetic pages rendered by `tests/synthetic.py`. Hand-written PAGE files cover tables, types, reading order and omissions. The protocol test runs the adapter as a host process with fake OCR-D modules (`tests/fixtures/ocrd-calamari/fake`) through the runner and covers success, an empty page, failing processors and missing output, and setup errors. The tests use placeholder model files with their own checksums, and check that a missing, extra or modified file, or a model directory with other weights, is rejected. The parameters and presets are also checked.

The opt-in `make ocrd-calamari-smoke` builds the image and runs the real pipeline on the synthetic pages, including the two-column and newspaper pages, through `run`, `evaluate` and `report`. It checks the recorded versions, the geometry and that the native PAGE is readable by the benchmark. It requires a character error rate below 15 % on the single-column and newspaper pages, and runs every preset.
