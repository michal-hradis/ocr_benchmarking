# Roadmap and current work

This is a navigational index, not a competing task tracker. GitHub's [roadmap issue #1](https://github.com/michal-hradis/ocr_benchmarking/issues/1) and the linked issues are authoritative for scope, ownership, state and acceptance criteria.

**Status:** early planning and documentation. Most items below are not yet implemented.

## Project foundations

- [#2 — Project documentation and agent development workflow](https://github.com/michal-hradis/ocr_benchmarking/issues/2)
- [#3 — Architecture and versioned contracts](https://github.com/michal-hradis/ocr_benchmarking/issues/3)
- [#4 — Linux development tooling and compatible Docker base images](https://github.com/michal-hradis/ocr_benchmarking/issues/4) — **no initial CI**

## Benchmark foundation

- [#5 — Dataset manifest and PAGE XML reader](https://github.com/michal-hradis/ocr_benchmarking/issues/5)
- [#6 — JSON result schema with typed regions and nested lines](https://github.com/michal-hradis/ocr_benchmarking/issues/6)
- [#7 — Common runner and Docker protocol](https://github.com/michal-hradis/ocr_benchmarking/issues/7)
- [#8 — Baseline CER/WER and lexical overlap](https://github.com/michal-hradis/ocr_benchmarking/issues/8)
- [#9 — Scalable order-independent alignment](https://github.com/michal-hradis/ocr_benchmarking/issues/9)
- [#10 — Evaluation reports and reproducible runs](https://github.com/michal-hradis/ocr_benchmarking/issues/10)

## First engine integrations

- [#11 — Tesseract](https://github.com/michal-hradis/ocr_benchmarking/issues/11)
- [#12 — PaddleOCR PP-OCRv6](https://github.com/michal-hradis/ocr_benchmarking/issues/12)
- [#13 — Kraken historical OCR/HTR](https://github.com/michal-hradis/ocr_benchmarking/issues/13)
- [#14 — LightOnOCR-2](https://github.com/michal-hradis/ocr_benchmarking/issues/14)
- [#15 — FireRed-OCR](https://github.com/michal-hradis/ocr_benchmarking/issues/15)
- [#16 — OvisOCR2](https://github.com/michal-hradis/ocr_benchmarking/issues/16)
- [#17 — Infinity-Parser2-Flash](https://github.com/michal-hradis/ocr_benchmarking/issues/17)

Issues may proceed in parallel once their shared contracts and prerequisites are complete. The numeric order is not a blanket requirement to implement all engines serially.
