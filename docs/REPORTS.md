# Reports

`ocrbench report` compares evaluated runs, typically one per engine or engine configuration. It reads the stored scores of `ocrbench evaluate` ([EVALUATION.md](EVALUATION.md#command-and-stored-scores)) and never rescores a sample or reruns an engine. The decisions are recorded in [#10](https://github.com/michal-hradis/ocr_benchmarking/issues/10) and the proposed [ADR 0010](adr/0010-reports.md).

## Commands

```sh
uv run ocrbench evaluate runs/<run-a>                 # once per run, or after a config change
uv run ocrbench evaluate runs/<run-b>
uv run ocrbench report runs/<run-a> runs/<run-b> -o reports/<name> [--inspect]
```

- Each argument is a run directory, which uses its newest evaluation (`evaluations/<n>` with the highest `n`), or an evaluation directory such as `runs/<run-a>/evaluations/2`.
- `-o` must name a **new** directory. An existing directory is refused, so a report never mixes files from different evaluations.
- `--inspect` also writes one GT-versus-prediction page per run and sample. It is off by default.
- The command logs every non-`success` sample on stderr, prints the evaluation used for each run and exits with status 1, writing nothing, when the report cannot be built.

Everything runs offline, without a GPU.

## Re-evaluation and stale scores

Changing the evaluation configuration, or rerunning failed samples with `run resume --retry-failed`, needs a new `ocrbench evaluate`. The new scores go to a new `evaluations/<n>` directory, and the report picks them up by default. The report refuses an evaluation, and asks for `ocrbench evaluate`, when:

- `scores.json` is not the current scores format (version 2);
- `run.json` changed after the evaluation;
- a sample has a newer attempt or a different status than the one scored;
- a scored `prediction.json` no longer matches its stored SHA-256;
- a GT file differs from the GT frozen with the run (`gt-changed`, `gt-unreadable`, `gt-unpinned`).

Runs in one report must also be **comparable**. They need the same metrics version and the same evaluation configuration (`config_sha256`, which includes the Unicode form and the failure policy). Every sample ID must have the same frozen input in every run: image SHA-256, GT SHA-256, unit and document type. A sample whose image or GT was not pinned when its run was created cannot be compared, so a report with more than one run refuses it. A run may appear only once. Otherwise the report is refused, so scores computed under different failure policies, or on different images, are never shown side by side.

## Output

```text
reports/<name>/
  report.json                   provenance, evaluation settings, aggregates, non-success samples
  aggregates.csv                one row per run and group
  samples.csv                   one row per run and sample
  index.html                    the same, for reading
  samples/<run-id>/<id>.html    with --inspect: GT versus prediction
```

**`report.json`** (`ocrbench.report`, version 1) records:

- the ocrbench version and the shared evaluation settings (configuration and its hash, metrics version, text rules);
- per run, its provenance: run ID and path, creation time, `run.json` SHA-256, engine name, version, model and image, engine configuration hash, image ID, runtime, dataset name and manifest hash, host, the evaluation directory and time, the `scores.json` SHA-256, the ocrbench versions of the run and of the evaluation, the rapidfuzz version, the number of samples and the per-status counts;
- the stored aggregates of every run, each tagged with its `run_id`;
- `non_success`: every sample whose status is not `success`, with run, sample, unit, document type, status, reason, the engine's message, attempt, how it was scored and the path of its `record.json`.

**Groups** are those of the evaluation: all samples, each unit (`page`, `paragraph`) and each document type within a unit. A sample without a document type falls into the group *document type unspecified*. Each run is one engine and configuration, so the runs are the engine/config grouping. Dataset splits are not used for grouping ([ADR 0010](adr/0010-reports.md)).

**`aggregates.csv`** has one row per run and group, in this column order:

- the run, engine, version and model, and the failure policy;
- `group` (`all`, `unit` or `doc_type`), `unit` and `doc_type`;
- `samples`, `scored`, `coverage` and the count of each status;
- for `cer`, `wer`, `fcer` and `fwer`: the micro rate, errors and reference length, and the macro rate with its sample count and its number of undefined values;
- for `fcer` and `fwer` also `lower_bound_rate` and `not_exact`;
- for `bow` and `rouge2`: micro precision, recall, F1, matched, reference and hypothesis counts, and the macro F1 with its sample count.

Rates are fractions, not percentages, and an undefined value is an empty cell. A text cell that a spreadsheet could read as a formula (one starting with `=`, `+`, `-` or `@`, also after leading whitespace, or with a tab or carriage return) gets a leading `'`. This matters for free-text fields such as the model or the document type.

**`samples.csv`** has one row per run and sample:

- the run, engine, version and model;
- the sample, unit, document type, status, reason, attempt and `scored_as`;
- the rate, errors and reference length of every edit metric, and `exact` for the flexible ones;
- precision, recall and F1 of bag-of-words and ROUGE-2.

An excluded sample has empty metric cells. Engine messages are left out of the CSV files; they are in `report.json` and `index.html`.

**`index.html`** shows:

- the evaluation settings and a note that scores on a small sample set are diagnostics, not evidence of generalizable rankings;
- a prominent warning when the failure policy is `exclude`;
- the provenance and status counts of each run;
- per group, one micro table (samples, scored, coverage, GT characters and words, CER, WER, fCER, fWER, bag-of-words P/R/F1 and ROUGE-2 F1) and one macro table (each mean with its sample count and its number of undefined values);
- the list of samples that did not succeed, with links to their execution records;
- a per-sample table.

A flexible score marked `*` includes samples whose result is not proven optimal ([EVALUATION.md](EVALUATION.md#flexible-cer-and-wer-metrics-version-2)).

## Failures

Under both failure policies, every `failure`, `partial` and `truncated` sample is listed with its status, reason, the engine's message and a link to its execution record. Each run also shows its per-status counts. Under `empty-output` (the primary score), a failure is scored as an empty prediction. Under `exclude`, the report states that failures are left out, and coverage (scored / all samples) is shown for every group.

## Inspection pages

With `--inspect`, each run and sample gets a page that shows:

- the status, reason, the engine's message and a link to the execution record;
- the counts (S, D, I and reference length) of CER, WER, fCER and fWER;
- the **ordered differences**: the normalized reference in reading order against the normalized page text of the stored prediction. Missing reference text is marked as deleted, extra predicted text as inserted;
- the **flexible groups** of fCER, in prediction order. Each group shows its reference and prediction line ranges, from the stored `group_lines`, with the differences inside it. The ranges are indices of the input lines; lines that normalize to nothing are skipped, as in the alignment. The unmatched non-empty reference and prediction lines are listed with their line numbers.

A failure scored as `empty` shows the whole reference as deleted. An `excluded` sample shows only its reference text.

The pages read the PAGE XML and the stored prediction on the host. The differences are only for display: the scores on the page are the stored ones. Image or position overlays are not part of the pages yet.

## Safety

OCR text, GT text and engine messages are untrusted. Every value in the HTML is escaped, and the pages contain no scripts. CSV text cells are protected against formula evaluation (see above). Links to execution records and `scores.json` are relative paths to the local run directory.
