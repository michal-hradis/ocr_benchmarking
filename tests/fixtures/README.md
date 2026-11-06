# Test fixtures

Only small, deterministic, **synthetic** files that may be redistributed belong here: generated page images, hand-written PAGE XML, sample engine outputs. Never commit real scans, ground truth from the development set, model weights or run outputs.

Prefer generating a fixture in the test (for example under `tmp_path`) when it is a few lines of code; commit a file only when it has to be inspected or reused across tests. Keep each file small (tens of kB) and describe its purpose in the test that uses it.

Exception: an unchanged, redistributable third-party **schema** needed for offline validation may be larger, if it is vendored with its license and source (for example [`page-schema/`](page-schema/README.md), the 86 kB Apache-2.0 PAGE 2019 schema).
