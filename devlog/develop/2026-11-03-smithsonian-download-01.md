---
id: 2026-11-03-smithsonian-download-01
config:
---

## Session 2026-11-05

### Prompts
- Build a small `download_smithsonian.py` for https://source.coop/harvard-lil/smithsonian-open-access with two options: (1) pull an exact file by name (e.g. `NMAAHC-0039C24D4B742_4004.jpg`) plus its metadata, and (2) a bulk random download of n records plus their source files; filter by Smithsonian unit, since a first task should avoid units with 3D models.
- Reframed the target as an entity-linkage (semantic join) task: given media, join it to the right metadata record. Use only `.jpg`s, with `search/*.parquet` as ground truth; asked for 30 empirical samples (stratified across units with jpgs) to find patterns before writing the loader.
- Task decisions: no distractors (every candidate has a correct answer); keep only records whose jpgs are linked to no other record, so the join is disjoint; feed all of a record's jpgs, max 5 (the p99), skipping larger records; text and barcodes visible in images are fair game; keep realistic hard cases (micrographs, blank paper); no byte cap yet; first test is 5 records from each of the 28 units with eligible records.
- The downloader saves raw data only
- DuckDB only as an optional `--extra smithsonian`, so users of other datasets don't need it. Asked to verify that the `uv.lock` re-resolve doesn't change anything for the `abacus` extra.
- Asked for strict PEP8 and descriptive names, reviewed the file line by line, then asked for a leaner rewrite matching the style of their own edits (concise docstrings, one blank line between functions for now, readable SQL names) that drops redundant work.

### Implemented
`data/download_smithsonian.py` (behind the new `smithsonian` extra, `duckdb>=1.1`) caches the mirror's `search/{linkages,metadata}.parquet` catalogs and queries them with DuckDB. `file` resolves a bare filename, relative path or URL (failing loud on ambiguous names) and fetches the file plus its linked records from the JSONL metadata shards. `sample` builds an eligibility table (1..`--max-jpgs` media jpgs, each linked only to that record), takes up to `--per-unit` records per unit ordered by `md5(seed:record_id)`, downloads the records and jpgs raw, and writes `gold_pairs.csv` plus `manifest.json` under `data/smithsonian/samples/<params>/`. `tests/test_download_smithsonian.py` has 6 passing tests against synthetic parquets; `uv.lock` installs identical package sets for every extras combination except the added duckdb, and the full suite passes with `abacus` installed. The `--per-unit 5 --max-jpgs 5 --seed 0` run produced 227 gold pairs over 136 records (some units have fewer than 5 eligible records), about 1.4 GB of jpgs.

### Commits
- 9423392 Added in Smithsonia data loader and tests
