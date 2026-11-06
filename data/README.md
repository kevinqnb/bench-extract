# data/

Everything under `data/` except this file and the `download_*.py` scripts is
gitignored -- regenerate on demand.

## VRDU

bench-extract benchmarks extraction strategies against
[VRDU](https://arxiv.org/abs/2211.15421) ("A Benchmark for Structured Extractions
from Complex Documents"): FCC political ad-buy forms (`ad-buy-form`, aka DeepForm)
and FARA foreign-agent registration forms (`registration-form`). Original data:
https://github.com/google-research-datasets/vrdu (Apache 2.0). Original evaluator:
https://github.com/google-research/google-research/tree/master/vrdu -- its match
functions are ported into `src/bench_extract/datasets/matching.py`, cited there.

Run once (clones fresh from GitHub, no auth needed):

    uv run data/download_vrdu.py

Produces:

    data/vrdu/
      pdfs/<corpus>/<filename>.pdf
      ocr/<corpus>/<document_id>.json    {"text": ..., "pages": [[start, end], ...]}
      few_shot-splits/<corpus>/*.json    VRDU's official train/valid/test splits,
                                          copied verbatim (train sizes 10/50/100/200
                                          x 3 seeds x task level -- this is the axis
                                          experiments/run_training.py sweeps)
      data.json                          {"corpora": {...}, "documents": {document_id: {...}}}
                                          -- everything bench_extract.datasets.vrdu reads

`document_id` is the PDF filename's stem (VRDU's own UUID-style filenames), used
consistently as the join key across `data.json`, `ocr/`, run output, and the
profiling table.

## Smithsonian Open Access

[Smithsonian Open Access](https://www.si.edu/openaccess), via Harvard LIL's
mirror on Source Cooperative:
https://source.coop/harvard-lil/smithsonian-open-access (updated weekly; usage
terms are per record, see each record's `metadata_usage` / `online_media`
`usage` fields). Intended for an image <-> record entity-linkage (semantic join)
task: the ground-truth join comes from the mirror's `search/linkages.parquet`.

Needs the `smithsonian` extra (DuckDB, to query the catalogs):

    uv run --extra smithsonian data/download_smithsonian.py file NMAAHC-0039C24D4B742_4004.jpg
    uv run --extra smithsonian data/download_smithsonian.py sample --per-unit 5 --max-jpgs 5 --seed 0 [--edan nmaahc npg]

`file` fetches one file (bare filename, `media/<folder>/<name>`, or full URL)
plus every record it is linked to. `sample` takes up to `--per-unit` records
from each Smithsonian unit (`--edan` restricts which, case-insensitive), ordered
by `md5(seed:record_id)`, plus all of their jpgs. A record is eligible only if
every jpg linked to it under `media/` is linked to no other record (so sampled
image sets are disjoint), it has 1..`--max-jpgs` of them, and the catalog names
its metadata shard.

Produces:

    data/smithsonian/
      search/{linkages,metadata}.parquet   cached catalogs (--refresh-catalog re-fetches)
      search/catalog.json                  each catalog's upstream Last-Modified
      records/<record_id>.json             full EDAN record, raw and unmasked
      media/<folder>/<file>.jpg            same relative path as upstream; <folder> is not
                                            always the unit code (e.g. all NMNH* units -> nmnh/)
      samples/per-unit-<n>_max-jpgs-<m>_seed-<s>[_edan-<units>]/
        gold_pairs.csv                     unit_code, record_id, record_path, jpg_path
                                            -- one row per jpg; written only after every
                                            download succeeded
        manifest.json                      args, catalog Last-Modified, eligible
                                            records per unit

Records are deliberately raw: they still contain the answer to the join
(`content.descriptiveNonRepeating.online_media` lists every linked image, and
filenames / some free-text fields carry catalog and image ids). Masking those
and renaming files to opaque ids is the job of whatever consumes this data,
not the downloader.
