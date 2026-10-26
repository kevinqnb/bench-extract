# data/

Everything under `data/` except this file and `download_vrdu.py` / `download_cuad.py` is gitignored --
regenerate on demand.

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

## CUAD

[CUAD](https://arxiv.org/abs/2103.06268) (Contract Understanding Atticus Dataset):
510 commercial contracts, 41 clause categories, expert-labelled with verbatim clause
spans (zero, one or many per category per contract). Data (CC BY 4.0):
https://zenodo.org/records/4595826. Official split, category descriptions and
evaluator: https://github.com/TheAtticusProject/cuad -- its match rule is ported
into `src/bench_extract/datasets/cuad_matching.py`.

Run once (downloads ~106 MB from Zenodo, md5-checked, and clones the GitHub repo):

    uv run data/download_cuad.py

Produces:

    data/cuad/
      txt/<document_id>.txt       full contract text (the labels' character offsets index into this)
      pdfs/<document_id>.pdf      509 of 510 (one contract has no PDF upstream)
      splits/official.json        {"train": [408 ids], "test": [102 ids]} -- the official split
      data.json                   {"source", "categories": {slug: CUAD name}, "documents": {document_id: {title, text, fields}}}
      category_descriptions.csv   copied verbatim; descriptions feed dataset-configs/cuad.py
      master_clauses.csv          copied verbatim; the only place normalized answers live (unused)

`document_id` is the contract title, NFC-normalized (some upstream filenames differ
from the json titles in Unicode normalization form). CUAD has no validation split;
`data.json` keeps all 41 categories; `experiments/dataset-configs/cuad.py` extracts only the 10 that aren't yes/no questions and documents the `valid_source` options.
