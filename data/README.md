# data/

Everything under `data/` except this file and `download_vrdu.py` is gitignored --
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
