---
id: 2026-09-28-pluggable-datasets-01
config:
---

## Session 2026-09-28

### Prompts

"We need to make the experiment configs more generally defined. Right now they
just assume that we're using the VRDU dataset. That needs to be made into an
explicit option for the user to choose in case we add more datasets in the future.
Then anything specific about that dataset like the corpus or the split files should
be placed into the dataset config file for VRDU." Scope check: asked whether to add
just a config-layer `dataset` field (checked but otherwise inert) or full loader
dispatch so the runners actually route through it -- answered "full loader dispatch
now."

Follow-up in the same vein: "Any dataset specific extraction fields should be
defined within their own config and not assumed anywhere else." When that led to
trying to download the `registration-form` corpus to get its real field schema
rather than guess it, declined: "That's okay, we can keep registration blank and
list it as to do for now."

### Implemented

Added `params.dataset` (a key into a self-discovered `DATASET_REGISTRY` in
`experiments/config.py`) and `params.dataset_params` to the benchmark/training
config schema; `run_benchmark.py`/`run_training.py` now dispatch through a dataset
module's `DatasetRunContext` (new `bench_extract.datasets.base` contract:
`load_run_context`, `load_split_doc_ids`, `describe_split`,
`training_split_params`, plus `windowed_text`/`score_field` callables) instead of
importing VRDU's loader and matching code directly -- a second dataset needs one
new `dataset-configs/<key>.py` file, not runner changes. Moved VRDU's per-corpus
field list and match functions out of implicit inference from the gitignored
`data/vrdu/data.json` into an explicit, committed `CORPORA[corpus].match_func_by_field`
in `dataset-configs/vrdu.py`, populated from real downloaded `ad-buy-form` data;
`registration-form`'s map is left an explicit empty TODO rather than guessed, and
`load_run_context` refuses to run it (`NotImplementedError`) until it's filled in,
cross-checking the config against `data.json` as a drift guard everywhere else.
Updated both example experiment configs, `CLAUDE.md`, and test fixtures/coverage
to match; 62 tests pass, both runners' dry-runs and the new field/drift checks were
exercised against real downloaded data (registration-form untested, as noted above).

### Commits

- `7e9bd3a` Generalize experiment configs to a pluggable dataset registry
