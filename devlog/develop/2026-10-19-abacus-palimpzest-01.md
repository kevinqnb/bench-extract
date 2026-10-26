---
id: 2026-10-19-abacus-palimpzest-01
config: 2026-10-19-abacus-qwen3-smoke-01
---

## Session 2026-10-19

### Prompts
- Set up `AbacusExtractor` (`src/bench_extract/extractors/abacus.py`) on top of the pre-installed palimpzest library: configurable input dataset (VRDU `ad-buy-form` OCR text for now), configurable set of fields to extract (field descriptions to be written in a later session), a configurable custom validator class plus a training split defined in the experiment configs (reusing VRDU scoring), a choice of palimpzest's native optimization objectives (MaxQuality, MinCost, MinTime, ...), and, to be investigated, locally served vLLM models.
- Asked for a plan first. Review feedback: explain what the profiling table is and confirm palimpzest does not make the train/test split (the split is ours, defined in the dataset config); put the palimpzest helpers in `extractors/utils/`; tests and verification are in scope, using `Qwen/Qwen3-0.6B` (possibly several instances).
- Asked whether multiple non-local LLMs are allowed in palimpzest (answer: yes; the one-model limit applies to anything with an `api_base`).
- Follow-up questions on how the VRDU scoring function works, how palimpzest builds and applies its plan (decided to keep one `sem_map` over all fields), and whether it supports image models (it does; not used here).
- Asked to fix the serving mismatch so the Qwen servers can be run.

### Implemented
`AbacusExtractor.fit()` samples palimpzest candidate operators on the dataset's train documents, scored by a `score_field`-backed validator, and `extract_many()` runs the optimized plan; palimpzest imports and version-pinned shims (multi-vLLM check, vLLM input-token counting) live in `extractors/utils/pz_support.py`. Policy, `QueryProcessorConfig` fields and validator are set under `method_kwargs.abacus`. Added `bind_run_context`/`extract_many`/`fit_metrics` hooks to the extractor base class and both runners, a `served_model_name` model-config field, Qwen3-0.6B model configs and a smoke experiment config, an `abacus` extra, and changed `serve_model.py`/`submit.sh serve` to accept any `cuda` model on a free port. 80 tests pass, including mock-LLM end-to-end tests and one showing the plan chosen in `fit()` is the one `extract_many()` runs; the benchmark runner also ran end to end against a local stub server. No GPU job or real model has been run yet.

### Commits
- 95baf67 Wire palimpzest into AbacusExtractor; fix serve guard for external GPU models
