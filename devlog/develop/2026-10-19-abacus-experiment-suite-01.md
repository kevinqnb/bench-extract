---
id: 2026-10-19-abacus-experiment-suite-01
config:
---

## Session 2026-10-19

### Prompts
- Set up the experiment suite for testing the AbacusExtractor on VRDU:
  - fuller descriptions for every field in the extraction corpora (stubbed in `VRDUCorpusConfig`);
  - model configs for `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-4B` and `google/gemma-4-12B`;
  - support for serving each model independently and calling them all at once from a separate Abacus experiment;
  - real Abacus experiment configs over these models plus `gpt-oss-120b`, with `Qwen3.5-0.8B` left out for now;
  - one config per Abacus/Palimpzest optimization policy, under `experiments/experiment-configs/benchmark/abacus`;
  - basic output analysis in `analysis/abacus.py` (time vs. accuracy, tokens vs. accuracy, per-model usage).

### Implemented
Wrote `ad-buy-form` field descriptions, with a check that their keys equal `match_func_by_field`. Added Qwen3.5-0.8B/4B and Gemma-4-12B-it model configs (the `-it` repo, since `google/gemma-4-12B` is the pretrained base). Added seven Abacus benchmark configs, one per palimpzest policy, differing only in `policy`. Made `.endpoints.json` registration lock-protected and atomic for concurrent serve jobs. Extended `experiments/run_benchmark.py` to record tokens, elapsed time and per-model call counts, plus `artifacts/steps.csv`. Added `analysis/abacus.py`. Unit tests pass and a `--dry-run` of one config resolves; no serve or benchmark jobs were run. Known caveat: palimpzest prices all vLLM models at $0, so the cost-based policies degenerate.

### Commits
8b29e77 Add Abacus experiment suite: field descriptions, model/policy configs, analysis
