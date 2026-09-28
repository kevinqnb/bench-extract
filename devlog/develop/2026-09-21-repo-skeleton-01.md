---
id: 2026-09-21-repo-skeleton-01
config:
---

## Session 2026-09-21

### Prompts

Initialize the repo skeleton for benchmarking routing/cascade/scheduling extraction
strategies (HybridLLM, RouteLLM, FrugalGPT, BARGAIN, Task Cascades, Automix,
Abacus/Palimpzest, Doctopus, unified cascade-routing) against the VRDU benchmark
(FCC ad-buy forms, FARA registration forms): a gitignored data pipeline referencing
the original VRDU library, three base classes (`RouteExtractor`/`CascadeExtractor`/
`ScheduleExtractor`) with high-level outlines for each strategy family, and
experiment infrastructure for (1) accuracy-vs-efficiency benchmarking and (2)
training-dataset-size sweeps -- not a finished product, an organized skeleton. A
specific `model-configs/gpt-oss-120b.yaml` example was requested, modeled on an
existing sibling project's.

Follow-up after the initial build: why does `.env.example` hold a base URL for a
model that's "always run locally," given the cluster reality is likely multiple LLMs
served on different nodes at once and called from a separate extraction job; and
`CLAUDE.md` should not carry anything specific to a personal build environment
(e.g. the private notes directory) since this is a shared repository.

### Implemented

Built the full skeleton: `data/download_vrdu.py` (clones+reshapes the official VRDU
release), `src/route_extract/` (ported VRDU match functions, VRDU loading, an
`OpenAIChatModel`, the shared profiling-table spine, and the three base extractor
classes with real working baseline strategies plus 9 typed stub adapters, one per
method), and `experiments/` (model-configs/dataset-configs/experiment-configs,
`run_benchmark.py`/`run_training.py` following the project's config envelope and
run-output contract). In response to follow-up review: replaced the single-job
local-vLLM serving default with a proper serve/client split (`experiments/
serve_model.py` starts one model's server standalone, registered via
`experiments/.endpoints.json`; `submit.sh` gained a `serve` subcommand), since
comparing routing/cascade strategies needs several models reachable at once rather
than served one at a time inside the benchmark job; and split personal-harness
content out of `CLAUDE.md` into a gitignored `CLAUDE.local.md`. 51 unit tests pass;
dry-run and a full non-network pipeline run were both verified against real
downloaded VRDU data. None of the 9 methods' upstream code is wired in yet -- each
adapter documents what that requires.

### Commits
51e0cfd Scaffold route-extract: VRDU pipeline, extraction strategy base classes, experiment harness
