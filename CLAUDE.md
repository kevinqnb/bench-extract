# bench-extract

Benchmarks pre-existing routing / cascade / scheduling strategies for information
extraction -- methods that take a fixed collection of extraction models M (LLMs or
smaller NLP models) and decide, per document or per field, which model(s) in M to
call -- against the [VRDU](https://arxiv.org/abs/2211.15421) benchmark (FCC
political ad-buy forms, FARA foreign-agent registration forms).

## Conventions

Every experiment run is defined by a committed config,
`experiments/experiment-configs/{benchmark,training}/<id>.yaml` (`id` format
`YYYY-MM-DD-slug-NN`), with a fixed envelope --
`id`/`project`/`description`/`seed`/`params` -- and a free-form `params` block.
**No magic numbers**: every value that varies between runs lives there, never
hardcoded in runner code. Every run writes `run.json` (manifest), a verbatim
`config.snapshot.yaml`, flat-scalar `metrics.json`, and `log.txt` under
`experiments/results/{benchmark,training}/<id>/` (gitignored).

`devlog/` holds a curated, public record of AI-assisted work on this repo --
see `devlog/README.md`.

(If you develop this repo through Kevin's personal Claude Code harness --
`notes/hub/conventions.md`, the `/develop`/`/devlog`/`/experiment`/`/explog`/
`/debrief` commands -- see `CLAUDE.local.md`, which is gitignored and specific
to that setup.)

## Data

[VRDU](https://arxiv.org/abs/2211.15421) ("A Benchmark for Structured Extractions
from Complex Documents"): `ad-buy-form` (aka DeepForm, FCC political ad-buy forms)
and `registration-form` (aka FARA, foreign-agent registration forms). Original data:
https://github.com/google-research-datasets/vrdu. Original evaluator:
https://github.com/google-research/google-research/tree/master/vrdu -- its match
functions are ported (not reinvented) into `src/bench_extract/datasets/matching.py`.

Everything under `data/` is gitignored except `data/README.md` and
`data/download_vrdu.py`. Run `uv run data/download_vrdu.py` once to populate it. See
`data/README.md` for the reshaped layout.

## The profiling-table spine

`RouteExtractor`, `CascadeExtractor`, and `ScheduleExtractor`
(`src/bench_extract/extractors/base.py` + the three `*_extractor.py` files) all fit
against the same artifact: a table with one row per `(document_id, field,
model_key)` -- prediction, ground truth, `correct` (via `datasets/matching.py`),
wall-clock time, token usage. Built by `src/bench_extract/profiling.py`. Keying on
`(document_id, field)` rather than just `document_id` is required for the
scheduling methods (Abacus, Doctopus), which assign per attribute, not per document.

Each base class has a real, working baseline strategy (see its module docstring):
`RouteExtractor` always routes to the best model on the training profile,
`CascadeExtractor` escalates through a fixed model order and accepts the first
non-empty prediction, `ScheduleExtractor` greedily assigns the cheapest model that
clears an accuracy floor per field. The seven method-specific adapters
(`hybridllm.py`, `routellm.py`, `frugalgpt.py`, `bargain.py`, `task_cascade.py`,
`automix.py`, `abacus.py`, `doctopus.py`) and the unified `cascade_routing.py`
subclass one of these three and currently raise `NotImplementedError` from `fit()`,
documenting exactly which upstream repo/module wiring them in requires -- see
`EXTRACTOR_REGISTRY` in `experiments/config.py` for the full method list, including
the three working baselines under `route_baseline`/`cascade_baseline`/
`schedule_baseline`.

## Serving multiple models

A routing/cascade/scheduling comparison needs every model_key in M reachable
*at the same time*, not served one at a time inside the benchmark job -- so
each self-hosted LLM gets its own standalone server, typically its own
node/GPU allocation:

    bash experiments/submit.sh serve <model-key>        # one per self-hosted model, stays running
    bash experiments/submit.sh benchmark <experiment-id> # calls out to all of them

`experiments/serve_model.py` starts one model's vLLM server and registers its
endpoint in `experiments/.endpoints.json` (gitignored); `experiments/
serving.py`'s `endpoint_for()` discovers it from there automatically. See
`experiments/model-configs/README.md`'s "Serving" section for the
`local_vllm`/`external` distinction and when each applies.

## Layout

```
data/                            download_vrdu.py + README; everything else gitignored
src/bench_extract/
  datasets/vrdu.py                loads data/vrdu/data.json -> VRDUDocument
  datasets/matching.py            ported VRDU match functions (source: google-research)
  models/base.py                  ExtractionModel ABC + OpenAIChatModel
  profiling.py                    build_profiling_table -- the shared evaluation path
  extractors/base.py               MultiModelExtractor ABC (+ ExtractionResult/Step)
  extractors/route_extractor.py    RouteExtractor + working baseline
  extractors/cascade_extractor.py  CascadeExtractor + working baseline
  extractors/schedule_extractor.py ScheduleExtractor + working baseline
  extractors/{hybridllm,routellm,frugalgpt,bargain,task_cascade,automix,abacus,
              doctopus,cascade_routing}.py   method stubs, see module docstrings
experiments/
  config.py                       model-configs/dataset-configs loaders, EXTRACTOR_REGISTRY
  serving.py                      LocalVLLMServer + endpoint discovery (ported from govscape-extract)
  serve_model.py                  starts one model's server as its own long-running job
  run_benchmark.py                 accuracy vs. efficiency runner
  run_training.py                  training-set-size sweep runner (VRDU's real few_shot-splits)
  model-configs/*.yaml            one file per LLM in M
  dataset-configs/vrdu.py          per-corpus windowing + extraction prompt template
  experiment-configs/{benchmark,training}/<id>.yaml
  results/                        gitignored run output
tests/                            fixture-based unit tests, see tests/fixtures/vrdu_mini/
```

## Status

Skeleton stage: the profiling-table spine, VRDU loading/scoring, the three base
extractor classes with working baselines, and the config-dispatched runners are
implemented and unit-tested. None of the seven methods' actual upstream code
(HybridLLM, RouteLLM, FrugalGPT, BARGAIN, Task Cascades, Automix, Abacus/Palimpzest,
Doctopus) or the unified cascade-routing paper is wired in yet -- each adapter file
documents what that requires. No GPU job has been run; `data/download_vrdu.py` has
not yet been run to populate `data/vrdu/`.
