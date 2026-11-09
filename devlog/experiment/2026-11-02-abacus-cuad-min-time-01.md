---
id: 2026-11-02-abacus-cuad-min-time-01
config: experiments/experiment-configs/benchmark/abacus-cuad/2026-11-02-abacus-cuad-min-time-01.yaml
---

## Session 2026-11-02

### Prompts

"Please help me set up and run an experiments like 2026-10-19-abacus-max-quality-01.yaml
and 2026-10-19-abacus-min-time-01.yaml, but now for the newly implemented CUAD dataset
(see experiments/dataset-configs/cuad.py). I want to serve and use the same set of
models, then run extraction over this dataset using the abacus extractor. After
extraction, I want to analyze time vs accuracy in the same fashion as
analysis/abacus_compare.py. Note that the measure of accuracy for the CUAD dataset is
different than that for VRDU, so this will be a separate analysis."

Follow-ups: use `max_chars=30000`; proceed as is with the failing qwen calls but record
how many occur during the full run; run the benchmark as jobs rather than in the shell.

### Hypothesis

"My hypothesis was that the max quality run will have higher accuracy at the cost of more expensive runtime."

(Stated for the pair of runs: the MaxQuality run versus the MinTime run.)

### What was run

Abacus (palimpzest) under the MinTime policy on CUAD, over M = {qwen3.5-4b, gemma-4-12b,
gpt-oss-120b}, each model served by its own `experiments/submit.sh serve <model-key>`
job beforehand. Config: `experiments/experiment-configs/benchmark/abacus-cuad/2026-11-02-abacus-cuad-min-time-01.yaml`;
entry point `experiments/run_benchmark.py` (`uv run -m experiments.run_benchmark`), submitted as a CPU
cluster job via `experiments/submit.sh benchmark 2026-11-02-abacus-cuad-min-time-01 --qsub` (that option is a local,
uncommitted addition to the gitignored wrapper; the two runs were sequential). Run directory:
`experiments/results/benchmark/2026-11-02-abacus-cuad-min-time-01/`.

### Commits

d4b4a09 Add CUAD Abacus configs (MaxQuality/MinTime), CUAD analysis script, Python.h fix for jobs
