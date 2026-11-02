---
id: 2026-11-02-abacus-cuad-suite-01
config: experiments/experiment-configs/benchmark/abacus-cuad/2026-11-02-abacus-cuad-max-quality-01.yaml
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

Follow-ups: use `max_chars=30000` as planned; keep the runner as is; proceed as is with
the failing qwen calls but record how many occur during the full run; run the benchmark
as jobs rather than in the shell.

### Implemented

Added a MaxQuality and a MinTime Abacus config for CUAD under
`experiments/experiment-configs/benchmark/abacus-cuad/`, identical to the VRDU pair in
models, seed and `pz_config` except for the dataset (30000-character window, 100 train
and 100 validation contracts). Added `analysis/abacus_cuad_compare.py`, a CUAD
counterpart of `abacus_compare.py` that labels the CUAD accuracy measure and draws a
"predict nothing" baseline and a "window ceiling" computed from the data. Fixed
`experiments/run_job.sh` to put the cluster's Python headers on `CPATH`, which stopped a
vLLM start-up crash (Triton JIT needing `Python.h`) on the Gemma server. Also made a local,
gitignored change to `experiments/submit.sh` for opt-in CPU qsub jobs; it is not part of
the commit.

### Commits

d4b4a09 Add CUAD Abacus configs (MaxQuality/MinTime), CUAD analysis script, Python.h fix for jobs
