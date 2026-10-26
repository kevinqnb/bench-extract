---
id: 2026-10-26-cuad-dataset-01
config:
---

## Session 2026-10-26

### Prompts

"Please help me set up a new dataset to be used within this repo. Specifically, I
want to use the CUAD [https://github.com/The-Atticus-Project/cuad] dataset for
information extraction. Please look into the source repository to see where we can
find the documents, text files, labeled annotations, and accuracy scoring scheme.
If they have pre-split training and testing sets that would be helpful as well.
This should all be built into the `data/cuad/` directory (storing data files) and
an `experiments/dataset-configs/cuad.py` file defining extraction fields / schemas
and accuracy scoring rules. Follow the implementation for the `vrdu` dataset as an
example. If any piece is not available, please alert me and we may discuss
solutions together."

Follow-up: "We should only keep the extraction fields which are not yes/no
questions."

### Implemented

Added CUAD as a second registered dataset: `data/download_cuad.py` pulls the
contracts and labels from Zenodo and the official train/test split from the CUAD
GitHub repo into the gitignored `data/cuad/`; `src/bench_extract/datasets/cuad.py`
loads it (seeded validation holdout and train-size subsampling, since CUAD ships only
train/test), and `datasets/cuad_matching.py` ports the upstream Jaccard match rule as
a per-field correct/incorrect check (the upstream confidence-based AUPR metrics need
per-span probabilities LLM extraction doesn't provide). `experiments/dataset-configs/cuad.py`
declares the 10 non-yes/no categories as fields, lists the 31 excluded ones so the
`data.json` drift guard stays strict, and defines the prompt template. Added unit
tests with a small fixture; the full suite passed, and ground truth scored as 100%
correct on every real (contract, field) pair. No experiment config yet -- `dataset_params`
(`valid_source`, `valid_size`, `train_size`, `seed`, `max_chars`) are documented in the
module docstring.

### Commits

- `c15f6c7` Add CUAD dataset: downloader, loader, ported scorer, dataset config
