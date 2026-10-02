---
id: 2026-10-02-vrdu-analysis-01
config:
---

## Session 2026-10-02

### Prompts

"Make some plots to describe the VRDU dataset. Build these into a new script
`experiments/vrdu_analysis.py`"

### Implemented

Added `experiments/vrdu_analysis.py`, a descriptive (non-benchmark) script, so it
has no experiment config. `--corpus <key>` produces four matplotlib figures plus a
`summary.json` under `experiments/results/analysis/vrdu/<corpus>/`: document length
(pages and OCR characters vs. the prompt window), field coverage, mean annotated
spans per field, and where each field's ground truth sits relative to the OCR
window. Fields, match functions and window size come from `CORPORA` in
`experiments/dataset-configs/vrdu.py`. Run on `ad-buy-form` (641 documents) and
inspected the rendered figures; no unit tests were added. The window check uses the
character limit only, not the page limit, and `registration-form` is unsupported
until it is downloaded.

### Commits

17ce5df Add vrdu_analysis.py: descriptive plots of VRDU
