---
id: 2026-11-16-unified-accuracy-01
config: experiments/experiment-configs/benchmark/2026-11-09-example-unified-scoring-01.yaml
---

## Session 2026-11-16

### Prompts

Initial request (lightly trimmed): design a consistent, cross-dataset accuracy measure
for VRDU and CUAD, configurable per experiment and kept separate from the native
measures. Score per field type: dates and quantities by exact match after
normalization; free text normalized first, short spans by 1 − normalized Levenshtein,
long text by token F1; lists matched with `linear_sum_assignment`, a 0.5 similarity
threshold, and F1 over matched pairs. Remove quirks of the old measures (e.g. CUAD's
`split(" ")`). "Alert me with any discovered decision points or inconsistencies that
arise in the process."

Follow-ups (summary): use true Levenshtein, soft F1 and per-field declared types;
"Accuracy should be penalized" when exactly one side is empty, and "Both empty fields
should just be left out"; "if the original VRDU did per-record scoring for line items
then we should do the same here"; after a line-item scoring ceiling was found, "Let's go
over the options for the line item ceiling in more detail", then chose record-shaped
model output for line items; "commit and then /developlog".

### Implemented

Added `src/bench_extract/scoring/unified.py`, an opt-in measure selected by
`params.scoring` (`{measure: unified, match_threshold, quantity_decimals}`; absent means
each dataset's native rule). It scores fields by declared type, matches lists with
Hungarian matching into a soft F1, scores VRDU line items as records, and leaves cells
empty on both sides unscored. VRDU line items are now extracted as one `line_items`
field (a list of row objects); the native measure flattens them back to its 14 per-field
cells. `DatasetRunContext` carries the scoring pieces, so profiling, both runners,
`ScheduleExtractor` and Abacus's validator all use the same measure, and
`data/download_vrdu.py` now keeps VRDU's entity and line-item grouping. Example config:
`experiments/experiment-configs/benchmark/2026-11-09-example-unified-scoring-01.yaml`;
the test suite passes (186 passed, 1 skipped).

### Commits

7bca56c Add opt-in unified cross-dataset accuracy measure; extract VRDU line items as rows
