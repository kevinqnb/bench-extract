"""Builds the profiling table every RouteExtractor/CascadeExtractor/
ScheduleExtractor.fit() consumes: one row per (document_id, field,
model_key), scored against ground truth via the active dataset's own
`score_field` (part of the DatasetRunContext every experiments/
dataset-configs/<key>.py module returns -- see bench_extract.datasets.base).
This module never imports a specific dataset's matching logic -- what
"correct" means is entirely that dataset's business.

This is deliberately the single evaluation code path -- experiments/
run_benchmark.py's accuracy numbers and every method's *fit* both read
`correct` from here, so a router's training signal and its reported
benchmark accuracy can never silently disagree.

Simplification vs. the official VRDU evaluator: for the `vrdu` dataset, this
scores one predicted value against the flat list of a field's ground-truth
text occurrences, regardless of `entity_appearance_pattern` (`unrepeated` vs.
`line_item`) -- enough to drive routing/cascade/scheduling decisions, but not
the official line-item-aware micro/macro F1. Use the official `vrdu.evaluate`
for a paper-comparable number. (The opt-in unified measure,
bench_extract.scoring.unified, does score VRDU line items as records and
unrepeated fields as single entities, but as a per-cell soft F1 averaged over
cells, not the official corpus-pooled F1.)

Timing caveat: experiments/run_benchmark.py's `_profile_models` makes ONE
`extract(text, fields)` call per (document, model) covering every field at
once, so `wall_seconds`/`prompt_tokens`/`completion_tokens` are identical
across every field row for a given (document_id, model_key) -- this table
does not (yet) carry true per-field cost. `ScheduleExtractor`'s
`mean_wall_seconds` grouped by `(field, model_key)` is therefore really
"mean per-document cost of a model that happens to have been asked for this
field among others," not that field's isolated cost -- fine for picking a
model per field, but do not sum `wall_seconds` across a document's field
rows (it overcounts by the number of fields). Getting real per-field cost
means profiling one call per field instead.
"""

from __future__ import annotations

import math

import pandas as pd

from bench_extract.datasets.base import DatasetRunContext

PROFILING_TABLE_COLUMNS = [
    "document_id",
    "field",
    "model_key",
    "prediction",
    "ground_truth",
    "correct",
    "wall_seconds",
    "prompt_tokens",
    "completion_tokens",
    "error",
]


def build_profiling_table(
    run_predictions: dict[str, dict[str, dict[str, list[str]]]],
    ctx: DatasetRunContext,
    timing: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """
    run_predictions: model_key -> document_id -> extraction field -> predicted value(s)
    ctx:             the active DatasetRunContext -- documents (`all_docs`), the
                      scored fields, and the measure's `prediction_for` /
                      `ground_truth` / `score_field` (native or unified)
    timing:          model_key -> document_id -> {'wall_seconds', 'prompt_tokens',
                      'completion_tokens', 'error'} (see utils.timing.UsageRecord)

    Returns a DataFrame with PROFILING_TABLE_COLUMNS, one row per
    (document_id, scored field, model_key) actually present in
    `run_predictions`. `field` is a *scored* field (ctx.scored_fields: under the
    unified measure VRDU's `line_items` is one record row; under the native one
    its five sub-fields are five rows).
    `correct` is a float score in [0, 1] -- 0/1 under a native measure, a soft
    F1 under the unified one -- or NaN for a cell the measure does not score
    (unified: both sides empty). NaN rows are kept so the table stays one row
    per (document, field, model) for timing; pandas' mean() skips them, so
    every strategy's accuracy aggregates only over scored cells.
    """
    rows = []
    for model_key, by_doc in run_predictions.items():
        for document_id, field_values in by_doc.items():
            doc = ctx.all_docs[document_id]
            doc_timing = timing.get(model_key, {}).get(document_id, {})
            for field in ctx.scored_fields:
                predicted = ctx.prediction_for(field_values, field)
                ground_truth = ctx.ground_truth(doc, field)
                score = ctx.score_field(predicted, ground_truth, field)
                rows.append(
                    {
                        "document_id": document_id,
                        "field": field,
                        "model_key": model_key,
                        "prediction": predicted,
                        "ground_truth": ground_truth,
                        "correct": math.nan if score is None else float(score),
                        "wall_seconds": doc_timing.get("wall_seconds"),
                        "prompt_tokens": doc_timing.get("prompt_tokens"),
                        "completion_tokens": doc_timing.get("completion_tokens"),
                        "error": doc_timing.get("error"),
                    }
                )
    return pd.DataFrame(rows, columns=PROFILING_TABLE_COLUMNS)
