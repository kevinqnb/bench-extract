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
for a paper-comparable number.

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

from typing import Any, Callable

import pandas as pd

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
    documents: dict[str, Any],
    fields: list[str],
    score_field: Callable[[list[str], list[str], str], bool],
    timing: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """
    run_predictions: model_key -> document_id -> field -> predicted value(s)
    documents:       document_id -> Document (must expose `.fields: dict[str, list[str]]`)
    score_field:     DatasetRunContext.score_field -- (predicted, ground_truth, field) -> correct
    timing:          model_key -> document_id -> {'wall_seconds', 'prompt_tokens',
                      'completion_tokens', 'error'} (see utils.timing.UsageRecord)

    Returns a DataFrame with PROFILING_TABLE_COLUMNS, one row per
    (document_id, field, model_key) actually present in `run_predictions`.
    """
    rows = []
    for model_key, by_doc in run_predictions.items():
        for document_id, field_values in by_doc.items():
            doc = documents[document_id]
            doc_timing = timing.get(model_key, {}).get(document_id, {})
            for field in fields:
                predicted = field_values.get(field, [])
                ground_truth = doc.fields.get(field, [])
                rows.append(
                    {
                        "document_id": document_id,
                        "field": field,
                        "model_key": model_key,
                        "prediction": predicted,
                        "ground_truth": ground_truth,
                        "correct": score_field(predicted, ground_truth, field),
                        "wall_seconds": doc_timing.get("wall_seconds"),
                        "prompt_tokens": doc_timing.get("prompt_tokens"),
                        "completion_tokens": doc_timing.get("completion_tokens"),
                        "error": doc_timing.get("error"),
                    }
                )
    return pd.DataFrame(rows, columns=PROFILING_TABLE_COLUMNS)
