"""Builds the profiling table every RouteExtractor/CascadeExtractor/
ScheduleExtractor.fit() consumes: one row per (document_id, field,
model_key), scored against ground truth with the field's VRDU match
function (datasets.matching).

This is deliberately the single evaluation code path -- experiments/
run_benchmark.py's accuracy numbers and every method's *fit* both read
`correct` from here, so a router's training signal and its reported
benchmark accuracy can never silently disagree.

Simplification vs. the official VRDU evaluator: this scores one predicted
value against the flat list of a field's ground-truth text occurrences,
regardless of `entity_appearance_pattern` (`unrepeated` vs. `line_item`) --
enough to drive routing/cascade/scheduling decisions, but not the official
line-item-aware micro/macro F1. Use the official `vrdu.evaluate` for a
paper-comparable number.

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

import pandas as pd

from bench_extract.datasets import matching
from bench_extract.datasets.vrdu import CorpusSchema, VRDUDocument

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


def score_prediction(predicted_values: list[str], ground_truth_values: list[str], match_func_name: str) -> bool:
    """True if any one of `predicted_values` matches any one of
    `ground_truth_values` under the named match function."""
    if not predicted_values or not ground_truth_values:
        return False
    match_cls = matching.MATCH_FUNCS.get(match_func_name, matching.DefaultMatch)
    return any(match_cls.match(pred, ground_truth_values) for pred in predicted_values)


def build_profiling_table(
    run_predictions: dict[str, dict[str, dict[str, list[str]]]],
    documents: dict[str, VRDUDocument],
    schema: CorpusSchema,
    timing: dict[str, dict[str, dict]],
) -> pd.DataFrame:
    """
    run_predictions: model_key -> document_id -> field -> predicted value(s)
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
            for field, match_func_name in schema.entity_name_to_match_func.items():
                predicted = field_values.get(field, [])
                ground_truth = doc.fields.get(field, [])
                rows.append(
                    {
                        "document_id": document_id,
                        "field": field,
                        "model_key": model_key,
                        "prediction": predicted,
                        "ground_truth": ground_truth,
                        "correct": score_prediction(predicted, ground_truth, match_func_name),
                        "wall_seconds": doc_timing.get("wall_seconds"),
                        "prompt_tokens": doc_timing.get("prompt_tokens"),
                        "completion_tokens": doc_timing.get("completion_tokens"),
                        "error": doc_timing.get("error"),
                    }
                )
    return pd.DataFrame(rows, columns=PROFILING_TABLE_COLUMNS)
