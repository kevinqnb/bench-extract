"""Generic dataset-adapter contract: what an experiments/dataset-configs/<key>.py
module must provide so run_benchmark.py / run_training.py can drive any
dataset through the same code path, selected by an experiment config's
params["dataset"] rather than a hardcoded import. See dataset-configs/vrdu.py
for the reference implementation and experiments/config.py's
load_dataset_registry() for how a module gets discovered and validated.

A dataset module needs:
    EXTRACTION_PROMPT_TEMPLATE: str
    load_run_context(dataset_params: dict, scoring: dict | None = None) -> DatasetRunContext
        -- `scoring` is the experiment's raw params.scoring block (absent =>
           the dataset's native measure); parse it with
           bench_extract.scoring.unified.ScoringConfig.from_params
    load_split_doc_ids(dataset_params: dict, part: str) -> list[str]
        -- part is "train" or "valid"; returns document ids
    describe_split(dataset_params: dict) -> dict
        -- scalar labels (e.g. {"train_size": 100, "seed": 0}) used to
           namespace metrics.json keys when one run sweeps multiple
           dataset_params (run_training.py's train-size ladder)
    training_split_params(dataset_params: dict) -> list[dict]
        -- expands one experiment config's dataset_params into one
           dataset_params dict per sweep point run_training.py should loop
           over (each one valid input to load_run_context / describe_split /
           load_split_doc_ids); how a dataset names/enumerates its sweep is
           entirely that dataset's business, not the runner's

Extraction fields vs. scored fields: `fields` is what models are asked to
extract. `scored_fields` is what gets scored. Every scored field is computed
from exactly one extraction field (`extraction_field_of`), and several scored
fields may share one: VRDU's `line_items` extraction field (a list of row
objects, see `record_fields`) is one scored record field under the unified
measure, but five scored sub-fields (channel, sub_amount, ...) under the native
one. `prediction_for` turns a model's extraction-field output into what
`score_field` expects for a scored field, and `ground_truth` does the same for
a document's labels. Every consumer (profiling table, runners' evaluation,
Abacus's validator, ScheduleExtractor) goes through these, so one run never
mixes measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _flat_prediction(field_values: dict[str, list], scored_field: str) -> list:
    return field_values.get(scored_field, []) or []


def _flat_ground_truth(doc: Any, scored_field: str) -> list[str]:
    return doc.fields.get(scored_field, [])


@dataclass(frozen=True)
class DatasetRunContext:
    fields: list[str]
    all_docs: dict[str, Any]  # document_id -> Document; must expose .document_id and .fields
    windowed_text: Callable[[Any], str]  # doc -> extraction-window text, windowing already bound
    # (predicted, ground_truth, scored_field) -> score in [0, 1], or None for a
    # cell that is not scored (unified measure: both sides empty). Native
    # measures return float(bool) and never None.
    score_field: Callable[[Any, Any, str], Optional[float]]
    # field -> natural-language description, for strategies that prompt per
    # field (Abacus/Palimpzest's semantic map); missing fields fall back to
    # the bare field name.
    field_descriptions: dict[str, str] = field(default_factory=dict)
    # None => same as `fields`, each scored field its own extraction field.
    scored_fields: Optional[list[str]] = None
    extraction_field_of: Optional[dict[str, str]] = None  # scored field -> extraction field
    # extraction field whose value is a list of row objects -> its sub-fields
    # (VRDU's `line_items`); every other extraction field is a list of strings.
    record_fields: dict[str, list[str]] = field(default_factory=dict)
    prediction_for: Callable[[dict[str, list], str], Any] = _flat_prediction
    ground_truth: Callable[[Any, str], Any] = _flat_ground_truth
    # scored field -> type reported in accuracy_type_<type> breakdowns
    # (unified measure only; empty under native).
    scored_field_types: dict[str, str] = field(default_factory=dict)
    scoring_measure: str = "native"

    def __post_init__(self):
        if self.scored_fields is None:
            object.__setattr__(self, "scored_fields", list(self.fields))
        if self.extraction_field_of is None:
            object.__setattr__(self, "extraction_field_of", {f: f for f in self.scored_fields})
        sources = set(self.extraction_field_of.values())
        if set(self.extraction_field_of) != set(self.scored_fields) or sources != set(self.fields):
            raise ValueError(
                "DatasetRunContext: extraction_field_of must be keyed by scored_fields and map onto every "
                f"extraction field -- scored_fields={self.scored_fields} extraction_field_of={self.extraction_field_of} "
                f"fields={self.fields}"
            )
        if not set(self.record_fields) <= set(self.fields):
            raise ValueError(f"DatasetRunContext: record_fields {sorted(self.record_fields)} not all in fields={self.fields}")

    def scored_fields_of(self, extraction_field: str) -> list[str]:
        """The scored fields computed from `extraction_field`."""
        return [s for s in self.scored_fields if self.extraction_field_of[s] == extraction_field]
