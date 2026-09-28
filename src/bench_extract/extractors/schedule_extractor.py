"""ScheduleExtractor: profile every model in M on a small training set, then
solve a static assignment of models to fields ahead of time -- unlike
Route/CascadeExtractor, no per-document decision happens at inference.

Concrete methods (Abacus/Palimpzest, Doctopus) subclass this and replace
`fit`'s optimizer with the paper's actual assignment procedure. The
baseline below is a real, if naive, solver: for each field, assign the
cheapest (lowest mean wall-clock time) model whose training-set accuracy on
that field clears `min_accuracy` -- falling back to the field's most
accurate model if none clears the floor.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.base import ExtractionResult, ExtractionStep, MultiModelExtractor


class ScheduleExtractor(MultiModelExtractor):
    #: Abacus/Doctopus assign per attribute, not per document.
    assignment_granularity = "field"

    def __init__(self, models, min_accuracy: float = 0.0):
        super().__init__(models)
        self.min_accuracy = min_accuracy
        self._assignment: dict[str, str] = {}  # field -> model_key

    def fit(self, profiling_table: pd.DataFrame) -> None:
        profile = (
            profiling_table.groupby(["field", "model_key"])
            .agg(accuracy=("correct", "mean"), mean_wall_seconds=("wall_seconds", "mean"))
            .reset_index()
        )
        self._assignment = {}
        for field, rows in profile.groupby("field"):
            eligible = rows[rows["accuracy"] >= self.min_accuracy]
            candidates = eligible if not eligible.empty else rows.sort_values("accuracy", ascending=False).head(1)
            best = candidates.sort_values("mean_wall_seconds").iloc[0]
            self._assignment[field] = best["model_key"]
        self._fitted = True

    def extract(self, document_text: str, fields: list[str]) -> ExtractionResult:
        self._require_fitted()
        steps: list[ExtractionStep] = []
        field_values: dict[str, list[str]] = {}
        # Group fields by their assigned model so each model is called once.
        fields_by_model: dict[str, list[str]] = {}
        for f in fields:
            model_key = self._assignment.get(f)
            if model_key is None:
                raise KeyError(f"No schedule assignment for field {f!r}; was fit() run with this field present?")
            fields_by_model.setdefault(model_key, []).append(f)
        for model_key, model_fields in fields_by_model.items():
            prediction = self.models[model_key].extract(document_text, model_fields)
            steps.append(ExtractionStep(model_key=model_key, usage=prediction.usage, accepted=True))
            field_values.update(prediction.field_values)
        return ExtractionResult(field_values=field_values, steps=steps)
