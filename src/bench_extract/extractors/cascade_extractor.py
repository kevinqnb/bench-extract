"""CascadeExtractor: escalate from a small to a larger model until a
quality threshold is met.

Concrete cascade methods (FrugalGPT, BARGAIN, Task Cascades, Automix)
subclass this and replace `fit`/`should_accept` with the paper's actual
acceptance rule (a trained scorer, confidence threshold, etc.). The
baseline below escalates through `model_order` (as declared, cheapest
first by convention) and accepts the first prediction where every
requested field came back non-empty -- no real quality judgment, just a
real, testable control-flow skeleton.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.base import ExtractionResult, ExtractionStep, MultiModelExtractor


class CascadeExtractor(MultiModelExtractor):
    #: Cascade acceptance is decided per field (a field can still be missing
    #: after a model call that satisfied every other field).
    assignment_granularity = "field"

    def __init__(self, models, model_order: list[str] | None = None):
        super().__init__(models)
        self._model_order = model_order or list(self.models)
        if set(self._model_order) != set(self.models):
            raise ValueError("model_order must contain exactly the keys in `models`.")

    def fit(self, profiling_table: pd.DataFrame) -> None:
        # Baseline keeps the constructor's declared order; a real method
        # reorders/thresholds from measured cost vs. accuracy in the table.
        self._fitted = True

    def should_accept(self, field_values: dict[str, list[str]], fields: list[str]) -> bool:
        """Escalate if any requested field is still missing. Subclasses
        override this with the paper's actual acceptance rule."""
        return all(field_values.get(f) for f in fields)

    def extract(self, document_text: str, fields: list[str]) -> ExtractionResult:
        self._require_fitted()
        steps: list[ExtractionStep] = []
        field_values: dict[str, list[str]] = {}
        for model_key in self._model_order:
            prediction = self.models[model_key].extract(document_text, fields)
            accepted = self.should_accept(prediction.field_values, fields)
            steps.append(ExtractionStep(model_key=model_key, usage=prediction.usage, accepted=accepted))
            field_values = prediction.field_values  # best-effort even if nothing ever accepts
            if accepted:
                break
        return ExtractionResult(field_values=field_values, steps=steps)
