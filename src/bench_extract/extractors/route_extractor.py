"""RouteExtractor: train a router (typically a small classifier) to pick
ONE model in M per document, then call only that model.

Concrete routing methods (HybridLLM, RouteLLM) subclass this and replace
`fit`/`route` with their trained classifier. The baseline `fit` below
trains nothing -- it just ranks each model by mean accuracy on the
profiling table and always routes to the single best one -- so the base
class has a real, testable behavior before any adapter is filled in.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.base import ExtractionResult, ExtractionStep, MultiModelExtractor


class RouteExtractor(MultiModelExtractor):
    #: Routing methods decide per document, not per field.
    assignment_granularity = "document"

    def __init__(self, models):
        super().__init__(models)
        self._default_model_key: str | None = None

    def fit(self, profiling_table: pd.DataFrame) -> None:
        accuracy_by_model = profiling_table.groupby("model_key")["correct"].mean()
        self._default_model_key = accuracy_by_model.idxmax()
        self._fitted = True

    def route(self, document_text: str, fields: list[str]) -> str:
        """Pick one model_key in M for this document. Baseline: always the
        best model on the training profile. Subclasses override this with
        their trained router."""
        self._require_fitted()
        return self._default_model_key

    def extract(self, document_text: str, fields: list[str]) -> ExtractionResult:
        model_key = self.route(document_text, fields)
        prediction = self.models[model_key].extract(document_text, fields)
        step = ExtractionStep(model_key=model_key, usage=prediction.usage, accepted=True)
        return ExtractionResult(field_values=prediction.field_values, steps=[step])
