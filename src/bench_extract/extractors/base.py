"""Common contract shared by RouteExtractor, CascadeExtractor, and
ScheduleExtractor: any of the project's routing/cascade/scheduling
strategies is a way of combining a fixed list of ExtractionModel members of
M, fit from the same profiling table (see profiling.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from bench_extract.models.base import ExtractionModel
from bench_extract.utils.timing import UsageRecord


@dataclass
class ExtractionStep:
    """One model call made while producing an ExtractionResult -- including
    calls a cascade rejected or a router didn't ultimately use. Kept
    separate from the winning prediction so timing/cost analysis accounts
    for every call a strategy made, not just the one whose output was
    returned -- a cascade that rejects two cheap models before accepting a
    third has paid for all three."""

    model_key: str
    usage: UsageRecord
    accepted: bool


@dataclass
class ExtractionResult:
    field_values: dict[str, list[str]]
    steps: list[ExtractionStep]

    @property
    def total_wall_seconds(self) -> float:
        return sum(step.usage.wall_seconds for step in self.steps)


class MultiModelExtractor(ABC):
    """Base for every routing/cascade/scheduling strategy over M."""

    def __init__(self, models: list[ExtractionModel]):
        if not models:
            raise ValueError("MultiModelExtractor requires at least one model in M.")
        self.models: dict[str, ExtractionModel] = {m.key: m for m in models}
        self._fitted = False

    @abstractmethod
    def fit(self, profiling_table: pd.DataFrame) -> None:
        """Fit the strategy from a profiling table with one row per
        (document_id, field, model_key), columns `prediction`,
        `ground_truth`, `correct`, `wall_seconds`, `prompt_tokens`,
        `completion_tokens`, `error` (see profiling.PROFILING_TABLE_COLUMNS).
        Implementations must set self._fitted = True on success."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, document_text: str, fields: list[str]) -> ExtractionResult:
        raise NotImplementedError

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__}.fit() must be called before extract().")
