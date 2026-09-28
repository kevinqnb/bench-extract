"""Abacus (Palimpzest) (https://arxiv.org/abs/2505.14661): estimates each
candidate model/operator's cost, latency, and quality on a training sample,
then optimizes a physical execution plan over those estimates -- a
declarative-query-optimizer take on model assignment.

Code: https://github.com/mitdbg/palimpzest

Not wired in yet. Wiring this in means depending on `palimpzest` (add an
`abacus` extra to pyproject.toml, isolated from the other adapters'
dependencies per the note there) and adapting its plan-optimization step to
consume `profiling_table` (bench_extract.profiling) as the per-operator
cost/quality sample it optimizes over. Until then, `ScheduleExtractor`'s
greedy cheapest-above-accuracy-floor baseline is available directly via
`ScheduleExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.schedule_extractor import ScheduleExtractor


class AbacusExtractor(ScheduleExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "AbacusExtractor.fit: wire in mitdbg/palimpzest's plan optimization "
            "here (see this module's docstring)."
        )
