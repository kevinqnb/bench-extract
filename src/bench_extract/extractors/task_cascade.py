"""Task Cascades (https://arxiv.org/pdf/2601.05536): cascades scoped per
*task* rather than per query, learning which stage in a model cascade is
worth invoking for a given task's accuracy/cost tradeoff.

Code: https://github.com/ucbepic/task-cascades

Not wired in yet. Wiring this in means depending on task-cascades' package
(add a `task_cascade` extra to pyproject.toml, isolated from the other
adapters' dependencies per the note there) and adapting its per-task
calibration to consume `profiling_table` (bench_extract.profiling), where
"task" maps onto this project's `field`. Until then, `CascadeExtractor`'s
fixed-order/non-empty-field baseline is available directly via
`CascadeExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.cascade_extractor import CascadeExtractor


class TaskCascadeExtractor(CascadeExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "TaskCascadeExtractor.fit: wire in ucbepic/task-cascades' per-task "
            "calibration here (see this module's docstring)."
        )
