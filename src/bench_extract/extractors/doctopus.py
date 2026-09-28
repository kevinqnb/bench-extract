"""Doctopus (https://dl.acm.org/doi/10.1145/3722212.3725103): profiles
extraction models per document-attribute pair and solves an assignment
that meets an accuracy target at minimum cost, targeted specifically at
document (form/PDF) information extraction -- the closest prior work to
this project's own setting.

Code: https://github.com/mutong184/Doctopus

Not wired in yet. Wiring this in means depending on Doctopus's package (add
a `doctopus` extra to pyproject.toml, isolated from the other adapters'
dependencies per the note there) and adapting its per-attribute profiling
and assignment solver to consume `profiling_table` (bench_extract.profiling)
directly, since it already matches Doctopus's own (document, attribute,
model) granularity. Until then, `ScheduleExtractor`'s greedy
cheapest-above-accuracy-floor baseline is available directly via
`ScheduleExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.schedule_extractor import ScheduleExtractor


class DoctopusExtractor(ScheduleExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "DoctopusExtractor.fit: wire in mutong184/Doctopus's per-attribute "
            "profiling and assignment solver here (see this module's docstring)."
        )
