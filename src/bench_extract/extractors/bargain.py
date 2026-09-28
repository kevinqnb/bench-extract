"""BARGAIN (https://arxiv.org/abs/2509.02896): a cascade that calibrates
per-stage acceptance thresholds using proxy-model agreement and
statistically bounded sampling, aimed at semantic (LLM-as-judge-style)
extraction/filtering tasks.

Code: https://github.com/ucbepic/BARGAIN

Not wired in yet. Wiring this in means depending on BARGAIN's package (add
a `bargain` extra to pyproject.toml, isolated from the other adapters'
dependencies per the note there) and adapting its threshold-calibration
procedure to consume `profiling_table` (bench_extract.profiling). Until
then, `CascadeExtractor`'s fixed-order/non-empty-field baseline is
available directly via `CascadeExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.cascade_extractor import CascadeExtractor


class BargainExtractor(CascadeExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "BargainExtractor.fit: wire in ucbepic/BARGAIN's threshold calibration "
            "here (see this module's docstring)."
        )
