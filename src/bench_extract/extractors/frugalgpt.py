"""FrugalGPT (https://arxiv.org/abs/2305.05176): a cascade of LLMs ordered
cheap-to-expensive, with a trained scoring function per stage that decides
whether to accept a cheap model's answer or escalate.

Code: https://github.com/stanford-futuredata/Frugalgpt

Not wired in yet. Wiring this in means depending on FrugalGPT's package
(add a `frugalgpt` extra to pyproject.toml, isolated from the other
adapters' dependencies per the note there) and adapting its per-stage
scorer training to consume `profiling_table` (bench_extract.profiling).
Until then, `CascadeExtractor`'s fixed-order/non-empty-field baseline is
available directly via `CascadeExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.cascade_extractor import CascadeExtractor


class FrugalGPTExtractor(CascadeExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "FrugalGPTExtractor.fit: wire in stanford-futuredata/FrugalGPT's "
            "per-stage scorer training here (see this module's docstring)."
        )
