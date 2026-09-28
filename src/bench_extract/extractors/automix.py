"""Automix (https://arxiv.org/abs/2310.12963): uses a small model's own
self-verification (few-shot, POMDP-based) to decide whether to trust its
answer or escalate to a larger model, without training a separate router.

Code: https://github.com/ulab-uiuc/LLMRouter/tree/main/llmrouter/models/automix

Not wired in yet. Wiring this in means depending on `llmrouter` (add an
`automix` extra to pyproject.toml, isolated from the other adapters'
dependencies per the note there) and adapting its self-verification /
POMDP acceptance logic to run over `profiling_table` (bench_extract.profiling)
-- Automix's "fit" is largely calibrating the POMDP's transition/reward
estimates from held-out self-verification outcomes, not a supervised
classifier. Until then, `CascadeExtractor`'s fixed-order/non-empty-field
baseline is available directly via `CascadeExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.cascade_extractor import CascadeExtractor


class AutomixExtractor(CascadeExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "AutomixExtractor.fit: wire in ulab-uiuc/LLMRouter's automix "
            "self-verification/POMDP calibration here (see this module's docstring)."
        )
