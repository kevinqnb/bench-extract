"""RouteLLM (https://arxiv.org/pdf/2406.18665): trains a router (several
variants -- similarity-weighted ranking, matrix factorization, a BERT
classifier) on human preference data to route between a strong and a weak
model while controlling cost.

Code: https://github.com/ulab-uiuc/LLMRouter/tree/main/llmrouter/models/mfrouter

Not wired in yet. Wiring this in means depending on `llmrouter` (add a
`routellm` extra to pyproject.toml, isolated from the other adapters'
dependencies per the note there) and adapting its router-training loop to
consume `profiling_table` (bench_extract.profiling) instead of preference
pairs. Until then, `RouteExtractor`'s always-best-model baseline is
available directly via `RouteExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.route_extractor import RouteExtractor


class RouteLLMExtractor(RouteExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "RouteLLMExtractor.fit: wire in ulab-uiuc/LLMRouter's mfrouter training "
            "here (see this module's docstring)."
        )
