"""HybridLLM (https://openreview.net/pdf?id=02f3mUtqnM): trains a
BERT-style router to predict whether a small model's answer to a query will
be acceptable, and only escalates to a large model when it predicts not.

Code: https://github.com/ulab-uiuc/LLMRouter/tree/main/llmrouter/models/hybrid_llm

Not wired in yet. Wiring this in means depending on `llmrouter` (add a
`hybridllm` extra to pyproject.toml, isolated from the other adapters'
dependencies per the note there) and adapting its router-training loop to
consume `profiling_table` (bench_extract.profiling) instead of its own data
format. Until then, `RouteExtractor`'s always-best-model baseline is
available directly via `RouteExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.route_extractor import RouteExtractor


class HybridLLMExtractor(RouteExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "HybridLLMExtractor.fit: wire in ulab-uiuc/LLMRouter's hybrid_llm router "
            "training here (see this module's docstring)."
        )
