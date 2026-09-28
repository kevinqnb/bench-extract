"""Unified routing/cascading (https://proceedings.mlr.press/v267/dekoninck25a.html):
frames routing and cascading as instances of one general "cascade-routing"
optimization problem and solves for the jointly optimal policy, rather than
treating them as separate strategy families.

Code: https://github.com/eth-sri/cascade-routing

Subclasses CascadeExtractor rather than RouteExtractor: cascade-routing's
policy generalizes cascading (sequential escalation) and only degenerates
to pure routing (a single one-shot model choice) as a special case, so the
cascade base class's `model_order` + `should_accept` control flow is the
closer fit, even though a fitted policy may in practice terminate after one
step for many documents.

Not wired in yet. Wiring this in means depending on `cascade-routing` (add
a `cascade_routing` extra to pyproject.toml, isolated from the other
adapters' dependencies per the note there) and adapting its joint
optimization to consume `profiling_table` (bench_extract.profiling). Until
then, `CascadeExtractor`'s fixed-order/non-empty-field baseline is
available directly via `CascadeExtractor(models).fit(profiling_table)`.
"""

from __future__ import annotations

import pandas as pd

from bench_extract.extractors.cascade_extractor import CascadeExtractor


class CascadeRoutingExtractor(CascadeExtractor):
    def fit(self, profiling_table: pd.DataFrame) -> None:
        raise NotImplementedError(
            "CascadeRoutingExtractor.fit: wire in eth-sri/cascade-routing's joint "
            "optimization here (see this module's docstring)."
        )
