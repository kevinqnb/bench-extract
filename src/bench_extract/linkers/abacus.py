"""The image <-> record join in Palimpzest (pz), optionally optimized by
Abacus (https://arxiv.org/abs/2505.14661). Needs `--extra abacus`
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from palimpzest.core.models import PlanStats
from palimpzest.query.processor.config import QueryProcessorConfig

from bench_extract.linkers.base import (
    IMAGE_COLUMN,
    IMAGE_ID_COLUMN,
    IMAGE_PLACEHOLDER,
    RECORD_COLUMN,
    RECORD_ID_COLUMN,
    RECORD_PLACEHOLDER,
    Linker,
    LinkModel,
    LinkResult,
)
from bench_extract.linkers.utils import pz_join_support

def to_pz_condition(join_condition: str) -> str:
    """pz takes the condition as a plain sentence: each placeholder is
    replaced by words, "{image}" by "the image\""""
    return join_condition.replace(
        IMAGE_PLACEHOLDER, f"the {IMAGE_COLUMN}"
    ).replace(RECORD_PLACEHOLDER, f"the {RECORD_COLUMN}")

def to_pairs(joined_rows: list) -> list[tuple[str, str]]:
    """(image_id, record_id) of every row pz joined"""
    return [
        (getattr(row, IMAGE_ID_COLUMN), getattr(row, RECORD_ID_COLUMN))
        for row in joined_rows
    ]

def count_tokens(plan_stats: PlanStats) -> tuple[int, int]:
    """(prompt, completion) tokens pz recorded across the plan"""
    prompt_tokens, completion_tokens = 0, 0
    for operator_stats in plan_stats.operator_stats.values():
        for record_stats in operator_stats.record_op_stats_lst:
            prompt_tokens += int(record_stats.input_text_tokens)
            completion_tokens += int(record_stats.output_text_tokens)
    return prompt_tokens, completion_tokens

class AbacusLinker(Linker):
    """Set under `method_kwargs.abacus` in the experiment config:
        optimize:  True has fit() run Abacus on the training tables; False
                   has link() run the join as pz ships it
        policy:    {name: <pz_support.POLICIES key>, **policy_args}
        pz_config: pz QueryProcessorConfig fields (seed, sample_budget, ...)
    """
    def __init__(
        self,
        models: list[LinkModel],
        embedding_model: str | None,
        join_condition: str,
        *,
        optimize: bool,
        policy: dict[str, Any],
        pz_config: dict[str, Any],
    ):
        super().__init__(models, embedding_model, join_condition)
        self.optimize = optimize
        self.policy = dict(policy)
        self.pz_config = dict(pz_config)
        self.pz_condition = to_pz_condition(join_condition)
        self.optimized_join: pz_join_support.OptimizedJoin | None = None

    def build_config(self) -> QueryProcessorConfig:
        return pz_join_support.build_config(
            self.models, self.embedding_model, self.policy, self.pz_config
        )

    def fit(
        self,
        images: pd.DataFrame,
        records: pd.DataFrame,
        gold_pairs: set[tuple[str, str]],
    ) -> None:
        """With optimize on, Abacus picks the join's plan by trying the
        candidates on the training tables and scoring them against gold"""
        if not self.optimize:
            return
        self.optimized_join = pz_join_support.OptimizedJoin(
            self.pz_condition,
            images,
            records,
            self.build_config(),
            pz_join_support.GoldPairValidator(gold_pairs),
        )
        stats = self.optimized_join.optimize()
        self.fit_metrics = {
            "optimization_seconds": stats.optimization_time,
            "optimization_input_tokens": stats.sum_plan_stats_field(
                "input_text_tokens"
            ),
            "optimization_output_tokens": stats.sum_plan_stats_field(
                "output_text_tokens"
            ),
            "n_train_images": len(images),
            "n_train_records": len(records),
        }

    def link(
        self, images: pd.DataFrame, records: pd.DataFrame
    ) -> LinkResult:
        if not self.optimize:
            joined_rows, plan_stats = pz_join_support.run_join(
                self.pz_condition, images, records, self.build_config()
            )
        elif self.optimized_join is None:
            raise RuntimeError("AbacusLinker.fit() must be called first")
        else:
            joined_rows, plan_stats = self.optimized_join.run(images, records)
        prompt_tokens, completion_tokens = count_tokens(plan_stats)
        return LinkResult(
            pairs=to_pairs(joined_rows),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            plan=plan_stats.plan_str or "",
        )
