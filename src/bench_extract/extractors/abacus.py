"""Abacus (Palimpzest) (https://arxiv.org/abs/2505.14661): estimates each
candidate model/operator's cost, latency, and quality on a training sample,
then optimizes a physical execution plan over those estimates -- a
declarative-query-optimizer take on model assignment.

Code: https://github.com/mitdbg/palimpzest

Wired in via palimpzest's `sem_map` + sentinel (sample-based) optimizer; all
pz-touching code lives in `extractors/utils/pz_support.py` so this module
imports without the `abacus` extra (`uv sync --extra abacus`).

- `fit()` does NOT read the profiling table: pz samples candidate operators
  itself, on the training documents handed over by `bind_run_context`
  (the dataset's own train split -- pz never splits or holds out anything),
  scoring them with a Validator (default: the dataset's `score_field`
  against ground truth; see pz_support.ScoreFieldValidator). The optimization
  phase's LLM spend is reported via `fit_metrics`.
- `extract_many()` runs the optimized plan over a batch of documents.
  Unlike the other strategies the prompts are pz's, not the dataset's
  EXTRACTION_PROMPT_TEMPLATE, and a plan may be a mixture-of-agents or critic
  pipeline rather than one model per field (`pz_config` flags
  `allow_mixtures`, `allow_critic`, `allow_model_selection`, ... restrict it).
- Candidate models are the benchmark's M; each is an OpenAI-compatible
  endpoint reached through litellm (`hosted_vllm/<served name>`). pz refuses
  more than one such model per run; pz_support.allow_multiple_vllm steps
  around that check.

Constructor kwargs (all set under `method_kwargs.abacus` in the experiment
config -- nothing here is hardcoded):
    policy:      {name: <pz_support.POLICIES key>, **policy_args}, e.g.
                 {name: MaxQuality} or {name: MinCostAtFixedQuality, min_quality: 0.8}
    pz_config:   QueryProcessorConfig fields passed through verbatim
                 (sample_budget, seed, k, j, allow_mixtures, ...)
    validator:   pz_support.VALIDATORS key or "package.module:Class"
    instruction: the sem_map's natural-language task description
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from bench_extract.datasets.base import DatasetRunContext
from bench_extract.extractors.base import ExtractionResult, ExtractionStep
from bench_extract.extractors.schedule_extractor import ScheduleExtractor
from bench_extract.utils.timing import UsageRecord

DEFAULT_INSTRUCTION = "Extract the requested fields from the document text."


class AbacusExtractor(ScheduleExtractor):
    def __init__(
        self,
        models,
        *,
        policy: Optional[dict[str, Any]] = None,
        pz_config: Optional[dict[str, Any]] = None,
        validator: str = "score_field",
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        super().__init__(models)
        self.policy_spec = dict(policy or {"name": "MaxQuality"})
        self.pz_config = dict(pz_config or {})
        self.validator_name = validator
        self.instruction = instruction
        self._ctx: Optional[DatasetRunContext] = None
        self._train_docs: list = []
        self._program = None  # pz_support.OptimizedProgram, set by fit()
        self._pz_model_key: dict[str, str] = {}  # pz model id -> our model_key
        self._fields: list[str] = []

    def bind_run_context(self, ctx: DatasetRunContext, train_docs: list) -> None:
        self._ctx = ctx
        self._train_docs = train_docs

    def fit(self, profiling_table: pd.DataFrame) -> None:
        """`profiling_table` is intentionally unused -- see module docstring."""
        if self._ctx is None or not self._train_docs:
            raise RuntimeError("AbacusExtractor.fit needs bind_run_context(ctx, train_docs) with training documents first.")
        from palimpzest.query.processor.config import QueryProcessorConfig

        from bench_extract.extractors.utils import pz_support as pzs

        ctx = self._ctx
        self._fields = list(ctx.fields)
        columns = [
            {
                "name": f,
                "desc": ctx.field_descriptions.get(f, f.replace("_", " ")),
                "type": list[str],
            }
            for f in self._fields
        ]
        pz_models = []
        for key, model in self.models.items():
            kwargs: dict[str, Any] = {}
            if getattr(model, "max_tokens", None) is not None:
                kwargs["max_tokens"] = model.max_tokens
            if getattr(model, "extra_body", None):
                kwargs["extra_body"] = model.extra_body
            # pz otherwise sends $VLLM_API_KEY / a dummy key; use this endpoint's own.
            kwargs["api_key"] = model.client.api_key
            pz_model = pzs.make_pz_model(model.model, str(model.client.base_url).rstrip("/"), **kwargs)
            self._pz_model_key[pz_model.model_id] = key
            pz_models.append(pz_model)

        policy_args = {k: v for k, v in self.policy_spec.items() if k != "name"}
        reserved = {"policy", "available_models"} & set(self.pz_config)
        if reserved:
            raise ValueError(f"pz_config must not set {sorted(reserved)}; they come from `policy` and the model keys.")
        config = QueryProcessorConfig(
            **{
                "progress": False,
                **self.pz_config,
                "policy": pzs.build_policy(self.policy_spec["name"], **policy_args),
                "available_models": pz_models,
            }
        )
        ground_truth = {d.document_id: d.fields for d in self._train_docs}
        validator = pzs.resolve_validator_cls(self.validator_name)(ground_truth, ctx.score_field)
        train_rows = [
            {pzs.DOCUMENT_ID_COLUMN: d.document_id, pzs.CONTENTS_COLUMN: ctx.windowed_text(d)}
            for d in self._train_docs
        ]
        self._program = pzs.OptimizedProgram(columns, self.instruction, train_rows, config, validator)
        stats = self._program.optimize()
        self.fit_metrics = {
            "optimization_seconds": stats.optimization_time,
            "optimization_cost_usd": stats.optimization_cost,
            "optimization_input_tokens": stats.sum_plan_stats_field("input_text_tokens"),  # finish() never runs for the optimization-only pass
            "optimization_output_tokens": stats.sum_plan_stats_field("output_text_tokens"),
            "n_train_documents": len(train_rows),
        }
        self._fitted = True

    def extract(self, document_text: str, fields: list[str]) -> ExtractionResult:
        return self.extract_many([document_text], fields)[0]

    def extract_many(self, document_texts: list[str], fields: list[str]) -> list[ExtractionResult]:
        self._require_fitted()
        unknown = set(fields) - set(self._fields)
        if unknown:
            raise KeyError(f"AbacusExtractor was fit on fields {self._fields}; cannot extract {sorted(unknown)}")
        from bench_extract.extractors.utils import pz_support as pzs

        rows = [{pzs.DOCUMENT_ID_COLUMN: str(i), pzs.CONTENTS_COLUMN: t} for i, t in enumerate(document_texts)]
        records, plan_stats = self._program.run(rows)

        steps_by_source: dict[str, list[ExtractionStep]] = {}
        for op_stats in plan_stats.operator_stats.values():
            for rec in op_stats.record_op_stats_lst:
                if rec.model_name is None and not rec.total_llm_calls:  # scan / non-LLM operators
                    continue
                usage = UsageRecord(
                    wall_seconds=rec.time_per_record,
                    prompt_tokens=int(rec.input_text_tokens),
                    completion_tokens=int(rec.output_text_tokens),
                    total_tokens=int(rec.input_text_tokens + rec.output_text_tokens),
                )
                # Composite operators (mixture-of-agents, critic) span several
                # models and report no single model_name; label them by operator.
                key = self._pz_model_key.get(rec.model_name) or rec.model_name or f"pz:{rec.op_name}"
                steps_by_source.setdefault(str(rec.record_source_indices), []).append(
                    ExtractionStep(model_key=key, usage=usage, accepted=True)
                )

        by_doc: dict[str, ExtractionResult] = {}
        for record in records:
            values = {f: pzs.as_str_list(getattr(record, f, None)) for f in fields}
            steps = steps_by_source.get(str(record._source_indices), [])
            by_doc[record.document_id] = ExtractionResult(field_values=values, steps=steps)
        # pz drops a document whose generation failed outright; score it as empty.
        return [
            by_doc.get(str(i), ExtractionResult(field_values={f: [] for f in fields}, steps=[]))
            for i in range(len(document_texts))
        ]

