"""Everything in AbacusExtractor that touches `palimpzest` (pz), isolated
here so `import bench_extract.extractors.abacus` works without the `abacus`
extra installed (pz drags in torch/chromadb/gradio/...; see pyproject.toml).
Import this module only from inside AbacusExtractor's methods.

Pinned to the pz version the shims below were written against: pz 1.5.x's
`QueryProcessorFactory` rejects >1 vLLM model per run (see
`allow_multiple_vllm`), a check we have to step around to benchmark several
locally-served models at once.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.metadata
from typing import Any, Callable, Iterator

from palimpzest import policy as pz_policy
from palimpzest.constants import Model
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.data.iter_dataset import IterDataset
from palimpzest.core.models import ExecutionStats, PlanStats
from palimpzest.prompts.prompt_manager import PromptManager
from palimpzest.query.optimizer.cost_model import SampleBasedCostModel
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.query_processor_factory import QueryProcessorFactory
from palimpzest.validator.validator import Validator

SUPPORTED_PZ_VERSION_PREFIX = "1.5."

POLICIES: dict[str, type] = {
    "MaxQuality": pz_policy.MaxQuality,
    "MinCost": pz_policy.MinCost,
    "MinTime": pz_policy.MinTime,
    "MaxQualityAtFixedCost": pz_policy.MaxQualityAtFixedCost,
    "MaxQualityAtFixedTime": pz_policy.MaxQualityAtFixedTime,
    "MinCostAtFixedQuality": pz_policy.MinCostAtFixedQuality,
    "MinTimeAtFixedQuality": pz_policy.MinTimeAtFixedQuality,
}

DOCUMENT_ID_COLUMN = "document_id"
CONTENTS_COLUMN = "contents"


def build_policy(name: str, **kwargs: Any) -> pz_policy.Policy:
    """`name` is a key of POLICIES; `kwargs` are that policy's constructor
    args (`max_cost` / `max_time` / `min_quality`, per policy)."""
    if name not in POLICIES:
        raise KeyError(f"Unknown palimpzest policy {name!r}; choices: {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)


class DocumentDataset(IterDataset):
    """Root pz dataset over already-loaded documents: one row per document,
    `{document_id, contents}`. Takes text (not a path) so it sees exactly the
    dataset's own windowed text -- the same input every other method gets."""

    def __init__(self, id: str, rows: list[dict[str, str]]):
        self._rows = rows
        super().__init__(
            id=id,
            schema=[
                {"name": DOCUMENT_ID_COLUMN, "type": str, "desc": "Identifier of the source document"},
                {"name": CONTENTS_COLUMN, "type": str, "desc": "OCR text of the document"},
            ],
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        return self._rows[idx]


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


class ScoreFieldValidator(Validator):
    """Default pz Validator: scores each extracted field with the active
    dataset's own `score_field` (DatasetRunContext.score_field -- e.g. VRDU's
    ported match functions) against that document's ground truth, and
    returns the mean over the fields pz is currently scoring. Without this,
    pz falls back to an LLM judge (o4-mini, paid); `map_score_fn` being
    overridden is what prevents that.

    Custom validators subclass this (or pz's Validator) and are selected by
    `method_kwargs.abacus.validator` in the experiment config; the
    constructor signature `(ground_truth, score_field)` is the contract."""

    def __init__(
        self,
        ground_truth: dict[str, dict[str, list[str]]],
        score_field: Callable[[list[str], list[str], str], bool],
    ):
        super().__init__()
        self.ground_truth = ground_truth
        self.score_field = score_field

    def map_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        truth = self.ground_truth.get(input_record[DOCUMENT_ID_COLUMN])
        if truth is None or not fields:
            return None
        scores = [
            float(self.score_field(as_str_list(output.get(f)), truth.get(f, []), f))
            for f in fields
        ]
        return sum(scores) / len(scores)


VALIDATORS: dict[str, type[Validator]] = {"score_field": ScoreFieldValidator}


def resolve_validator_cls(name: str) -> type[Validator]:
    """`name` is a VALIDATORS key, or a `package.module:ClassName` path to a
    custom Validator subclass with the (ground_truth, score_field) ctor."""
    if name in VALIDATORS:
        return VALIDATORS[name]
    if ":" not in name:
        raise KeyError(f"Unknown validator {name!r}; choices: {sorted(VALIDATORS)} or 'package.module:Class'")
    module_name, cls_name = name.split(":", 1)
    return getattr(importlib.import_module(module_name), cls_name)


def make_pz_model(served_name: str, base_url: str, **litellm_kwargs: Any) -> Model:
    """A pz Model for one OpenAI-compatible endpoint (a `vllm serve`).
    litellm routes `hosted_vllm/<served_name>` to `base_url`; `litellm_kwargs`
    (max_tokens, extra_body, ...) ride along on every completion call.
    pz's registry/equality key on the model id, so each endpoint needs a
    distinct served name even when the weights are identical."""
    return Model(f"hosted_vllm/{served_name}", api_base=base_url, **litellm_kwargs)


@contextlib.contextmanager
def allow_multiple_vllm() -> Iterator[None]:
    """pz's `QueryProcessorFactory._config_validation_and_normalization`
    raises "Only one vLLM model can be used per run" if >1 available model
    has an `api_base`. Nothing downstream is single-model (api_base lives on
    each Model and is read per-call in the generators), so we hide the vLLM
    flag from just that check by flipping it off for the duration of the
    call, restoring it right after."""
    pz_version = importlib.metadata.version("palimpzest")
    if not pz_version.startswith(SUPPORTED_PZ_VERSION_PREFIX):
        raise RuntimeError(
            f"allow_multiple_vllm was written against palimpzest {SUPPORTED_PZ_VERSION_PREFIX}x, found {pz_version}; "
            "re-verify QueryProcessorFactory._config_validation_and_normalization before using."
        )
    original = QueryProcessorFactory._config_validation_and_normalization.__func__

    def patched(cls, config, train_dataset, validator):
        models = [m for m in (config.available_models or []) if isinstance(m, Model)]
        hidden = [m for m in models if m.model_specs.get("is_vllm_model")]
        for m in hidden:
            m.model_specs["is_vllm_model"] = False
        try:
            return original(cls, config, train_dataset, validator)
        finally:
            for m in hidden:
                m.model_specs["is_vllm_model"] = True

    QueryProcessorFactory._config_validation_and_normalization = classmethod(patched)
    try:
        yield
    finally:
        QueryProcessorFactory._config_validation_and_normalization = classmethod(original)


@contextlib.contextmanager
def count_vllm_input_tokens() -> Iterator[None]:
    """pz 1.5.x never records input tokens for vLLM models
    (`PromptManager.extract_usage_stats`'s vLLM branch is a bare `pass`),
    which would understate Abacus's token usage in an efficiency comparison.
    Fills `input_text_tokens` from the response's `prompt_tokens` while
    active; output tokens are already recorded by pz."""
    original = PromptManager.extract_usage_stats

    def patched(self, usage, is_audio_op):
        stats = original(self, usage, is_audio_op)
        if self.model.is_vllm_model() and not stats["input_text_tokens"]:
            stats["input_text_tokens"] = usage.get("prompt_tokens") or 0
        return stats

    PromptManager.extract_usage_stats = patched
    try:
        yield
    finally:
        PromptManager.extract_usage_stats = original


#: Shared root-dataset id: pz swaps the train dataset in for the run dataset
#: by matching root ids, so both must carry the same one.
ROOT_DATASET_ID = "documents"


class OptimizedProgram:
    """Splits pz's `Dataset.optimize_and_run` (optimize on train, then run on
    the dataset, in one call) into `optimize()` once and `run()` many times,
    so `AbacusExtractor.fit` really fits and inference never re-optimizes.
    Mirrors `QueryProcessor.execute` in pz 1.5.x (private API, hence the
    version pin in `allow_multiple_vllm`)."""

    def __init__(
        self,
        columns: list[dict],
        desc: str,
        train_rows: list[dict[str, str]],
        config: QueryProcessorConfig,
        validator: Validator,
    ):
        self.train_ds = DocumentDataset(ROOT_DATASET_ID, train_rows)
        self.program = self.train_ds.sem_map(columns, desc=desc)
        self.program._generate_unique_logical_op_ids()
        with allow_multiple_vllm():
            self.processor = QueryProcessorFactory.create_processor(
                self.program, config, {ROOT_DATASET_ID: self.train_ds}, validator
            )
        self.enforce_types = config.enforce_types
        self.cost_model: SampleBasedCostModel | None = None
        self.optimization_stats: ExecutionStats | None = None

    def optimize(self) -> ExecutionStats:
        p = self.processor
        stats = ExecutionStats(execution_id=p.execution_id())
        stats.start()
        sentinel_plan = p._create_sentinel_plan(p.train_dataset)
        with count_vllm_input_tokens():
            sentinel_stats = p.sentinel_execution_strategy.execute_sentinel_plan(sentinel_plan, p.train_dataset, p.validator)
        stats.add_plan_stats(sentinel_stats)
        stats.finish_optimization()
        self.cost_model = SampleBasedCostModel(sentinel_stats, p.verbose)
        self.optimization_stats = stats
        return stats

    def run(self, rows: list[dict[str, str]]) -> tuple[list, PlanStats]:
        """Executes the optimized plan on `rows`; returns (records, plan_stats)."""
        if self.cost_model is None:
            raise RuntimeError("OptimizedProgram.optimize() must be called before run().")
        p = self.processor
        dataset: Dataset = self.program.copy()
        run_ds = DocumentDataset(ROOT_DATASET_ID, rows)
        # create_processor relaxes types on the program + train dataset when
        # not enforcing types; a fresh run dataset must match or schema unions fail.
        if not self.enforce_types:
            run_ds.relax_types()
        dataset._set_root_datasets({ROOT_DATASET_ID: run_ds})
        dataset._generate_unique_logical_op_ids()
        optimizer = p.optimizer.deepcopy_clean()
        optimizer.update_cost_model(self.cost_model)
        with count_vllm_input_tokens():
            records, plan_stats = p._execute_best_plan(dataset, optimizer)
        return records, plan_stats[0]
