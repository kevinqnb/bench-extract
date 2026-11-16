"""Palimpzest (pz) plumbing for AbacusLinker: the join's datasets, its
models and config, and the workarounds pz 1.5.x needs. Builds on
extractors/utils/pz_support.py and leaves that module untouched
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Iterator

import pandas as pd
from palimpzest.constants import Model
from palimpzest.core.data.dataset import Dataset
from palimpzest.core.data.iter_dataset import MemoryDataset
from palimpzest.core.lib.schemas import ImageFilepath
from palimpzest.core.models import PlanStats
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.query.processor.query_processor_factory import (
    QueryProcessorFactory,
)
from palimpzest.validator.validator import Validator

from bench_extract.extractors.utils import pz_support
from bench_extract.linkers.base import (
    IMAGE_COLUMN,
    IMAGE_ID_COLUMN,
    RECORD_COLUMN,
    RECORD_ID_COLUMN,
    LinkModel,
)

IMAGES_DATASET_ID = "images"
RECORDS_DATASET_ID = "records"
# Tells pz the image column holds file paths to load, not text
IMAGE_SCHEMA = [
    {"name": IMAGE_ID_COLUMN, "type": str, "desc": "Identifier of the image"},
    {"name": IMAGE_COLUMN, "type": ImageFilepath, "desc": "The image"},
]
PROMPT_COLUMNS = [IMAGE_COLUMN, RECORD_COLUMN]

# Config fields this module sets itself
RESERVED_CONFIG_FIELDS = {"policy", "available_models"}
TOGETHER_KEY_ENV = "TOGETHER_API_KEY"
TOGETHER_KEY_PLACEHOLDER = "unused-clip-runs-locally"

class GoldPairValidator(Validator):
    """Scores a join decision for Abacus: 1.0 when the match / no-match
    call agrees with the gold pairs. Without it pz asks a paid LLM judge"""
    def __init__(self, gold_pairs: set[tuple[str, str]]):
        super().__init__()
        self.gold_pairs = gold_pairs

    def join_score_fn(
        self,
        condition: str,
        left_input_record: dict,
        right_input_record: dict,
        output: bool,
    ) -> float:
        pair = (
            left_input_record[IMAGE_ID_COLUMN],
            right_input_record[RECORD_ID_COLUMN],
        )
        is_gold_pair = pair in self.gold_pairs
        return float(output == is_gold_pair)

def make_vision_model(model: LinkModel) -> Model:
    """The pz Model for one vision LLM endpoint"""
    litellm_kwargs: dict[str, Any] = {"api_key": model.api_key}
    if model.max_tokens is not None:
        litellm_kwargs["max_tokens"] = model.max_tokens
    if model.extra_body:
        litellm_kwargs["extra_body"] = model.extra_body
    pz_model = pz_support.make_pz_model(
        model.served_name, model.base_url, **litellm_kwargs
    )
    # pz looks vision support up by model id and does not know ours
    pz_model.model_specs["is_vision_model"] = True
    return pz_model

def build_config(
    models: list[LinkModel],
    embedding_model: str | None,
    policy: dict[str, Any],
    pz_config: dict[str, Any],
) -> QueryProcessorConfig:
    """pz's run settings: the policy, the models it may use, and the
    experiment config's pz_config fields"""
    reserved = RESERVED_CONFIG_FIELDS & set(pz_config)
    if reserved:
        raise ValueError(
            f"pz_config must not set {sorted(reserved)}; they come from "
            "`policy` and the model keys"
        )
    available_models = []
    if embedding_model is not None:
        available_models.append(Model(embedding_model))
    for model in models:
        available_models.append(make_vision_model(model))
    policy_args = {
        name: value for name, value in policy.items() if name != "name"
    }
    # No progress bars in log.txt unless pz_config asks for them
    settings = {"progress": False, **pz_config}
    return QueryProcessorConfig(
        policy=pz_support.build_policy(policy["name"], **policy_args),
        available_models=available_models,
        **settings,
    )

def make_datasets(
    images: pd.DataFrame, records: pd.DataFrame
) -> dict[str, MemoryDataset]:
    return {
        IMAGES_DATASET_ID: MemoryDataset(
            IMAGES_DATASET_ID, images, schema=IMAGE_SCHEMA
        ),
        RECORDS_DATASET_ID: MemoryDataset(RECORDS_DATASET_ID, records),
    }

def make_join(datasets: dict[str, MemoryDataset], condition: str) -> Dataset:
    """The pz program: images sem_join records"""
    return datasets[IMAGES_DATASET_ID].sem_join(
        datasets[RECORDS_DATASET_ID], condition, depends_on=PROMPT_COLUMNS
    )

@contextlib.contextmanager
def local_clip_passes_key_check() -> Iterator[None]:
    """pz files clip-ViT-B-32 under provider together_ai and so demands
    TOGETHER_API_KEY, though it runs CLIP locally. Sets a placeholder while
    pz checks; nothing is sent to Together"""
    if os.environ.get(TOGETHER_KEY_ENV):
        yield
        return
    os.environ[TOGETHER_KEY_ENV] = TOGETHER_KEY_PLACEHOLDER
    try:
        yield
    finally:
        del os.environ[TOGETHER_KEY_ENV]

def run_join(
    condition: str,
    images: pd.DataFrame,
    records: pd.DataFrame,
    config: QueryProcessorConfig,
) -> tuple[list, PlanStats]:
    """Runs the join as pz ships it, with no Abacus optimization; returns
    (joined records, plan_stats)"""
    program = make_join(make_datasets(images, records), condition)
    with local_clip_passes_key_check(), \
            pz_support.count_vllm_input_tokens():
        output = program.run(config)
    (plan_stats,) = output.execution_stats.plan_stats.values()
    return output.data_records, plan_stats

class OptimizedJoin(pz_support.OptimizedProgram):
    """The join with Abacus: `optimize()` once on labeled training tables,
    then `run()` the chosen plan on new tables"""
    def __init__(
        self,
        condition: str,
        train_images: pd.DataFrame,
        train_records: pd.DataFrame,
        config: QueryProcessorConfig,
        validator: Validator,
    ):
        n_train_pairs = len(train_images) * len(train_records)
        if config.sample_budget > n_train_pairs:
            # pz samples until the budget is spent and never stops when the
            # training tables run out first
            raise ValueError(
                f"pz_config.sample_budget ({config.sample_budget}) exceeds "
                f"the {n_train_pairs} training pairs; pz would never finish"
            )
        self.train_datasets = make_datasets(train_images, train_records)
        self.program = make_join(self.train_datasets, condition)
        self.program._generate_unique_logical_op_ids()
        with local_clip_passes_key_check(), \
                pz_support.allow_multiple_vllm():
            self.processor = QueryProcessorFactory.create_processor(
                self.program, config, self.train_datasets, validator
            )
        self.enforce_types = config.enforce_types
        self.cost_model = None
        self.optimization_stats = None

    def run(
        self, images: pd.DataFrame, records: pd.DataFrame
    ) -> tuple[list, PlanStats]:
        """Runs the optimized plan; returns (joined records, plan_stats)"""
        if self.cost_model is None:
            raise RuntimeError("optimize() must be called before run()")
        run_datasets = make_datasets(images, records)
        # pz relaxes the program's types unless told to enforce them, and
        # new datasets must match (see pz_support.OptimizedProgram.run)
        if not self.enforce_types:
            for run_dataset in run_datasets.values():
                run_dataset.relax_types()
        program = self.program.copy()
        program._set_root_datasets(run_datasets)
        program._generate_unique_logical_op_ids()
        optimizer = self.processor.optimizer.deepcopy_clean()
        optimizer.update_cost_model(self.cost_model)
        with pz_support.count_vllm_input_tokens():
            joined, plan_stats = self.processor._execute_best_plan(
                program, optimizer
            )
        return joined, plan_stats[0]
