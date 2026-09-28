"""Generic dataset-adapter contract: what an experiments/dataset-configs/<key>.py
module must provide so run_benchmark.py / run_training.py can drive any
dataset through the same code path, selected by an experiment config's
params["dataset"] rather than a hardcoded import. See dataset-configs/vrdu.py
for the reference implementation and experiments/config.py's
load_dataset_registry() for how a module gets discovered and validated.

A dataset module needs:
    EXTRACTION_PROMPT_TEMPLATE: str
    load_run_context(dataset_params: dict) -> DatasetRunContext
    load_split_doc_ids(dataset_params: dict, part: str) -> list[str]
        -- part is "train" or "valid"; returns document ids
    describe_split(dataset_params: dict) -> dict
        -- scalar labels (e.g. {"train_size": 100, "seed": 0}) used to
           namespace metrics.json keys when one run sweeps multiple
           dataset_params (run_training.py's train-size ladder)
    training_split_params(dataset_params: dict) -> list[dict]
        -- expands one experiment config's dataset_params into one
           dataset_params dict per sweep point run_training.py should loop
           over (each one valid input to load_run_context / describe_split /
           load_split_doc_ids); how a dataset names/enumerates its sweep is
           entirely that dataset's business, not the runner's
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DatasetRunContext:
    fields: list[str]
    all_docs: dict[str, Any]  # document_id -> Document; must expose .document_id and .fields
    windowed_text: Callable[[Any], str]  # doc -> extraction-window text, windowing already bound
    score_field: Callable[[list[str], list[str], str], bool]  # (predicted, ground_truth, field) -> correct
