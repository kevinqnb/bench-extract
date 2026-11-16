"""Common contract for semantic-join strategies: given a table of images and
a table of records, return the (image, record) pairs judged to match
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Columns of the images table (id, path to the image file) and the records
# table (id, record text) that every linker joins
IMAGE_ID_COLUMN = "image_id"
IMAGE_COLUMN = "image"
RECORD_ID_COLUMN = "record_id"
RECORD_COLUMN = "record"
# How a join condition names the two sides
IMAGE_PLACEHOLDER = "{" + IMAGE_COLUMN + "}"
RECORD_PLACEHOLDER = "{" + RECORD_COLUMN + "}"

@dataclass(frozen=True)
class LinkModel:
    """One OpenAI-compatible endpoint serving one model in M"""
    key: str
    served_name: str
    base_url: str
    api_key: str
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

@dataclass
class LinkResult:
    """Matched (image_id, record_id) pairs and the system's own token counts"""
    pairs: list[tuple[str, str]]
    prompt_tokens: int
    completion_tokens: int
    #: What ran, in the system's own words (e.g. the physical plan)
    plan: str = ""

class Linker(ABC):
    """Base for every semantic-join strategy
    """
    def __init__(
        self,
        models: list[LinkModel],
        embedding_model: str | None,
        join_condition: str,
    ):
        if not models:
            raise ValueError("Linker requires at least one model")
        for placeholder in (IMAGE_PLACEHOLDER, RECORD_PLACEHOLDER):
            if placeholder not in join_condition:
                raise ValueError(f"join_condition must contain {placeholder}")
        self.models = models
        self.embedding_model = embedding_model
        self.join_condition = join_condition
        self.fit_metrics: dict[str, float] = {}

    def fit(
        self,
        images: pd.DataFrame,
        records: pd.DataFrame,
        gold_pairs: set[tuple[str, str]],
    ) -> None:
        """Optimize on labeled training tables. Default: nothing to fit"""

    @abstractmethod
    def link(
        self, images: pd.DataFrame, records: pd.DataFrame
    ) -> LinkResult:
        raise NotImplementedError
