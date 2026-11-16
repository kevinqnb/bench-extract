"""The image <-> record join in LOTUS (https://arxiv.org/abs/2407.11418), as
a `sem_join` of two pandas DataFrames. Needs `--extra lotus`
"""

from __future__ import annotations

from typing import Any

import lotus
import pandas as pd
from lotus.dtype_extensions import ImageArray
from lotus.models import LM, SentenceTransformersRM
from lotus.types import CascadeArgs
from lotus.vector_store import FaissVS

from bench_extract.linkers.base import (
    IMAGE_COLUMN,
    IMAGE_ID_COLUMN,
    RECORD_ID_COLUMN,
    Linker,
    LinkModel,
    LinkResult,
)

VLLM_PROVIDER = "hosted_vllm"

class LotusLinker(Linker):
    """Set under `method_kwargs.lotus` in the experiment config:
        cascade_args: lotus.types.CascadeArgs fields, to let embedding
                      similarity settle the clear pairs; null sends every
                      pair to the LLM. Needs the config's embedding_model
        lm_kwargs:    extra lotus.models.LM arguments (max_ctx_len, ...)
    """
    def __init__(
        self,
        models: list[LinkModel],
        embedding_model: str | None,
        join_condition: str,
        *,
        cascade_args: dict[str, Any] | None,
        lm_kwargs: dict[str, Any],
    ):
        super().__init__(models, embedding_model, join_condition)
        if len(models) != 1:
            raise ValueError("LotusLinker joins with exactly one LLM")
        if cascade_args is not None and embedding_model is None:
            raise ValueError("cascade_args needs an embedding_model")
        self.cascade_args = cascade_args
        self.lm_kwargs = dict(lm_kwargs)

    def make_language_model(self) -> LM:
        model = self.models[0]
        extra_kwargs = dict(self.lm_kwargs)
        if model.extra_body:
            extra_kwargs["extra_body"] = model.extra_body
        return LM(
            model=f"{VLLM_PROVIDER}/{model.served_name}",
            api_base=model.base_url,
            api_key=model.api_key,
            max_tokens=model.max_tokens,
            **extra_kwargs,
        )

    def configure_lotus(self, language_model: LM) -> None:
        """LOTUS reads its models from global settings"""
        if self.embedding_model is None:
            lotus.settings.configure(lm=language_model)
            return
        lotus.settings.configure(
            lm=language_model,
            rm=SentenceTransformersRM(model=self.embedding_model),
            vs=FaissVS(),
        )

    def link(
        self, images: pd.DataFrame, records: pd.DataFrame
    ) -> LinkResult:
        language_model = self.make_language_model()
        self.configure_lotus(language_model)
        cascade_args = None
        if self.cascade_args is not None:
            cascade_args = CascadeArgs(**self.cascade_args)
        # ImageArray has LOTUS send the files' pixels, not their paths
        image_files = ImageArray(list(images[IMAGE_COLUMN]))
        joined = images.assign(**{IMAGE_COLUMN: image_files}).sem_join(
            records, self.join_condition, cascade_args=cascade_args
        )
        usage = language_model.stats.physical_usage
        pairs = zip(joined[IMAGE_ID_COLUMN], joined[RECORD_ID_COLUMN])
        return LinkResult(
            pairs=list(pairs),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
