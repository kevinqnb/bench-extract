"""A single member of M: one extraction model (LLM or smaller NLP model).

Every routing/cascade/scheduling strategy in extractors/ is built over a
list of these -- adapters never talk to a model API directly, so swapping
in a new model type (e.g. a small non-generative span extractor, mirroring
govscape_extract's GLiNER2 backend) never touches extractor code.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from bench_extract.utils.timing import UsageRecord

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class ExtractionPrediction:
    field_values: dict[str, list[str]]
    usage: UsageRecord


class ExtractionModel(ABC):
    key: str

    @abstractmethod
    def extract(self, document_text: str, fields: list[str]) -> ExtractionPrediction:
        """Extract `fields` from `document_text`, recording wall-clock time
        and token usage on the returned prediction's `usage` -- the
        profiling table and every routing/cascade/scheduling decision
        depends on that being real, not estimated after the fact."""
        raise NotImplementedError


def _loads_lenient(text: str) -> dict:
    """Tolerates ```json fences and leading/trailing preamble around the
    JSON object -- models asked for JSON reliably wrap it in commentary
    anyway (same reasoning as govscape_extract.extractors.llm.LLMExtractor)."""
    stripped = _JSON_FENCE_RE.sub("", text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class OpenAIChatModel(ExtractionModel):
    """One LLM member of M, reached through any OpenAI-compatible chat
    completions endpoint -- a hosted API or a local `vllm serve` (see
    experiments/serving.py). `prompt_template` is `.format()`-ed with
    `fields` (comma-joined) and `document_text`; the default extraction
    prompt lives in experiments/dataset-configs/vrdu.py."""

    def __init__(
        self,
        key: str,
        model: str,
        prompt_template: str,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        temperature: Optional[float] = 0.0,
        max_tokens: Optional[int] = 1024,
        extra_body: Optional[dict] = None,
        response_format: Optional[dict] = None,
    ):
        self.key = key
        self.model = model
        self.prompt_template = prompt_template
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        # None means "omit the parameter" (some hosted models 400 on it);
        # unset by the caller means the common default, sent explicitly.
        self.response_format = response_format if response_format is not None else {"type": "json_object"}

    def extract(self, document_text: str, fields: list[str]) -> ExtractionPrediction:
        prompt = self.prompt_template.format(fields=", ".join(fields), document_text=document_text)
        t0 = time.monotonic()
        error: Optional[str] = None
        field_values: dict[str, list[str]] = {}
        usage_counts: dict[str, Optional[int]] = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        try:
            kwargs = dict(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=self.extra_body,
            )
            if self.response_format is not None:
                kwargs["response_format"] = self.response_format
            response = self.client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            parsed = _loads_lenient(raw)
            for field_name in fields:
                value = parsed.get(field_name, [])
                field_values[field_name] = value if isinstance(value, list) else [str(value)]
            if response.usage:
                usage_counts = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
        except Exception as e:  # noqa: BLE001 -- one bad call shouldn't crash a whole benchmark run
            error = str(e)
        wall_seconds = time.monotonic() - t0
        return ExtractionPrediction(field_values=field_values, usage=UsageRecord(wall_seconds=wall_seconds, error=error, **usage_counts))
