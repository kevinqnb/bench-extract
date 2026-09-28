"""Shared timing/usage primitives for a single extraction call -- used by
every ExtractionModel implementation and read back by profiling.py, so a
model's cost/latency numbers come from one place instead of being
recomputed differently per adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class UsageRecord:
    wall_seconds: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    error: Optional[str] = None
