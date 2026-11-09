"""Loads the reshaped CUAD dataset written by data/download_cuad.py.

CUAD (https://arxiv.org/abs/2103.06268): 510 commercial legal contracts,
each annotated by legal experts with verbatim clause spans for 41 categories.
Original data: https://zenodo.org/records/4595826 (CC BY 4.0). Original code,
official train/test split and evaluator: https://github.com/TheAtticusProject/cuad.

Reads data/cuad/data.json (produced by download_cuad.py) rather than the
SQuAD-format CUAD_v1.json directly. Unlike VRDU there is no page structure
(the text is a flat string), so windowing is a plain character cap.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# src/bench_extract/datasets/cuad.py -> repo root is four parents up.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "cuad"


@dataclass(frozen=True)
class CUADDocument:
    document_id: str  # NFC-normalized contract title
    title: str
    text: str
    fields: dict[str, list[str]]  # field slug -> ground-truth clause span(s); [] if the contract has none


def load_dataset(data_root: Path = DEFAULT_DATA_ROOT) -> dict:
    path = data_root / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `uv run data/download_cuad.py` first.")
    return json.loads(path.read_text())


def load_categories(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, str]:
    """field slug -> CUAD category name, as recorded in data.json."""
    return load_dataset(data_root)["categories"]


def load_documents(data_root: Path = DEFAULT_DATA_ROOT) -> list[CUADDocument]:
    data = load_dataset(data_root)
    return [
        CUADDocument(document_id=document_id, title=rec["title"], text=rec["text"], fields=rec["fields"])
        for document_id, rec in data["documents"].items()
    ]


def load_official_split(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, list[str]]:
    """The Atticus Project's official split: {'train': [408 ids], 'test': [102 ids]}."""
    path = data_root / "splits" / "official.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `uv run data/download_cuad.py` first.")
    return json.loads(path.read_text())


def resolve_split(
    official: dict[str, list[str]],
    valid_source: str,
    train_size: Optional[int],
    seed: int,
    valid_size: Optional[int] = None,
) -> dict[str, list[str]]:
    """Official split -> {'train': [...], 'valid': [...]}.

    CUAD ships only train/test, but the runners read 'train' and 'valid':
      valid_source="test":  valid = the official test set; train = official train.
      valid_source="train": valid = `valid_size` contracts held out (seeded) from the
                            official train set, leaving the official test set untouched;
                            train = the rest.
    Either way, `train_size` (None = all) then subsamples train, seeded.
    Subsampling is nested: a smaller train_size is a prefix of a larger one under
    the same seed.
    """
    rng = random.Random(seed)
    train = sorted(official["train"])
    if valid_source == "test":
        valid = sorted(official["test"])
    elif valid_source == "train":
        if valid_size is None or not 0 < valid_size < len(train):
            raise ValueError(f"valid_size must be in (0, {len(train)}) when valid_source='train', got {valid_size!r}")
        shuffled = train[:]
        rng.shuffle(shuffled)
        valid, train = sorted(shuffled[:valid_size]), shuffled[valid_size:]
    else:
        raise ValueError(f"valid_source must be 'train' or 'test', got {valid_source!r}")
    if train_size is not None:
        if not 0 < train_size <= len(train):
            raise ValueError(f"train_size {train_size} out of range (1..{len(train)})")
        pool = train[:]
        random.Random(seed).shuffle(pool)
        train = sorted(pool[:train_size])
    return {"train": train, "valid": valid}


def windowed_text(doc: CUADDocument, max_chars: Optional[int]) -> str:
    """First `max_chars` characters of the contract (None = the whole text).
    CUAD has no page metadata, and clauses are spread through the whole
    contract, so unlike VRDU's front-loaded forms a short window loses
    ground truth -- see experiments/dataset-configs/cuad.py."""
    return doc.text if max_chars is None else doc.text[:max_chars]
