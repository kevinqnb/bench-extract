"""Loads the reshaped VRDU dataset written by data/download_vrdu.py.

VRDU (https://arxiv.org/abs/2211.15421): FCC ad-buy forms (`ad-buy-form`,
aka DeepForm) and FARA registration forms (`registration-form`), each with
OCR text and human-annotated field spans. Original data + evaluator:
https://github.com/google-research-datasets/vrdu,
https://github.com/google-research/google-research/tree/master/vrdu.

Reads data/vrdu/data.json (produced by download_vrdu.py's reshape of the
official release) rather than dataset.jsonl directly, so nothing here has to
know about gzip/jsonl framing or the corpus-specific `main/` layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# src/bench_extract/datasets/vrdu.py -> repo root is four parents up.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "vrdu"


@dataclass(frozen=True)
class VRDUDocument:
    document_id: str
    corpus: str
    filename: str
    ocr_text: str
    pages: list[tuple[int, int]]  # char [start, end) per page, in reading order
    fields: dict[str, list[str]]  # field_name -> ground-truth text value(s)


@dataclass(frozen=True)
class CorpusSchema:
    dataset_name: str
    entity_name_to_match_func: dict[str, str]


def load_dataset(data_root: Path = DEFAULT_DATA_ROOT) -> dict:
    path = data_root / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `uv run data/download_vrdu.py` first.")
    return json.loads(path.read_text())


def load_corpus_schema(corpus: str, data_root: Path = DEFAULT_DATA_ROOT) -> CorpusSchema:
    data = load_dataset(data_root)
    if corpus not in data["corpora"]:
        raise KeyError(f"Unknown corpus {corpus!r}; choices: {sorted(data['corpora'])}")
    schema = data["corpora"][corpus]
    return CorpusSchema(dataset_name=schema["dataset_name"], entity_name_to_match_func=schema["entity_name_to_match_func"])


def load_documents(corpus: Optional[str] = None, data_root: Path = DEFAULT_DATA_ROOT) -> list[VRDUDocument]:
    """All documents, optionally filtered to one corpus (`ad-buy-form` or
    `registration-form`)."""
    data = load_dataset(data_root)
    docs = []
    for document_id, rec in data["documents"].items():
        if corpus is not None and rec["corpus"] != corpus:
            continue
        docs.append(
            VRDUDocument(
                document_id=document_id,
                corpus=rec["corpus"],
                filename=rec["filename"],
                ocr_text=rec["ocr_text"],
                pages=[tuple(p) for p in rec["pages"]],
                fields=rec["fields"],
            )
        )
    return docs


def load_split(corpus: str, split_name: str, data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, list[str]]:
    """One few_shot-splits/<corpus>/<split_name>.json -> {'train': [...],
    'valid': [...], 'test': [...]} of document ids. The split file itself
    stores raw `<uuid>.pdf` filenames; translated to document_id via the
    `.pdf` stem, matching how download_vrdu.py derives document_id."""
    path = data_root / "few_shot-splits" / corpus / f"{split_name}.json"
    raw = json.loads(path.read_text())
    return {part: [Path(name).stem for name in raw[part]] for part in ("train", "valid", "test")}


def windowed_text(doc: VRDUDocument, max_pages: int = 3, max_chars: int = 9000) -> str:
    """First `max_pages` pages, capped at `max_chars` -- mirrors
    govscape_extract.documents.extraction_window's reasoning: extraction
    fields are front-loaded, and this bounds LLM cost and small-model
    truncation. Falls back to a flat character cap if page metadata is
    missing or `max_pages` exceeds what the document has."""
    if doc.pages:
        end = doc.pages[min(max_pages, len(doc.pages)) - 1][1]
        text = doc.ocr_text[:end]
    else:
        text = doc.ocr_text
    return text[:max_chars]
