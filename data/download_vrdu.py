#!/usr/bin/env python3
"""Downloads and reshapes the VRDU benchmark into data/vrdu/.

VRDU: https://arxiv.org/abs/2211.15421
Data:  https://github.com/google-research-datasets/vrdu
Eval:  https://github.com/google-research/google-research/tree/master/vrdu
       (bench_extract.datasets.matching ports its match functions)

Always clones fresh (no dependency on any local checkout of the vrdu repo) into
a temporary directory, then reshapes into this project's layout -- one flat
data.json keyed by document_id (the PDF filename's stem) instead of VRDU's
per-corpus dataset.jsonl, so bench_extract.datasets.vrdu doesn't need to know
about the corpus-specific `main/` layout or jsonl/gzip framing. See
data/README.md for the produced layout.

Usage:
    uv run data/download_vrdu.py [--corpus ad-buy-form registration-form]
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

VRDU_REPO_URL = "https://github.com/google-research-datasets/vrdu.git"
CORPORA = ["ad-buy-form", "registration-form"]
DATA_ROOT = Path(__file__).resolve().parent / "vrdu"


def clone_vrdu(dest: Path) -> None:
    print(f"Cloning {VRDU_REPO_URL} ...")
    subprocess.run(["git", "clone", "--depth", "1", VRDU_REPO_URL, str(dest)], check=True)


def _read_dataset_records(main_dir: Path) -> list[dict]:
    plain_path = main_dir / "dataset.jsonl"
    gz_path = main_dir / "dataset.jsonl.gz"
    if plain_path.exists():
        text = plain_path.read_text()
    elif gz_path.exists():
        text = gzip.decompress(gz_path.read_bytes()).decode("utf-8")
    else:
        raise FileNotFoundError(f"Neither {plain_path} nor {gz_path} exists.")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def reshape_corpus(corpus: str, src_root: Path) -> tuple[dict, dict]:
    """Copies one corpus's PDFs/OCR/splits into data/vrdu/ and returns
    (schema, documents) for folding into the combined data.json."""
    main_dir = src_root / corpus / "main"
    meta = json.loads((main_dir / "meta.json").read_text())
    records = _read_dataset_records(main_dir)

    pdf_out_dir = DATA_ROOT / "pdfs" / corpus
    ocr_out_dir = DATA_ROOT / "ocr" / corpus
    pdf_out_dir.mkdir(parents=True, exist_ok=True)
    ocr_out_dir.mkdir(parents=True, exist_ok=True)

    documents: dict[str, dict] = {}
    for rec in records:
        document_id = Path(rec["filename"]).stem
        src_pdf = main_dir / "pdfs" / rec["filename"]
        if src_pdf.exists():
            shutil.copy2(src_pdf, pdf_out_dir / rec["filename"])

        pages = [list(page["segment"]) for page in rec["ocr"]["pages"]]

        # annotations: [[field_name, [[text, bbox, segments], ...]], ...] for an
        # ordinary ("unrepeated") field -- ONE entity, whose occurrences are
        # its on-page appearances (alternative surface forms; upstream VRDU
        # accepts a match against any of them). For VRDU's "line_item" fields
        # (meta.json's entity_appearance_pattern, e.g. ad-buy-form's channel/
        # program_desc/program_start_date/program_end_date/sub_amount)
        # field_name is instead a *list* of the sub-field names present on
        # that row, and occurrences holds the row: (text, bbox, segments)
        # triples, one per sub-field name, in matching order. Each such
        # annotation entry is one line-item record (ad-buy-form: always
        # exactly one row per entry). Only the text is kept -- bbox/segments
        # locate the span on the page, which this project's text-only
        # extraction models never produce.
        #
        # Written three ways: `fields` (flat, every appearance -- what the
        # native VRDU matching scores against), and the grouped
        # `field_entities` / `line_items` the unified measure
        # (bench_extract.scoring.unified) scores against.
        fields: dict[str, list[str]] = {}
        field_entities: dict[str, list[list[str]]] = {}
        line_items: list[dict[str, str]] = []
        for field_name, occurrences in rec["annotations"]:
            if isinstance(field_name, list):
                for row in occurrences:
                    record = {}
                    for name, (text, _bbox, _segments) in zip(field_name, row, strict=True):
                        fields.setdefault(name, []).append(text)
                        record[name] = text
                    line_items.append(record)
            else:
                texts = [text for text, _bbox, _segments in occurrences]
                fields.setdefault(field_name, []).extend(texts)
                field_entities.setdefault(field_name, []).append(texts)

        (ocr_out_dir / f"{document_id}.json").write_text(
            json.dumps({"text": rec["ocr"]["text"], "pages": pages})
        )

        documents[document_id] = {
            "corpus": corpus,
            "filename": rec["filename"],
            "ocr_text": rec["ocr"]["text"],
            "pages": pages,
            "fields": fields,
            "field_entities": field_entities,
            "line_items": line_items,
        }

    schema = {
        "dataset_name": meta["dataset_name"],
        "entity_name_to_match_func": meta["entity_name_to_match_func"],
        # Read by the unified measure to cross-check which fields form
        # line-item records (dataset-configs/vrdu.py's line_item_fields).
        "entity_appearance_pattern": meta.get("entity_appearance_pattern"),
    }

    split_src = src_root / corpus / "few_shot-splits"
    split_out = DATA_ROOT / "few_shot-splits" / corpus
    split_out.mkdir(parents=True, exist_ok=True)
    for split_file in split_src.glob("*.json"):
        shutil.copy2(split_file, split_out / split_file.name)

    return schema, documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", nargs="+", choices=CORPORA, default=CORPORA)
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    all_corpora: dict[str, dict] = {}
    all_documents: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        src_root = Path(tmp)
        clone_vrdu(src_root)
        for corpus in args.corpus:
            print(f"Reshaping {corpus} ...")
            schema, documents = reshape_corpus(corpus, src_root)
            all_corpora[corpus] = schema
            all_documents.update(documents)

    data_json_path = DATA_ROOT / "data.json"
    combined = {"corpora": {}, "documents": {}}
    if data_json_path.exists():
        combined = json.loads(data_json_path.read_text())
    combined["corpora"].update(all_corpora)
    combined["documents"].update(all_documents)
    data_json_path.write_text(json.dumps(combined))

    print(f"Wrote {len(all_documents)} documents across {len(args.corpus)} corpus/corpora to {DATA_ROOT}")


if __name__ == "__main__":
    main()
