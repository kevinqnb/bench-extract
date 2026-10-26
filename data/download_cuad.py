#!/usr/bin/env python3
"""Downloads and reshapes the CUAD benchmark into data/cuad/.

CUAD: https://arxiv.org/abs/2103.06268 (CC BY 4.0 data)
Data (full contracts, PDFs, txt, labels): https://zenodo.org/records/4595826 (CUAD_v1.zip)
Code + official train/test split + category descriptions:
       https://github.com/TheAtticusProject/cuad (data.zip, category_descriptions.csv)
Eval:  the GitHub repo's evaluate.py (bench_extract.datasets.cuad_matching ports
       its match rule)

Two sources are needed because neither has everything: Zenodo has the PDFs and
contract text but no train/test split; the GitHub repo's data.zip has the
split (train_separate_questions.json / test.json, by contract title) but no
PDFs. The labels in Zenodo's CUAD_v1.json and the GitHub repo's CUADv1.json
are byte-identical (checked at download time).

Always downloads fresh into a temporary directory, then reshapes into this
project's layout -- one flat data.json keyed by document_id instead of
SQuAD-format nested json. See data/README.md for the produced layout.

Usage:
    uv run data/download_cuad.py
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
import zipfile
from pathlib import Path

CUAD_REPO_URL = "https://github.com/TheAtticusProject/cuad.git"
ZENODO_ZIP_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
ZENODO_ZIP_MD5 = "c38f490a984420b8a62600db401fafd5"
DATA_ROOT = Path(__file__).resolve().parent / "cuad"


def normalize_title(title: str) -> str:
    """document_id: the contract title, NFC-normalized. Zenodo's txt/pdf
    filenames and the json titles can differ in Unicode normalization form
    (e.g. 'LECLANCHE' with a precomposed vs. combining accent)."""
    return unicodedata.normalize("NFC", title)


def slugify(category: str) -> str:
    """'Rofr/Rofo/Rofn' -> 'rofr_rofo_rofn' (field keys that are safe in metrics.json)."""
    return re.sub(r"[^0-9a-z]+", "_", category.lower()).strip("_")


def download_zenodo(dest: Path) -> Path:
    zip_path = dest / "CUAD_v1.zip"
    print(f"Downloading {ZENODO_ZIP_URL} ...")
    # curl rather than urllib: uv-managed Pythons often lack the system CA bundle.
    subprocess.run(["curl", "-sSL", "--fail", "-o", str(zip_path), ZENODO_ZIP_URL], check=True)
    md5 = hashlib.md5(zip_path.read_bytes())
    if md5.hexdigest() != ZENODO_ZIP_MD5:
        raise RuntimeError(f"CUAD_v1.zip md5 {md5.hexdigest()} != expected {ZENODO_ZIP_MD5}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest / "zenodo")
    return dest / "zenodo" / "CUAD_v1"


def clone_github(dest: Path) -> tuple[Path, str]:
    print(f"Cloning {CUAD_REPO_URL} ...")
    subprocess.run(["git", "clone", "--depth", "1", CUAD_REPO_URL, str(dest / "repo")], check=True)
    sha = subprocess.run(
        ["git", "-C", str(dest / "repo"), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    with zipfile.ZipFile(dest / "repo" / "data.zip") as zf:
        zf.extractall(dest / "github_data")
    return dest / "github_data", sha


def reshape(zenodo_dir: Path, github_data_dir: Path, github_sha: str) -> None:
    squad = json.loads((zenodo_dir / "CUAD_v1.json").read_text(encoding="utf-8"))
    if (zenodo_dir / "CUAD_v1.json").read_bytes() != (github_data_dir / "CUADv1.json").read_bytes():
        raise RuntimeError("Zenodo CUAD_v1.json and GitHub CUADv1.json differ; expected identical labels.")

    categories: dict[str, str] = {}  # slug -> CUAD category name (as in the qa ids)
    documents: dict[str, dict] = {}
    for contract in squad["data"]:
        document_id = normalize_title(contract["title"])
        if document_id in documents:
            raise RuntimeError(f"Duplicate document_id after NFC normalization: {document_id!r}")
        if "/" in document_id:
            raise RuntimeError(f"document_id {document_id!r} contains '/', unusable as a filename")
        (para,) = contract["paragraphs"]
        fields: dict[str, list[str]] = {}
        for qa in para["qas"]:
            # id is '<title>__<Category>'; the title itself may not contain '__'
            # reliably, so split from the right.
            category = qa["id"].rsplit("__", 1)[1]
            slug = slugify(category)
            if categories.setdefault(slug, category) != category:
                raise RuntimeError(f"Category slug collision on {slug!r}: {categories[slug]!r} vs {category!r}")
            fields[slug] = [a["text"] for a in qa["answers"]]
        documents[document_id] = {"title": contract["title"], "text": para["context"], "fields": fields}

    def title_ids(path: Path) -> list[str]:
        ids = [normalize_title(c["title"]) for c in json.loads(path.read_text(encoding="utf-8"))["data"]]
        unknown = [i for i in ids if i not in documents]
        if unknown:
            raise RuntimeError(f"{path.name}: titles not in CUAD_v1.json: {unknown[:3]}")
        return ids

    split = {
        "train": title_ids(github_data_dir / "train_separate_questions.json"),
        "test": title_ids(github_data_dir / "test.json"),
    }
    if set(split["train"]) & set(split["test"]) or set(split["train"]) | set(split["test"]) != set(documents):
        raise RuntimeError("Official train/test split does not partition CUAD_v1.json's contracts.")

    # Cross-check test.json's answers against the full label file (the split
    # files are what evaluate.py scores against). Compared as sorted lists:
    # test.json lists the same answers in a different order for some questions
    # (order is irrelevant to evaluate.py's matching).
    test_squad = json.loads((github_data_dir / "test.json").read_text(encoding="utf-8"))
    for contract in test_squad["data"]:
        doc = documents[normalize_title(contract["title"])]
        for qa in contract["paragraphs"][0]["qas"]:
            slug = slugify(qa["id"].rsplit("__", 1)[1])
            if sorted(a["text"] for a in qa["answers"]) != sorted(doc["fields"][slug]):
                raise RuntimeError(f"test.json answers disagree with CUAD_v1.json for {qa['id']!r}")

    txt_dir = DATA_ROOT / "txt"
    pdf_dir = DATA_ROOT / "pdfs"
    split_dir = DATA_ROOT / "splits"
    for d in (txt_dir, pdf_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)
    for document_id, doc in documents.items():
        (txt_dir / f"{document_id}.txt").write_text(doc["text"], encoding="utf-8")

    # PDFs live in nested Part_I/Part_II/Part_III/<contract type>/ folders with
    # mixed .pdf/.PDF extensions; match to documents by NFC-normalized stem,
    # ignoring the stray trailing spaces/dashes a few upstream filenames have.
    def pdf_key(stem: str) -> str:
        return normalize_title(stem).strip().rstrip("- ")

    pdf_by_key = {pdf_key(p.stem): p for p in (zenodo_dir / "full_contract_pdf").rglob("*") if p.suffix.lower() == ".pdf"}
    unmatched = []
    for document_id in documents:
        src = pdf_by_key.get(pdf_key(document_id))
        if src is None:
            unmatched.append(document_id)
        else:
            shutil.copy2(src, pdf_dir / f"{document_id}.pdf")
    print(f"PDFs matched to documents by title: {len(documents) - len(unmatched)}/{len(documents)}; no PDF for: {unmatched}")

    (split_dir / "official.json").write_text(json.dumps(split))
    # Provenance copy: category names + descriptions + answer formats, used to
    # author experiments/dataset-configs/cuad.py's field_descriptions.
    shutil.copy2(github_data_dir.parent / "repo" / "category_descriptions.csv", DATA_ROOT / "category_descriptions.csv")
    shutil.copy2(zenodo_dir / "master_clauses.csv", DATA_ROOT / "master_clauses.csv")
    (DATA_ROOT / "data.json").write_text(
        json.dumps(
            {
                "source": {"zenodo_md5": ZENODO_ZIP_MD5, "github_commit": github_sha},
                "categories": categories,
                "documents": documents,
            }
        )
    )
    print(
        f"Wrote {len(documents)} documents, {len(categories)} categories, "
        f"split train={len(split['train'])} test={len(split['test'])} to {DATA_ROOT}"
    )


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zenodo_dir = download_zenodo(tmp_path)
        github_data_dir, sha = clone_github(tmp_path)
        reshape(zenodo_dir, github_data_dir, sha)


if __name__ == "__main__":
    main()
