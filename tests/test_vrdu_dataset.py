from pathlib import Path

import pytest

from bench_extract.datasets import vrdu

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "vrdu_mini"


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vrdu.load_dataset(data_root=tmp_path)


def test_load_corpus_schema():
    schema = vrdu.load_corpus_schema("ad-buy-form", data_root=FIXTURE_ROOT)
    assert schema.dataset_name == "DeepForm"
    assert schema.entity_name_to_match_func == {
        "advertiser": "GeneralStringMatch",
        "gross_amount": "PriceMatch",
        "flight_from": "DateMatch",
    }


def test_load_corpus_schema_unknown_corpus():
    with pytest.raises(KeyError):
        vrdu.load_corpus_schema("not-a-corpus", data_root=FIXTURE_ROOT)


def test_load_documents_all():
    docs = vrdu.load_documents(data_root=FIXTURE_ROOT)
    assert {d.document_id for d in docs} == {"doc-ad-1", "doc-reg-1"}


def test_load_documents_filtered_by_corpus():
    docs = vrdu.load_documents(corpus="registration-form", data_root=FIXTURE_ROOT)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.document_id == "doc-reg-1"
    assert doc.corpus == "registration-form"
    assert doc.fields["registration_num"] == ["4821"]
    assert doc.pages == [(0, 47)]


def test_load_split():
    splits = vrdu.load_split("ad-buy-form", "tiny-split", data_root=FIXTURE_ROOT)
    assert splits == {"train": ["doc-ad-1"], "valid": ["doc-ad-1"], "test": []}


def test_windowed_text_respects_page_boundary():
    doc = vrdu.load_documents(corpus="ad-buy-form", data_root=FIXTURE_ROOT)[0]
    # max_pages=1 should cut exactly at the end of page 1 (char 77), dropping "Page 2\nmore stuff".
    windowed = vrdu.windowed_text(doc, max_pages=1, max_chars=9000)
    assert windowed == doc.ocr_text[:77]
    assert "more stuff" not in windowed


def test_windowed_text_respects_max_chars():
    doc = vrdu.load_documents(corpus="ad-buy-form", data_root=FIXTURE_ROOT)[0]
    windowed = vrdu.windowed_text(doc, max_pages=2, max_chars=10)
    assert windowed == doc.ocr_text[:10]
