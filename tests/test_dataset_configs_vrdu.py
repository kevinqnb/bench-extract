import json
from pathlib import Path

import pytest

from bench_extract.datasets.base import DatasetRunContext
from experiments.config import DATASET_REGISTRY

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "vrdu_mini"

vrdu_dataset = DATASET_REGISTRY["vrdu"]


def test_load_run_context_returns_context_for_corpus():
    ctx = vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    assert isinstance(ctx, DatasetRunContext)
    assert set(ctx.all_docs) == {"doc-ad-1"}


def test_load_run_context_fields_come_from_corpus_config_not_data_json():
    ctx = vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    # The field list is CORPORA["ad-buy-form"].match_func_by_field, not
    # whatever this fixture doc happens to have values for (doc-ad-1 only
    # has 3 of the 14 configured fields).
    assert ctx.fields == sorted(vrdu_dataset.CORPORA["ad-buy-form"].match_func_by_field)
    assert len(ctx.fields) == 14
    doc = ctx.all_docs["doc-ad-1"]
    assert doc.fields.get("agency", []) == []  # configured field this doc has no value for


def test_load_run_context_raises_when_config_disagrees_with_data_json(tmp_path):
    # dataset-configs/vrdu.py's CORPORA is authoritative; a data.json whose
    # schema disagrees with it (e.g. after a fresh re-download upstream
    # changed something) must fail loud, not silently pick one.
    drifted = {
        "corpora": {"ad-buy-form": {"dataset_name": "DeepForm", "entity_name_to_match_func": {"advertiser": "GeneralStringMatch"}}},
        "documents": {},
    }
    (tmp_path / "data.json").write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="disagrees with"):
        vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=tmp_path)


def test_load_run_context_unpopulated_corpus_raises_not_implemented():
    # registration-form's match_func_by_field is a deliberate TODO (not yet
    # downloaded locally) -- see CORPORA in dataset-configs/vrdu.py.
    with pytest.raises(NotImplementedError):
        vrdu_dataset.load_run_context({"corpus": "registration-form"}, data_root=FIXTURE_ROOT)


def test_load_run_context_score_field_uses_corpus_match_funcs():
    ctx = vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    # advertiser's match func is GeneralStringMatch: equal after stripping non-alphanumerics.
    assert ctx.score_field(["Acme Corp."], ["Acme Corp"], "advertiser")
    assert not ctx.score_field(["Widget Inc"], ["Acme Corp"], "advertiser")
    # gross_amount's match func is PriceMatch: numeric equality within tolerance.
    assert ctx.score_field(["$1,250.00"], ["1250.00"], "gross_amount")


def test_load_run_context_windowed_text_applies_corpus_windowing():
    ctx = vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    doc = ctx.all_docs["doc-ad-1"]
    # This fixture doc has fewer pages than ad-buy-form's configured
    # max_pages, so windowed_text should return the whole thing unclipped.
    assert len(doc.pages) < vrdu_dataset.CORPORA["ad-buy-form"].max_pages
    assert ctx.windowed_text(doc) == doc.ocr_text


def test_load_run_context_unknown_corpus_raises():
    with pytest.raises(KeyError):
        vrdu_dataset.load_run_context({"corpus": "not-a-corpus"}, data_root=FIXTURE_ROOT)


def test_load_split_doc_ids():
    train_ids = vrdu_dataset.load_split_doc_ids({"corpus": "ad-buy-form", "split_file": "tiny-split"}, "train", data_root=FIXTURE_ROOT)
    assert train_ids == ["doc-ad-1"]
    valid_ids = vrdu_dataset.load_split_doc_ids({"corpus": "ad-buy-form", "split_file": "tiny-split"}, "valid", data_root=FIXTURE_ROOT)
    assert valid_ids == ["doc-ad-1"]


def test_describe_split_decodes_train_size_and_seed():
    desc = vrdu_dataset.describe_split({"split_file": "DeepForm-unk_template-train_100-test_215-valid_100-SD_0"})
    assert desc == {"train_size": 100, "seed": 0}


def test_describe_split_unparseable_name_raises():
    with pytest.raises(ValueError):
        vrdu_dataset.describe_split({"split_file": "not-a-vrdu-split-name"})


def test_training_split_params_expands_one_dict_per_split_file():
    dataset_params = {
        "corpus": "ad-buy-form",
        "split_files": ["a-train_10-test_1-valid_1-SD_0", "a-train_50-test_1-valid_1-SD_1"],
    }
    expanded = vrdu_dataset.training_split_params(dataset_params)
    assert len(expanded) == 2
    assert [p["split_file"] for p in expanded] == dataset_params["split_files"]
    assert all(p["corpus"] == "ad-buy-form" for p in expanded)
    assert [vrdu_dataset.describe_split(p) for p in expanded] == [
        {"train_size": 10, "seed": 0},
        {"train_size": 50, "seed": 1},
    ]
