import json
from pathlib import Path

import pytest

from bench_extract.datasets import cuad as cuad_loader
from bench_extract.datasets.base import DatasetRunContext
from experiments.config import DATASET_REGISTRY

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "cuad_mini"

cuad_dataset = DATASET_REGISTRY["cuad"]

PARAMS = {"valid_source": "train", "valid_size": 1, "train_size": None, "seed": 0, "max_chars": None}


def test_registered_with_only_non_yes_no_fields_and_descriptions():
    assert len(cuad_dataset.FIELDS) == 10
    assert len(cuad_dataset.EXCLUDED_YES_NO_FIELDS) == 31
    assert not set(cuad_dataset.FIELDS) & set(cuad_dataset.EXCLUDED_YES_NO_FIELDS)
    assert set(cuad_dataset.FIELD_DESCRIPTIONS) == set(cuad_dataset.FIELDS)


def test_prompt_template_formats_with_runner_placeholders():
    out = cuad_dataset.EXTRACTION_PROMPT_TEMPLATE.format(fields="a, b", document_text="TEXT")
    assert "a, b" in out and "TEXT" in out


def test_load_run_context_returns_context():
    ctx = cuad_dataset.load_run_context(PARAMS, data_root=FIXTURE_ROOT)
    assert isinstance(ctx, DatasetRunContext)
    assert set(ctx.all_docs) == {"c0", "c1", "c2", "c3", "c4"}
    assert ctx.fields == sorted(cuad_dataset.FIELDS)
    assert ctx.all_docs["c2"].fields["governing_law"] == []  # configured field with no label
    assert "insurance" not in ctx.fields  # yes/no category, excluded


def test_drift_guard_raises_when_data_json_categories_disagree(tmp_path):
    drifted = {"categories": {"document_name": "Document Name"}, "documents": {}}
    (tmp_path / "data.json").write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="disagrees with"):
        cuad_dataset.load_run_context(PARAMS, data_root=tmp_path)


def test_drift_guard_raises_when_data_json_gains_a_category(tmp_path):
    cats = {**cuad_loader.load_categories(FIXTURE_ROOT), "brand_new": "Brand New"}
    (tmp_path / "data.json").write_text(json.dumps({"categories": cats, "documents": {}}))
    with pytest.raises(ValueError, match="only_in_data=\\['brand_new'\\]"):
        cuad_dataset.load_run_context(PARAMS, data_root=tmp_path)


def test_score_field_uses_category_rule():
    ctx = cuad_dataset.load_run_context(PARAMS, data_root=FIXTURE_ROOT)
    assert ctx.score_field(["Acme Corp"], ["Acme Corp."], "parties")
    assert ctx.score_field([], [], "renewal_term")
    assert not ctx.score_field(["invented clause"], [], "renewal_term")


def test_windowed_text_applies_max_chars():
    full = cuad_dataset.load_run_context(PARAMS, data_root=FIXTURE_ROOT)
    capped = cuad_dataset.load_run_context({**PARAMS, "max_chars": 20}, data_root=FIXTURE_ROOT)
    doc = full.all_docs["c0"]
    assert full.windowed_text(doc) == doc.text
    assert capped.windowed_text(doc) == doc.text[:20]


def test_valid_from_train_holds_out_and_never_touches_test():
    train = cuad_dataset.load_split_doc_ids(PARAMS, "train", data_root=FIXTURE_ROOT)
    valid = cuad_dataset.load_split_doc_ids(PARAMS, "valid", data_root=FIXTURE_ROOT)
    assert len(valid) == 1 and len(train) == 3
    assert not set(train) & set(valid)
    assert "c4" not in set(train) | set(valid)


def test_valid_from_test_uses_official_test():
    p = {**PARAMS, "valid_source": "test", "valid_size": None}
    assert cuad_dataset.load_split_doc_ids(p, "valid", data_root=FIXTURE_ROOT) == ["c4"]
    assert len(cuad_dataset.load_split_doc_ids(p, "train", data_root=FIXTURE_ROOT)) == 4


def test_train_size_subsample_is_nested_and_seeded():
    small = cuad_dataset.load_split_doc_ids({**PARAMS, "train_size": 1}, "train", data_root=FIXTURE_ROOT)
    big = cuad_dataset.load_split_doc_ids({**PARAMS, "train_size": 2}, "train", data_root=FIXTURE_ROOT)
    assert set(small) <= set(big)
    assert small == cuad_dataset.load_split_doc_ids({**PARAMS, "train_size": 1}, "train", data_root=FIXTURE_ROOT)


def test_bad_split_params_raise():
    official = cuad_loader.load_official_split(FIXTURE_ROOT)
    with pytest.raises(ValueError):
        cuad_loader.resolve_split(official, "train", None, 0, valid_size=None)
    with pytest.raises(ValueError):
        cuad_loader.resolve_split(official, "bogus", None, 0)
    with pytest.raises(ValueError):
        cuad_loader.resolve_split(official, "test", 99, 0)


def test_describe_and_training_split_params():
    assert cuad_dataset.describe_split({**PARAMS, "train_size": 10}) == {"train_size": 10, "seed": 0}
    base = {k: v for k, v in PARAMS.items() if k != "train_size"}
    out = cuad_dataset.training_split_params({**base, "train_sizes": [1, 2]})
    assert [p["train_size"] for p in out] == [1, 2]
    assert all("train_sizes" not in p for p in out)
