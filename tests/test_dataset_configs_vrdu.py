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
    # The field list comes from CORPORA["ad-buy-form"], not whatever this
    # fixture doc happens to have values for: the 9 header fields plus one
    # line_items field standing in for the 5 line-item sub-fields.
    cfg = vrdu_dataset.CORPORA["ad-buy-form"]
    assert ctx.fields == cfg.extraction_fields
    assert len(ctx.fields) == 10 and ctx.fields[-1] == "line_items"
    assert not set(cfg.line_item_fields) & set(ctx.fields)
    assert set(ctx.fields[:-1]) | set(cfg.line_item_fields) == set(cfg.match_func_by_field)
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


@pytest.mark.parametrize("corpus", sorted(vrdu_dataset.CORPORA))
def test_field_descriptions_cover_exactly_the_configured_fields(corpus):
    cfg = vrdu_dataset.CORPORA[corpus]
    if not cfg.match_func_by_field:
        pytest.skip(f"{corpus}: field map is still a TODO")
    assert set(cfg.field_descriptions) == set(cfg.match_func_by_field)
    assert all(len(d.split()) >= 8 for d in cfg.field_descriptions.values())


def test_field_descriptions_mismatch_is_rejected():
    with pytest.raises(ValueError, match="disagree"):
        vrdu_dataset.VRDUCorpusConfig(corpus="x", match_func_by_field={"a": "DateMatch"}, field_descriptions={"b": "..."})


UNIFIED = {"measure": "unified", "match_threshold": 0.5, "quantity_decimals": 2}


def test_native_is_the_default_and_scores_floats():
    ctx = vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    assert ctx.scoring_measure == "native"
    assert ctx.score_field(["Acme Corp."], ["Acme Corp"], "advertiser") == 1.0


def test_native_scores_the_14_entities_and_flattens_line_items():
    ctx = vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    cfg = vrdu_dataset.CORPORA["ad-buy-form"]
    assert ctx.scored_fields == sorted(cfg.match_func_by_field)
    assert ctx.extraction_field_of["sub_amount"] == "line_items" and ctx.extraction_field_of["advertiser"] == "advertiser"
    assert ctx.scored_fields_of("line_items") == sorted(cfg.line_item_fields)
    output = {"line_items": [{"channel": "3", "sub_amount": "$650.00"}, {"channel": "", "sub_amount": "$600.00", "junk": 1}, "bad row"]}
    assert ctx.prediction_for(output, "channel") == ["3"]  # blank cells dropped, not predicted as ""
    assert ctx.prediction_for(output, "sub_amount") == ["$650.00", "$600.00"]
    doc = ctx.all_docs["doc-ad-1"]
    assert ctx.score_field(ctx.prediction_for(output, "sub_amount"), ctx.ground_truth(doc, "sub_amount"), "sub_amount") == 1.0
    assert ctx.prediction_for({}, "channel") == []


def test_prompt_asks_for_line_items_as_row_objects():
    cfg = vrdu_dataset.CORPORA["ad-buy-form"]
    prompt = vrdu_dataset.EXTRACTION_PROMPT_TEMPLATE.format(fields="advertiser, line_items", document_text="X")
    assert "line_items" in prompt and ", ".join(cfg.line_item_fields) in prompt and '""' in prompt
    ctx = vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    assert all(f in ctx.field_descriptions["line_items"] for f in cfg.line_item_fields)
    assert not set(cfg.line_item_fields) & set(ctx.field_descriptions)


def test_line_items_that_differ_from_the_prompt_are_refused(monkeypatch):
    cfg = vrdu_dataset.CORPORA["ad-buy-form"]
    monkeypatch.setattr(vrdu_dataset, "PROMPT_LINE_ITEM_FIELDS", list(reversed(cfg.line_item_fields)))
    with pytest.raises(ValueError, match="EXTRACTION_PROMPT_TEMPLATE"):
        vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)


def test_unified_scores_line_items_as_one_record_field():
    ctx = vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, scoring=UNIFIED, data_root=FIXTURE_ROOT)
    cfg = vrdu_dataset.CORPORA["ad-buy-form"]
    assert ctx.scoring_measure == "unified"
    assert ctx.fields == cfg.extraction_fields and ctx.scored_fields == ctx.fields
    assert ctx.extraction_field_of == {f: f for f in ctx.fields}
    assert ctx.record_fields == {"line_items": cfg.line_item_fields}
    assert ctx.scored_field_types["line_items"] == "record" and ctx.scored_field_types["gross_amount"] == "quantity"

    doc = ctx.all_docs["doc-ad-1"]
    output = {
        "advertiser": ["ACME CORP"],
        "gross_amount": ["1250"],
        "line_items": [{"channel": "3", "sub_amount": "650", "program_desc": ""}, {"channel": 9, "sub_amount": "$600"}],
    }
    for f in ("advertiser", "gross_amount", "line_items"):
        assert ctx.score_field(ctx.prediction_for(output, f), ctx.ground_truth(doc, f), f) == 1.0
    assert ctx.score_field(ctx.prediction_for(output, "agency"), ctx.ground_truth(doc, "agency"), "agency") is None
    # rows that don't line up are penalized: the channels swapped across rows
    swapped = {"line_items": [{"channel": "9", "sub_amount": "650"}, {"channel": "3", "sub_amount": "$600"}]}
    assert ctx.score_field(ctx.prediction_for(swapped, "line_items"), ctx.ground_truth(doc, "line_items"), "line_items") == 0.5


def test_unified_refuses_stale_data_json(tmp_path):
    data = json.loads((FIXTURE_ROOT / "data.json").read_text())
    for doc in data["documents"].values():
        doc.pop("field_entities", None)
        doc.pop("line_items", None)
    (tmp_path / "data.json").write_text(json.dumps(data))
    vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, data_root=tmp_path)  # native still fine
    with pytest.raises(ValueError, match="download_vrdu"):
        vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, scoring=UNIFIED, data_root=tmp_path)


@pytest.mark.parametrize("scoring", [None, UNIFIED])
def test_line_item_fields_cross_checked_against_data_json(tmp_path, scoring):
    data = json.loads((FIXTURE_ROOT / "data.json").read_text())
    data["corpora"]["ad-buy-form"]["entity_appearance_pattern"]["sub_amount"] = "unrepeated"
    (tmp_path / "data.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="line_item_fields"):
        vrdu_dataset.load_run_context({"corpus": "ad-buy-form"}, scoring=scoring, data_root=tmp_path)


def test_field_types_mismatch_is_rejected():
    with pytest.raises(ValueError, match="field_type_by_field"):
        vrdu_dataset.VRDUCorpusConfig(corpus="x", match_func_by_field={"a": "DateMatch"}, field_type_by_field={"b": "date"})
