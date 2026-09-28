from bench_extract.datasets.vrdu import CorpusSchema, VRDUDocument
from bench_extract.profiling import build_profiling_table, score_prediction

SCHEMA = CorpusSchema(
    dataset_name="TestCorpus",
    entity_name_to_match_func={"advertiser": "GeneralStringMatch", "gross_amount": "PriceMatch"},
)

DOCUMENTS = {
    "doc-1": VRDUDocument(
        document_id="doc-1",
        corpus="ad-buy-form",
        filename="doc-1.pdf",
        ocr_text="...",
        pages=[(0, 3)],
        fields={"advertiser": ["Acme Corp"], "gross_amount": ["$100.00"]},
    ),
    "doc-2": VRDUDocument(
        document_id="doc-2",
        corpus="ad-buy-form",
        filename="doc-2.pdf",
        ocr_text="...",
        pages=[(0, 3)],
        fields={"advertiser": ["Widget Inc"], "gross_amount": ["$50.00"]},
    ),
}


def test_score_prediction_correct_and_incorrect():
    assert score_prediction(["Acme Corp"], ["Acme Corp"], "GeneralStringMatch")
    assert not score_prediction(["Widget Inc"], ["Acme Corp"], "GeneralStringMatch")


def test_score_prediction_empty_prediction_is_incorrect():
    assert not score_prediction([], ["Acme Corp"], "GeneralStringMatch")


def test_build_profiling_table_shape_and_correctness():
    run_predictions = {
        "model-a": {
            "doc-1": {"advertiser": ["Acme Corp"], "gross_amount": ["$100.00"]},  # both correct
            "doc-2": {"advertiser": ["Wrong Co"], "gross_amount": ["$50.00"]},  # advertiser wrong
        },
        "model-b": {
            "doc-1": {"advertiser": ["Acme Corp"], "gross_amount": ["$999.00"]},  # gross_amount wrong
            "doc-2": {"advertiser": ["Widget Inc"], "gross_amount": ["$50.00"]},  # both correct
        },
    }
    timing = {
        "model-a": {
            "doc-1": {"wall_seconds": 1.0, "prompt_tokens": 10, "completion_tokens": 5, "error": None},
            "doc-2": {"wall_seconds": 1.5, "prompt_tokens": 12, "completion_tokens": 6, "error": None},
        },
        "model-b": {
            "doc-1": {"wall_seconds": 2.0, "prompt_tokens": 20, "completion_tokens": 8, "error": None},
            "doc-2": {"wall_seconds": 2.5, "prompt_tokens": 22, "completion_tokens": 9, "error": None},
        },
    }

    table = build_profiling_table(run_predictions, DOCUMENTS, SCHEMA, timing)

    # 2 documents x 2 fields x 2 models = 8 rows.
    assert len(table) == 8
    assert set(table["model_key"]) == {"model-a", "model-b"}
    assert set(table["field"]) == {"advertiser", "gross_amount"}

    def correct_for(model_key, document_id, field):
        row = table[(table["model_key"] == model_key) & (table["document_id"] == document_id) & (table["field"] == field)]
        return row["correct"].iloc[0]

    assert correct_for("model-a", "doc-1", "advertiser")
    assert correct_for("model-a", "doc-1", "gross_amount")
    assert not correct_for("model-a", "doc-2", "advertiser")
    assert correct_for("model-a", "doc-2", "gross_amount")

    assert correct_for("model-b", "doc-1", "advertiser")
    assert not correct_for("model-b", "doc-1", "gross_amount")
    assert correct_for("model-b", "doc-2", "advertiser")
    assert correct_for("model-b", "doc-2", "gross_amount")
