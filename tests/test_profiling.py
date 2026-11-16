import math

from bench_extract.datasets.base import DatasetRunContext
from bench_extract.datasets.matching import score_prediction
from bench_extract.datasets.vrdu import VRDUDocument
from bench_extract.profiling import build_profiling_table
from bench_extract.scoring.unified import ScoringConfig, unified_context_kwargs

FIELDS = ["advertiser", "gross_amount"]
MATCH_FUNC_BY_FIELD = {"advertiser": "GeneralStringMatch", "gross_amount": "PriceMatch"}


def score_field(predicted, ground_truth, field):
    return float(score_prediction(predicted, ground_truth, MATCH_FUNC_BY_FIELD[field]))


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

    ctx = DatasetRunContext(fields=FIELDS, all_docs=DOCUMENTS, windowed_text=lambda d: d.ocr_text, score_field=score_field)
    table = build_profiling_table(run_predictions, ctx, timing)

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


def _unified_ctx(docs):
    config = ScoringConfig.from_params({"measure": "unified", "match_threshold": 0.5, "quantity_decimals": 2})
    return DatasetRunContext(
        fields=["advertiser", "line_items"],
        all_docs=docs,
        windowed_text=lambda d: d.ocr_text,
        record_fields={"line_items": ["channel", "sub_amount"]},
        **unified_context_kwargs(
            fields=["advertiser", "line_items"],
            field_types={"advertiser": "short_text", "channel": "identifier", "sub_amount": "quantity"},
            record_fields={"line_items": ["channel", "sub_amount"]},
            config=config,
            entity_ground_truth=lambda doc, f: doc.field_entities.get(f, []),
            record_ground_truth=lambda doc, _g: doc.line_items,
        ),
    )


def test_unified_profiling_rows_are_scored_fields_with_float_and_nan_correct():
    docs = {
        "d1": VRDUDocument(
            document_id="d1", corpus="ad-buy-form", filename="d1.pdf", ocr_text="...", pages=[(0, 3)],
            fields={"advertiser": ["Acme", "ACME"], "channel": ["9", "9"], "sub_amount": ["$600", "$50"]},
            field_entities={"advertiser": [["Acme", "ACME"]]},  # one entity, two appearances
            line_items=[{"channel": "9", "sub_amount": "$600"}, {"channel": "9", "sub_amount": "$50"}],
        ),
        "d2": VRDUDocument(
            document_id="d2", corpus="ad-buy-form", filename="d2.pdf", ocr_text="...", pages=[(0, 3)],
            fields={}, field_entities={}, line_items=[],
        ),
    }
    run_predictions = {
        "m": {
            "d1": {"advertiser": ["acme"], "line_items": [{"channel": "9", "sub_amount": "600.00"}, {"channel": "9", "sub_amount": "999"}]},
            "d2": {"advertiser": [], "line_items": []},
        }
    }
    table = build_profiling_table(run_predictions, _unified_ctx(docs), {})
    # line_items is one record row per (doc, model)
    assert set(table["field"]) == {"advertiser", "line_items"} and len(table) == 4
    cell = lambda d, f: table[(table["document_id"] == d) & (table["field"] == f)].iloc[0]  # noqa: E731
    assert cell("d1", "advertiser")["correct"] == 1.0  # one entity, not two appearances to recall
    # records: ("9","600") exact; ("9","999") vs ("9","$50") -> mean(1, 0) = 0.5 >= threshold
    # soft TP = 1.5 of 2 pred / 2 gt -> F1 0.75
    assert math.isclose(cell("d1", "line_items")["correct"], 0.75)
    assert math.isnan(cell("d2", "advertiser")["correct"])  # both empty: not scored
    assert table.groupby("field")["correct"].mean()["advertiser"] == 1.0  # NaN skipped by mean()
