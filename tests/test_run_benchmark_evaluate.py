"""experiments.run_benchmark._evaluate_extractor: scores scored fields with the
run context's measure, skips unscored cells, and keeps partial (soft) scores."""

import math
from pathlib import Path

from bench_extract.extractors.base import ExtractionResult, MultiModelExtractor
from experiments.config import DATASET_REGISTRY
from experiments.run_benchmark import _evaluate_extractor

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "vrdu_mini"
UNIFIED = {"measure": "unified", "match_threshold": 0.5, "quantity_decimals": 2}


class FixedExtractor(MultiModelExtractor):
    def __init__(self, values):
        self.models, self._fitted, self.fit_metrics, self.extraction_field_of = {}, True, {}, {}
        self.values = values

    def fit(self, profiling_table):
        pass

    def extract(self, document_text, fields):
        return ExtractionResult(field_values={f: self.values.get(f, []) for f in fields}, steps=[])


def _docs(ctx):
    return [ctx.all_docs["doc-ad-1"]]


def test_unified_metrics_skip_both_empty_cells_and_keep_partial_credit():
    ctx = DATASET_REGISTRY["vrdu"].load_run_context({"corpus": "ad-buy-form"}, scoring=UNIFIED, data_root=FIXTURE_ROOT)
    # advertiser right, gross_amount wrong, flight_from right; one of two line items
    values = {
        "advertiser": ["Acme Corp"],
        "gross_amount": ["$1"],
        "flight_from": ["01/02/2024"],
        "line_items": [{"channel": "9", "sub_amount": "$600"}],
    }
    metrics = _evaluate_extractor(FixedExtractor(values), _docs(ctx), ctx)
    # scored: advertiser, gross_amount, flight_from, line_items (the other 6 plain fields are both-empty)
    assert metrics["n_scored_cells"] == 4 and metrics["n_skipped_cells"] == 6
    line_items_f1 = 2 * 1.0 * 0.5 / 1.5
    assert math.isclose(metrics["accuracy"], (1 + 0 + 1 + line_items_f1) / 4)  # not truncated by int()
    assert math.isclose(metrics["accuracy_type_record"], line_items_f1)
    assert metrics["accuracy_type_quantity"] == 0.0 and metrics["accuracy_type_date"] == 1.0


def test_native_metrics_unchanged_shape():
    ctx = DATASET_REGISTRY["vrdu"].load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    values = {"advertiser": ["Acme Corp"], "line_items": [{"channel": "9", "sub_amount": ""}]}
    metrics = _evaluate_extractor(FixedExtractor(values), _docs(ctx), ctx)
    assert metrics["n_scored_cells"] == 14 and metrics["n_skipped_cells"] == 0  # the 14 VRDU entities, as before
    assert math.isclose(metrics["accuracy"], 2 / 14)  # advertiser, channel
    assert not any(k.startswith("accuracy_type_") for k in metrics)
