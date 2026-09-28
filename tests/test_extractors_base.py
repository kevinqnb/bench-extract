import pandas as pd
import pytest

from bench_extract.extractors.cascade_extractor import CascadeExtractor
from bench_extract.extractors.route_extractor import RouteExtractor
from bench_extract.extractors.schedule_extractor import ScheduleExtractor
from bench_extract.models.base import ExtractionModel, ExtractionPrediction
from bench_extract.utils.timing import UsageRecord


class FakeModel(ExtractionModel):
    """Returns a fixed field_values dict regardless of input, with a fixed
    wall_seconds -- enough to drive the base classes' control flow without
    any network call."""

    def __init__(self, key, field_values, wall_seconds=1.0):
        self.key = key
        self._field_values = field_values
        self.wall_seconds = wall_seconds
        self.calls = 0

    def extract(self, document_text, fields):
        self.calls += 1
        return ExtractionPrediction(
            field_values={f: self._field_values.get(f, []) for f in fields},
            usage=UsageRecord(wall_seconds=self.wall_seconds),
        )


PROFILING_TABLE = pd.DataFrame(
    [
        # model-cheap: cheap, less accurate on "advertiser"
        {"document_id": "d1", "field": "advertiser", "model_key": "model-cheap", "correct": True, "wall_seconds": 0.5},
        {"document_id": "d2", "field": "advertiser", "model_key": "model-cheap", "correct": False, "wall_seconds": 0.5},
        {"document_id": "d1", "field": "gross_amount", "model_key": "model-cheap", "correct": False, "wall_seconds": 0.5},
        {"document_id": "d2", "field": "gross_amount", "model_key": "model-cheap", "correct": False, "wall_seconds": 0.5},
        # model-expensive: slower, more accurate
        {"document_id": "d1", "field": "advertiser", "model_key": "model-expensive", "correct": True, "wall_seconds": 3.0},
        {"document_id": "d2", "field": "advertiser", "model_key": "model-expensive", "correct": True, "wall_seconds": 3.0},
        {"document_id": "d1", "field": "gross_amount", "model_key": "model-expensive", "correct": True, "wall_seconds": 3.0},
        {"document_id": "d2", "field": "gross_amount", "model_key": "model-expensive", "correct": True, "wall_seconds": 3.0},
    ]
)


def _models():
    cheap = FakeModel("model-cheap", {"advertiser": ["Acme"], "gross_amount": []}, wall_seconds=0.5)
    expensive = FakeModel("model-expensive", {"advertiser": ["Acme"], "gross_amount": ["$100"]}, wall_seconds=3.0)
    return [cheap, expensive]


def test_multi_model_extractor_requires_models():
    with pytest.raises(ValueError):
        RouteExtractor([])


def test_extract_before_fit_raises():
    extractor = RouteExtractor(_models())
    with pytest.raises(RuntimeError):
        extractor.extract("text", ["advertiser"])


def test_route_extractor_baseline_picks_best_model():
    extractor = RouteExtractor(_models())
    extractor.fit(PROFILING_TABLE)
    # model-expensive has higher mean accuracy (1.0 vs 0.25) across the table.
    assert extractor._default_model_key == "model-expensive"

    result = extractor.extract("text", ["advertiser", "gross_amount"])
    assert result.field_values == {"advertiser": ["Acme"], "gross_amount": ["$100"]}
    assert len(result.steps) == 1
    assert result.steps[0].model_key == "model-expensive"
    assert result.total_wall_seconds == 3.0


def test_cascade_extractor_baseline_escalates_until_all_fields_present():
    cheap, expensive = _models()
    extractor = CascadeExtractor([cheap, expensive], model_order=["model-cheap", "model-expensive"])
    extractor.fit(PROFILING_TABLE)

    result = extractor.extract("text", ["advertiser", "gross_amount"])
    # model-cheap never returns gross_amount, so the cascade must escalate.
    assert cheap.calls == 1
    assert expensive.calls == 1
    assert result.field_values == {"advertiser": ["Acme"], "gross_amount": ["$100"]}
    assert [s.model_key for s in result.steps] == ["model-cheap", "model-expensive"]
    assert result.steps[0].accepted is False
    assert result.steps[1].accepted is True


def test_cascade_extractor_accepts_first_model_if_sufficient():
    cheap = FakeModel("model-cheap", {"advertiser": ["Acme"], "gross_amount": ["$100"]}, wall_seconds=0.5)
    expensive = FakeModel("model-expensive", {"advertiser": ["Acme"], "gross_amount": ["$100"]}, wall_seconds=3.0)
    extractor = CascadeExtractor([cheap, expensive], model_order=["model-cheap", "model-expensive"])
    extractor.fit(PROFILING_TABLE)

    result = extractor.extract("text", ["advertiser", "gross_amount"])
    assert cheap.calls == 1
    assert expensive.calls == 0
    assert len(result.steps) == 1


def test_cascade_extractor_rejects_model_order_mismatch():
    with pytest.raises(ValueError):
        CascadeExtractor(_models(), model_order=["model-cheap"])


def test_schedule_extractor_assigns_cheapest_model_above_accuracy_floor():
    extractor = ScheduleExtractor(_models(), min_accuracy=0.5)
    extractor.fit(PROFILING_TABLE)

    # advertiser: model-cheap (acc 0.5, wall 0.5) clears the floor and is cheaper
    # than model-expensive (acc 1.0, wall 3.0) -- greedy picks model-cheap.
    assert extractor._assignment["advertiser"] == "model-cheap"
    # gross_amount: model-cheap (acc 0.0) doesn't clear 0.5, only model-expensive does.
    assert extractor._assignment["gross_amount"] == "model-expensive"

    result = extractor.extract("text", ["advertiser", "gross_amount"])
    assert result.field_values["advertiser"] == ["Acme"]
    assert result.field_values["gross_amount"] == ["$100"]
    # One call per distinct assigned model.
    assert {s.model_key for s in result.steps} == {"model-cheap", "model-expensive"}


def test_schedule_extractor_raises_for_unassigned_field():
    extractor = ScheduleExtractor(_models())
    extractor.fit(PROFILING_TABLE)
    with pytest.raises(KeyError):
        extractor.extract("text", ["not_a_field"])
