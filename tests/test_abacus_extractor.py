import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from bench_extract.models.base import OpenAIChatModel
from experiments.config import DATASET_REGISTRY

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "vrdu_mini"
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_abacus_module_imports_without_importing_palimpzest():
    code = "import sys, bench_extract.extractors.abacus; assert 'palimpzest' not in sys.modules"
    subprocess.run([sys.executable, "-I", "-c", code], check=True, env={"PYTHONPATH": str(REPO_ROOT / "src")})


pytest.importorskip("palimpzest")
from bench_extract.extractors.abacus import AbacusExtractor  # noqa: E402
from bench_extract.extractors.utils import pz_support as pzs  # noqa: E402


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("MaxQuality", {}),
        ("MinCost", {}),
        ("MinTime", {}),
        ("MaxQualityAtFixedCost", {"max_cost": 1.0}),
        ("MaxQualityAtFixedTime", {"max_time": 10.0}),
        ("MinCostAtFixedQuality", {"min_quality": 0.5}),
        ("MinTimeAtFixedQuality", {"min_quality": 0.5}),
    ],
)
def test_build_policy_covers_every_policy(name, kwargs):
    assert pzs.build_policy(name, **kwargs) is not None


def test_build_policy_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown palimpzest policy"):
        pzs.build_policy("MaxVibes")


def test_score_field_validator_averages_per_field_scores():
    gt = {"d1": {"advertiser": ["Acme"], "gross_amount": ["$1"]}}
    validator = pzs.ScoreFieldValidator(gt, lambda pred, truth, field: pred == truth)
    record = {"document_id": "d1", "contents": "..."}
    assert validator.map_score_fn(["advertiser", "gross_amount"], record, {"advertiser": ["Acme"], "gross_amount": ["$2"]}) == 0.5
    # a bare string / None output is normalized to a list
    assert validator.map_score_fn(["advertiser"], record, {"advertiser": "Acme"}) == 1.0
    assert validator.map_score_fn(["advertiser"], record, {"advertiser": None}) == 0.0
    # unknown document => no score (pz skips None)
    assert validator.map_score_fn(["advertiser"], {"document_id": "zzz"}, {"advertiser": ["Acme"]}) is None


def test_resolve_validator_cls_registry_and_dotted_path_and_unknown():
    assert pzs.resolve_validator_cls("score_field") is pzs.ScoreFieldValidator
    assert pzs.resolve_validator_cls("bench_extract.extractors.utils.pz_support:ScoreFieldValidator") is pzs.ScoreFieldValidator
    with pytest.raises(KeyError):
        pzs.resolve_validator_cls("nope")


def test_document_dataset_yields_rows():
    ds = pzs.DocumentDataset("docs", [{"document_id": "a", "contents": "x"}, {"document_id": "b", "contents": "y"}])
    assert len(ds) == 2 and ds[1] == {"document_id": "b", "contents": "y"}


def test_make_pz_model_routes_to_distinct_endpoints():
    a = pzs.make_pz_model("t-a", "http://h:1/v1")
    b = pzs.make_pz_model("t-b", "http://h:2/v1")
    assert a.model_id == "hosted_vllm/t-a" and a.api_base == "http://h:1/v1" and a != b


ANSWERS = {"advertiser": ["Acme Corp"], "gross_amount": ["$1,250.00"], "flight_from": ["01/02/2024"]}


def _fake_completion(calls, all_fields):
    def completion(model, messages, **kwargs):
        calls.append((model, kwargs.get("api_base")))
        response = mock.MagicMock()
        response.choices[0].message.content = json.dumps({f: ANSWERS.get(f, []) for f in all_fields})  # pz expects every field
        response.usage.model_dump.return_value = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return response

    return completion


@pytest.mark.parametrize("policy", ["MaxQuality", "MinCost"])
def test_fit_and_extract_many_end_to_end_with_two_local_models(policy):
    vrdu = DATASET_REGISTRY["vrdu"]
    ctx = vrdu.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    base = ctx.all_docs["doc-ad-1"]
    train_docs = [dataclasses.replace(base, document_id=f"train-{i}") for i in range(3)]
    models = [
        OpenAIChatModel(key=f"m-{s}", model=f"qwen3-0.6b-{s}", prompt_template="", base_url=f"http://localhost:{port}/v1", max_tokens=64)
        for s, port in (("a", 8000), ("b", 8001))
    ]
    extractor = AbacusExtractor(models, policy={"name": policy}, pz_config={"sample_budget": 3})
    calls: list = []
    with mock.patch("litellm.completion", _fake_completion(calls, ctx.fields)):
        extractor.bind_run_context(ctx, train_docs)
        extractor.fit(None)
        assert extractor._fitted
        assert extractor.fit_metrics["n_train_documents"] == 3
        assert {"optimization_seconds", "optimization_cost_usd"} <= set(extractor.fit_metrics)
        assert extractor.fit_metrics["optimization_input_tokens"] > 0 and extractor.fit_metrics["optimization_output_tokens"] > 0
        n_fit_calls = len(calls)
        results = extractor.extract_many([ctx.windowed_text(base), ctx.windowed_text(base)], ["advertiser", "gross_amount"])
    assert n_fit_calls > 0 and len(calls) > n_fit_calls  # optimization sampled, then the plan ran
    assert {api_base for _, api_base in calls} <= {"http://localhost:8000/v1", "http://localhost:8001/v1"}
    assert len(results) == 2
    for result in results:
        assert result.field_values == {"advertiser": ["Acme Corp"], "gross_amount": ["$1,250.00"]}
        assert result.steps and all(s.model_key in {"m-a", "m-b"} or s.model_key.startswith("pz:") for s in result.steps)


def test_extract_before_fit_and_unknown_field_raise():
    extractor = AbacusExtractor([OpenAIChatModel(key="m", model="m", prompt_template="", base_url="http://localhost:1/v1")])
    with pytest.raises(RuntimeError):
        extractor.extract_many(["x"], ["advertiser"])
    with pytest.raises(RuntimeError, match="bind_run_context"):
        extractor.fit(None)


def test_run_uses_the_plan_optimized_on_the_train_split():
    """model a answers correctly, model b answers nothing; after fit under
    MaxQuality (no mixtures/critics, so pz must pick ONE model) every
    run-time step must be model a -- i.e. the quality samples gathered in
    fit() really drive the plan run() executes."""
    vrdu = DATASET_REGISTRY["vrdu"]
    ctx = vrdu.load_run_context({"corpus": "ad-buy-form"}, data_root=FIXTURE_ROOT)
    base = ctx.all_docs["doc-ad-1"]
    train_docs = [dataclasses.replace(base, document_id=f"train-{i}") for i in range(4)]
    models = [
        OpenAIChatModel(key=f"m-{s}", model=f"qwen3-0.6b-{s}", prompt_template="", base_url=f"http://localhost:{port}/v1")
        for s, port in (("a", 8000), ("b", 8001))
    ]

    def completion(model, messages, **kwargs):
        good = kwargs.get("api_base", "").endswith(":8000/v1")
        response = mock.MagicMock()
        response.choices[0].message.content = json.dumps({f: (ANSWERS.get(f, []) if good else []) for f in ctx.fields})
        response.usage.model_dump.return_value = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return response

    extractor = AbacusExtractor(
        models,
        policy={"name": "MaxQuality"},
        pz_config={"sample_budget": 8, "allow_mixtures": False, "allow_critic": False},
    )
    with mock.patch("litellm.completion", completion):
        extractor.bind_run_context(ctx, train_docs)
        extractor.fit(None)
        results = extractor.extract_many([ctx.windowed_text(base)] * 2, ["advertiser", "gross_amount"])
    for result in results:
        assert result.field_values["advertiser"] == ["Acme Corp"]
        assert result.steps and {s.model_key for s in result.steps} == {"m-a"}
