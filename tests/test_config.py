from pathlib import Path

import pytest
import yaml

from experiments.config import (
    DATASET_REGISTRY,
    EXTRACTOR_REGISTRY,
    MODEL_REGISTRY,
    EXPERIMENT_CONFIGS_DIR,
    load_experiment_config,
)


def test_model_registry_loads_gpt_oss_120b():
    assert "gpt-oss-120b" in MODEL_REGISTRY
    config = MODEL_REGISTRY["gpt-oss-120b"]
    assert config.key == "gpt-oss-120b"
    assert config.serving == "external"
    assert config.base_url_env == "BENCH_EXTRACT_GPT_OSS_120B_BASE_URL"
    assert config.hardware.device == "cuda"
    assert config.hardware.gpu_count == 1


def test_extractor_registry_has_baselines_and_all_nine_methods():
    expected_methods = {
        "hybridllm",
        "routellm",
        "frugalgpt",
        "bargain",
        "task_cascade",
        "automix",
        "abacus",
        "doctopus",
        "cascade_routing",
    }
    assert expected_methods <= set(EXTRACTOR_REGISTRY)
    assert {"route_baseline", "cascade_baseline", "schedule_baseline"} <= set(EXTRACTOR_REGISTRY)

    assert EXTRACTOR_REGISTRY["hybridllm"].base_kind == "route"
    assert EXTRACTOR_REGISTRY["routellm"].base_kind == "route"
    assert EXTRACTOR_REGISTRY["frugalgpt"].base_kind == "cascade"
    assert EXTRACTOR_REGISTRY["bargain"].base_kind == "cascade"
    assert EXTRACTOR_REGISTRY["task_cascade"].base_kind == "cascade"
    assert EXTRACTOR_REGISTRY["automix"].base_kind == "cascade"
    assert EXTRACTOR_REGISTRY["abacus"].base_kind == "schedule"
    assert EXTRACTOR_REGISTRY["doctopus"].base_kind == "schedule"
    assert EXTRACTOR_REGISTRY["cascade_routing"].base_kind == "cascade"


def test_dataset_registry_has_vrdu():
    assert "vrdu" in DATASET_REGISTRY
    vrdu_module = DATASET_REGISTRY["vrdu"]
    assert "ad-buy-form" in vrdu_module.CORPORA
    assert vrdu_module.CORPORA["ad-buy-form"].max_pages == 3
    assert "registration-form" in vrdu_module.CORPORA
    assert vrdu_module.CORPORA["registration-form"].max_pages == 2


def test_dataset_registry_module_has_required_contract():
    vrdu_module = DATASET_REGISTRY["vrdu"]
    assert callable(vrdu_module.load_run_context)
    assert callable(vrdu_module.load_split_doc_ids)
    assert callable(vrdu_module.describe_split)


def test_load_extraction_prompt_template_has_placeholders():
    template = DATASET_REGISTRY["vrdu"].EXTRACTION_PROMPT_TEMPLATE
    assert "{fields}" in template
    assert "{document_text}" in template


def test_load_experiment_config_example_benchmark_config():
    path = EXPERIMENT_CONFIGS_DIR / "benchmark" / "2026-09-21-example-benchmark-01.yaml"
    config = load_experiment_config(path)
    assert config.id == "2026-09-21-example-benchmark-01"
    assert config.project == "bench-extract"
    assert config.params["dataset"] == "vrdu"
    assert config.params["dataset_params"]["corpus"] == "ad-buy-form"
    assert "route_baseline" in config.params["methods"]


def test_load_experiment_config_example_training_config():
    path = EXPERIMENT_CONFIGS_DIR / "training" / "2026-09-21-example-training-01.yaml"
    config = load_experiment_config(path)
    assert config.id == "2026-09-21-example-training-01"
    assert config.params["dataset"] == "vrdu"
    assert len(config.params["dataset_params"]["split_files"]) == 4


def test_load_experiment_config_missing_key_raises(tmp_path):
    path = tmp_path / "2026-01-01-bad-config-01.yaml"
    path.write_text(yaml.dump({"id": "2026-01-01-bad-config-01", "project": "bench-extract", "seed": 0, "params": {}}))
    with pytest.raises(AssertionError):
        load_experiment_config(path)


def test_load_experiment_config_id_filename_mismatch_raises(tmp_path):
    path = tmp_path / "2026-01-01-actual-name-01.yaml"
    path.write_text(
        yaml.dump({"id": "2026-01-01-different-id-01", "project": "bench-extract", "description": "x", "seed": 0, "params": {}})
    )
    with pytest.raises(AssertionError):
        load_experiment_config(path)


# --- every committed experiment config resolves --------------------------------

ALL_EXPERIMENT_CONFIGS = sorted(EXPERIMENT_CONFIGS_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("path", ALL_EXPERIMENT_CONFIGS, ids=lambda p: p.stem)
def test_every_experiment_config_references_known_models_and_methods(path):
    config = load_experiment_config(path)
    params = config.params
    assert set(params["model_keys"]) <= set(MODEL_REGISTRY)
    assert set(params["methods"]) <= set(EXTRACTOR_REGISTRY)
    assert params["dataset"] in DATASET_REGISTRY


def test_experiment_ids_are_unique_across_subdirectories():
    ids = [p.stem for p in ALL_EXPERIMENT_CONFIGS]
    assert len(ids) == len(set(ids))


def test_abacus_experiments_differ_only_in_policy():
    paths = sorted((EXPERIMENT_CONFIGS_DIR / "benchmark" / "abacus").glob("*.yaml"))
    assert len(paths) == 7
    normalized = []
    for path in paths:
        params = load_experiment_config(path).params
        params["method_kwargs"]["abacus"].pop("policy")
        normalized.append(params)
    assert all(p == normalized[0] for p in normalized)
    assert "qwen3.5-0.8b" not in normalized[0]["model_keys"]


def test_abacus_policy_configs_build_every_pz_policy():
    pzs = pytest.importorskip("bench_extract.extractors.utils.pz_support")
    names = set()
    for path in (EXPERIMENT_CONFIGS_DIR / "benchmark" / "abacus").glob("*.yaml"):
        spec = dict(load_experiment_config(path).params["method_kwargs"]["abacus"]["policy"])
        names.add(spec["name"])
        pzs.build_policy(spec.pop("name"), **spec)
    assert names == set(pzs.POLICIES)


def test_new_self_hosted_models_are_independently_servable():
    keys = ["qwen3.5-0.8b", "qwen3.5-4b", "gemma-4-12b"]
    names = [MODEL_REGISTRY[k].served_model_name for k in keys + ["gpt-oss-120b"]]
    envs = [MODEL_REGISTRY[k].base_url_env for k in keys]
    assert len(set(envs)) == len(envs)
    for k in keys:
        m = MODEL_REGISTRY[k]
        assert m.serving == "external" and m.hardware.device == "cuda"
        assert m.served_model_name == k
    assert None not in names[:3] and len(set(names[:3])) == 3
