from pathlib import Path

import pytest
import yaml

from experiments.config import (
    EXTRACTOR_REGISTRY,
    MODEL_REGISTRY,
    EXPERIMENT_CONFIGS_DIR,
    load_dataset_config,
    load_experiment_config,
    load_extraction_prompt_template,
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


def test_load_dataset_config_known_corpora():
    ad_buy = load_dataset_config("ad-buy-form")
    assert ad_buy.corpus == "ad-buy-form"
    assert ad_buy.max_pages == 3

    registration = load_dataset_config("registration-form")
    assert registration.corpus == "registration-form"
    assert registration.max_pages == 2


def test_load_dataset_config_unknown_corpus_raises():
    with pytest.raises(KeyError):
        load_dataset_config("not-a-corpus")


def test_load_extraction_prompt_template_has_placeholders():
    template = load_extraction_prompt_template()
    assert "{fields}" in template
    assert "{document_text}" in template


def test_load_experiment_config_example_benchmark_config():
    path = EXPERIMENT_CONFIGS_DIR / "benchmark" / "2026-09-21-example-benchmark-01.yaml"
    config = load_experiment_config(path)
    assert config.id == "2026-09-21-example-benchmark-01"
    assert config.project == "bench-extract"
    assert config.params["corpus"] == "ad-buy-form"
    assert "route_baseline" in config.params["methods"]


def test_load_experiment_config_example_training_config():
    path = EXPERIMENT_CONFIGS_DIR / "training" / "2026-09-21-example-training-01.yaml"
    config = load_experiment_config(path)
    assert config.id == "2026-09-21-example-training-01"
    assert len(config.params["split_files"]) == 4


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
