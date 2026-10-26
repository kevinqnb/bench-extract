import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("matplotlib")
from analysis import abacus as analysis_abacus  # noqa: E402


def _write_run(root: Path, run_id: str, policy: dict, accuracy: float, calls: dict, status="success"):
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"status": status}))
    (run_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump({"id": run_id, "params": {"method_kwargs": {"abacus": {"policy": policy}}}})
    )
    metrics = {
        "abacus_accuracy": accuracy,
        "abacus_mean_wall_seconds": 3.0 + accuracy,
        "abacus_elapsed_seconds": 40.0,
        "abacus_mean_prompt_tokens": 3000.0,
        "abacus_mean_completion_tokens": 200.0 * accuracy,
        "abacus_n_documents": 10,
        "abacus_fit_optimization_seconds": 12.0,
        "abacus_fit_optimization_input_tokens": 1000,
        "abacus_fit_optimization_output_tokens": 100,
        **{f"abacus_model_calls_{k}": v for k, v in calls.items()},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics))


def test_analyze_writes_summary_and_figures_and_skips_unfinished(tmp_path):
    results = tmp_path / "results"
    _write_run(results, "a-max-quality-01", {"name": "MaxQuality"}, 0.8, {"qwen3.5-4b": 5, "gpt-oss-120b": 15})
    _write_run(results, "a-min-time-at-fixed-quality-01", {"name": "MinTimeAtFixedQuality", "min_quality": 0.5}, 0.6, {"qwen3.5-4b": 20, "pz:MixtureOfAgents": 2})
    _write_run(results, "a-failed-01", {"name": "MinTime"}, 0.1, {}, status="failed")
    out = tmp_path / "out"
    runs = analysis_abacus.analyze(["a-max-quality-01", "a-min-time-at-fixed-quality-01", "a-failed-01", "a-missing-01"], results, out)
    assert [r["id"] for r in runs] == ["a-max-quality-01", "a-min-time-at-fixed-quality-01"]
    assert runs[1]["policy"] == "MinTimeAtFixedQuality(min_quality=0.5)"
    assert runs[0]["fit_tokens"] == 1100
    for name in ["summary.csv", "time_vs_accuracy.png", "tokens_vs_accuracy.png", "model_usage.png"]:
        assert (out / name).stat().st_size > 0


def test_analyze_with_no_finished_runs_exits(tmp_path):
    with pytest.raises(SystemExit):
        analysis_abacus.analyze(["nope"], tmp_path, tmp_path / "out")


def test_default_config_dir_ids_match_committed_abacus_configs():
    ids = analysis_abacus.config_ids(analysis_abacus.DEFAULT_CONFIG_DIR)
    assert len(ids) == 7
