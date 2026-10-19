"""Basic analysis of Abacus (palimpzest) benchmark runs: one point / bar per
experiment, i.e. per optimization policy.

    uv run python analysis/abacus.py
    uv run python analysis/abacus.py --ids 2026-10-19-abacus-max-quality-01 2026-10-19-abacus-min-time-01

By default analyses every experiment config under
experiments/experiment-configs/benchmark/abacus/ that has finished
(`status: success` in its run.json). Reads each run's metrics.json (written by
experiments/run_benchmark.py) and writes to `--out-dir` (default
experiments/results/analysis/abacus/, gitignored):

    summary.csv           one row per run: policy, accuracy, time, tokens, optimization spend
    time_vs_accuracy.png  accuracy vs. mean per-document call latency
    tokens_vs_accuracy.png  accuracy vs. mean tokens (prompt + completion) per document
    model_usage.png       share of model calls going to each model, per policy

"Time" is the sum of per-call latencies per document (`abacus_mean_wall_seconds`),
a compute-time measure. Palimpzest runs operators concurrently, so real
batch time (`abacus_elapsed_seconds`) is lower and is reported in summary.csv.
Optimization-phase spend (`abacus_fit_*`) is kept out of the plotted inference
numbers and reported separately in summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "experiments" / "experiment-configs" / "benchmark" / "abacus"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results" / "benchmark"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "results" / "analysis" / "abacus"

METHOD = "abacus"
CALLS_PREFIX = f"{METHOD}_model_calls_"

# dataviz reference palette (light mode), matching experiments/vrdu_analysis.py.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
CATEGORICAL = ["#2a78d6", "#eb6834", "#2f9e77", "#8e5bd1", "#a9a8a2"]
BLUE = CATEGORICAL[0]


def policy_label(config: dict) -> str:
    spec = config["params"]["method_kwargs"][METHOD]["policy"]
    args = ", ".join(f"{k}={v}" for k, v in spec.items() if k != "name")
    return f"{spec['name']}({args})" if args else spec["name"]


def load_run(run_dir: Path) -> dict | None:
    """One row for a finished run, else None (missing / not successful)."""
    manifest_path, metrics_path = run_dir / "run.json", run_dir / "metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        return None
    if json.loads(manifest_path.read_text()).get("status") != "success":
        return None
    metrics = json.loads(metrics_path.read_text())
    config = yaml.safe_load((run_dir / "config.snapshot.yaml").read_text())
    return {
        "id": config["id"],
        "policy": policy_label(config),
        "accuracy": metrics[f"{METHOD}_accuracy"],
        "mean_wall_seconds": metrics[f"{METHOD}_mean_wall_seconds"],
        "elapsed_seconds": metrics[f"{METHOD}_elapsed_seconds"],
        "n_documents": metrics[f"{METHOD}_n_documents"],
        "mean_tokens": metrics[f"{METHOD}_mean_prompt_tokens"] + metrics[f"{METHOD}_mean_completion_tokens"],
        "model_calls": {k[len(CALLS_PREFIX):]: v for k, v in metrics.items() if k.startswith(CALLS_PREFIX)},
        "fit_seconds": metrics.get(f"{METHOD}_fit_optimization_seconds"),
        "fit_tokens": (metrics.get(f"{METHOD}_fit_optimization_input_tokens") or 0)
        + (metrics.get(f"{METHOD}_fit_optimization_output_tokens") or 0),
    }


def load_runs(ids: list[str], results_root: Path) -> list[dict]:
    runs = []
    for run_id in ids:
        row = load_run(results_root / run_id)
        if row is None:
            print(f"  skipping {run_id}: no successful run under {results_root / run_id}")
        else:
            runs.append(row)
    return runs


def config_ids(config_dir: Path) -> list[str]:
    return sorted(p.stem for p in config_dir.glob("*.yaml"))


def _style(ax, grid_axis: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def plot_scatter(runs: list[dict], x_key: str, x_label: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.scatter([r[x_key] for r in runs], [r["accuracy"] for r in runs], color=BLUE, s=60, zorder=3)
    for r in runs:
        ax.annotate(r["policy"], (r[x_key], r["accuracy"]), textcoords="offset points", xytext=(6, 5), fontsize=8, color=INK_2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("field accuracy")
    ax.set_title(title, loc="left")
    _style(ax, "both")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_model_usage(runs: list[dict], path: Path) -> None:
    """Stacked horizontal bars: share of each policy's model calls per model.
    Composite pz operators (mixture-of-agents, critic) appear as `pz:<op>`."""
    models = sorted({m for r in runs for m in r["model_calls"]})
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(runs) + 2))
    fig.patch.set_facecolor(SURFACE)
    left = [0.0] * len(runs)
    for i, model in enumerate(models):
        shares = []
        for r in runs:
            total = sum(r["model_calls"].values())
            shares.append(r["model_calls"].get(model, 0) / total if total else 0.0)
        ax.barh([r["policy"] for r in runs], shares, left=left, color=CATEGORICAL[i % len(CATEGORICAL)], label=model, edgecolor=SURFACE, linewidth=1)
        left = [a + b for a, b in zip(left, shares)]
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of model calls")
    ax.set_title("Per-model usage by optimization policy", loc="left")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(len(models), 4))
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_summary(runs: list[dict], path: Path) -> None:
    columns = ["id", "policy", "accuracy", "mean_wall_seconds", "elapsed_seconds", "mean_tokens", "n_documents", "fit_seconds", "fit_tokens"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


def analyze(ids: list[str], results_root: Path, out_dir: Path) -> list[dict]:
    runs = load_runs(ids, results_root)
    if not runs:
        raise SystemExit(f"No finished Abacus runs among {len(ids)} experiment id(s) under {results_root}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(runs, out_dir / "summary.csv")
    plot_scatter(runs, "mean_wall_seconds", "mean call latency per document (s, summed over calls)", "Accuracy vs. time", out_dir / "time_vs_accuracy.png")
    plot_scatter(runs, "mean_tokens", "mean tokens per document (prompt + completion)", "Accuracy vs. tokens", out_dir / "tokens_vs_accuracy.png")
    plot_model_usage(runs, out_dir / "model_usage.png")
    print(f"wrote summary.csv and 3 figures for {len(runs)} run(s) -> {out_dir}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", default=None, help="Experiment ids (default: every config in --config-dir)")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    analyze(args.ids or config_ids(args.config_dir), args.results_root, args.out_dir)


if __name__ == "__main__":
    main()
