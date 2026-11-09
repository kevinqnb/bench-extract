"""Side-by-side comparison of a few finished Abacus runs: time vs. accuracy, and
per-model usage. Defaults to the MaxQuality vs. MinTime pair.

    uv run python analysis/abacus_compare.py
    uv run python analysis/abacus_compare.py --ids <id> <id> ...

Reads each run's metrics.json / run.json (via analysis/abacus.py's `load_run`) and
artifacts/steps.csv (one row per validation document: the model that served it,
call latency, tokens). Writes to `--out-dir` (default
analysis/results/abacus_compare/):

    summary.csv           one row per run: policy, accuracy, both time measures, optimization spend
    usage.csv             one row per (run, model): calls, summed latency, prompt/completion tokens
    time_vs_accuracy.png  accuracy vs. (left) summed per-call latency per document and
                          (right) real batch time for the whole validation set
    model_usage.png       each model's share of calls, tokens and summed latency, per run

Two time measures, because they differ a lot: `mean_wall_seconds` sums per-call latencies
for a document (compute time); `elapsed_seconds` is the real wall-clock for the batch,
which is lower because Palimpzest runs documents concurrently.

Accuracy is a single number per run (`abacus_accuracy`), so there is one point per run
per panel; steps.csv carries no per-document correctness. Composite pz operators
(mixture-of-agents, critic) show up as `pz:<operator>` in steps.csv and cannot be split
back into the models inside them.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# analysis/abacus.py is a sibling script, importable because running this file puts analysis/ on sys.path.
from abacus import CATEGORICAL, DEFAULT_RESULTS_ROOT, METHOD, SURFACE, _style, load_run  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ["2026-10-19-abacus-max-quality-01", "2026-10-19-abacus-min-time-01"]
DEFAULT_OUT_DIR = REPO_ROOT / "analysis" / "results" / "abacus_compare"

USAGE_MEASURES = [
    ("calls", "share of calls"),
    ("tokens", "share of tokens (prompt + completion)"),
    ("wall_seconds", "share of summed call latency"),
]


def load_usage(run_dir: Path) -> dict[str, dict[str, float]]:
    """model_key -> {calls, wall_seconds, prompt_tokens, completion_tokens, tokens}, from steps.csv."""
    usage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with open(run_dir / "artifacts" / "steps.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["method"] != METHOD:
                continue
            u = usage[row["model_key"]]
            u["calls"] += 1
            u["wall_seconds"] += float(row["wall_seconds"] or 0)
            u["prompt_tokens"] += float(row["prompt_tokens"] or 0)
            u["completion_tokens"] += float(row["completion_tokens"] or 0)
            u["tokens"] = u["prompt_tokens"] + u["completion_tokens"]
    return {k: dict(v) for k, v in usage.items()}


def load_runs(ids: list[str], results_root: Path) -> list[dict]:
    runs = []
    for run_id in ids:
        run = load_run(results_root / run_id)
        if run is None:
            print(f"  skipping {run_id}: no successful run under {results_root / run_id}")
            continue
        run["usage"] = load_usage(results_root / run_id)
        runs.append(run)
    return runs


def plot_time_vs_accuracy(runs: list[dict], path: Path) -> None:
    panels = [
        ("mean_wall_seconds", "mean per-document latency (s, summed over calls)"),
        ("elapsed_seconds", "elapsed time for the whole validation set (s, concurrent)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label) in zip(axes, panels):
        for i, r in enumerate(runs):
            ax.scatter(r[key], r["accuracy"], color=CATEGORICAL[i % len(CATEGORICAL)], s=70, zorder=3, label=r["policy"])
        ax.set_xlabel(label)
        ax.set_xlim(left=0)
        _style(ax, "both")
    axes[0].set_ylabel("field accuracy")
    axes[0].set_title("Accuracy vs. time", loc="left")
    axes[1].legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_model_usage(runs: list[dict], path: Path) -> None:
    """One stacked-bar panel per usage measure; a model keeps its color across panels."""
    models = sorted({m for r in runs for m in r["usage"]})
    colors = {m: CATEGORICAL[i % len(CATEGORICAL)] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, len(USAGE_MEASURES), figsize=(12, 0.7 * len(runs) + 2.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    # Numeric positions, not label strings: matplotlib merges bars that share a categorical label.
    ypos = list(range(len(runs)))
    for ax, (measure, title) in zip(axes, USAGE_MEASURES):
        left = [0.0] * len(runs)
        for model in models:
            shares = []
            for r in runs:
                total = sum(u[measure] for u in r["usage"].values())
                shares.append(r["usage"].get(model, {}).get(measure, 0.0) / total if total else 0.0)
            ax.barh(ypos, shares, left=left, color=colors[model], label=model, edgecolor=SURFACE, linewidth=1)
            left = [a + b for a, b in zip(left, shares)]
        ax.set_yticks(ypos, [r["policy"] for r in runs])
        ax.set_xlim(0, 1)
        ax.set_xlabel(title)
        ax.invert_yaxis()
        _style(ax, "x")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=min(len(models), 4))
    fig.suptitle("Per-model usage by run", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_summary(runs: list[dict], path: Path) -> None:
    columns = ["id", "policy", "accuracy", "mean_wall_seconds", "elapsed_seconds", "mean_tokens", "n_documents", "fit_seconds", "fit_tokens"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


def write_usage(runs: list[dict], path: Path) -> None:
    columns = ["id", "policy", "model", "calls", "wall_seconds", "prompt_tokens", "completion_tokens"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in runs:
            for model, u in sorted(r["usage"].items()):
                writer.writerow({"id": r["id"], "policy": r["policy"], "model": model, **u})


def analyze(ids: list[str], results_root: Path, out_dir: Path) -> list[dict]:
    runs = load_runs(ids, results_root)
    if not runs:
        raise SystemExit(f"No finished Abacus runs among {ids} under {results_root}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(runs, out_dir / "summary.csv")
    write_usage(runs, out_dir / "usage.csv")
    plot_time_vs_accuracy(runs, out_dir / "time_vs_accuracy.png")
    plot_model_usage(runs, out_dir / "model_usage.png")
    print(f"wrote summary.csv, usage.csv and 2 figures for {len(runs)} run(s) -> {out_dir}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", default=DEFAULT_IDS, help="Experiment ids to compare")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    analyze(args.ids, args.results_root, args.out_dir)


if __name__ == "__main__":
    main()
