"""CUAD counterpart of analysis/abacus_compare.py: time vs. accuracy and per-model
usage for finished Abacus runs on CUAD. Defaults to the MaxQuality vs. MinTime pair.

    uv run python analysis/abacus_cuad_compare.py
    uv run python analysis/abacus_cuad_compare.py --ids <id> <id> ...

Same inputs/outputs as abacus_compare.py (summary.csv, usage.csv, time_vs_accuracy.png,
model_usage.png, written to analysis/results/abacus_cuad_compare/), reusing its loaders and
usage plot. What differs is the accuracy measure, so the accuracy axis is labelled and two
reference lines drawn, both computed from the data on the runs' validation split:

  * accuracy is the fraction of (contract, category) pairs where the predicted span set
    matches the ground-truth set one-to-one under CUAD's Jaccard >= 0.5 rule
    (datasets/cuad_matching.py); a category with no labelled span is correct iff nothing
    is predicted. It is NOT the official AUPR / precision@recall.
  * "predict nothing" baseline: accuracy of returning [] for every field. CUAD is sparse,
    so this is far from 0 and is the floor to read the runs against.
  * "window ceiling": accuracy of a perfect extractor that sees only the first
    `max_chars` characters (it returns exactly the ground-truth spans lying fully inside the
    window). Late-clause fields (governing_law, warranty_duration) are largely cut off.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Sibling scripts, importable because running this file puts analysis/ on sys.path.
from abacus import CATEGORICAL, DEFAULT_RESULTS_ROOT, SURFACE, _style  # noqa: E402
from abacus_compare import load_runs, plot_model_usage, write_summary, write_usage  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ["2026-11-02-abacus-cuad-max-quality-01", "2026-11-02-abacus-cuad-min-time-01"]
DEFAULT_OUT_DIR = REPO_ROOT / "analysis" / "results" / "abacus_cuad_compare"
ACCURACY_LABEL = "accuracy (share of contract x category pairs matched, Jaccard >= 0.5)"


def reference_accuracies(results_root: Path, run_id: str) -> dict:
    """{'n_pairs', 'predict_nothing', 'window_ceiling', 'max_chars'} on run_id's validation split."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from experiments.config import DATASET_REGISTRY

    config = yaml.safe_load((results_root / run_id / "config.snapshot.yaml").read_text())
    params = config["params"]
    mod = DATASET_REGISTRY[params["dataset"]]
    dp = params["dataset_params"]
    ctx = mod.load_run_context(dp)
    docs = [ctx.all_docs[i] for i in mod.load_split_doc_ids(dp, "valid") if i in ctx.all_docs]
    if params.get("limit"):
        docs = docs[: params["limit"]]
    empty = ceiling = n = 0
    for d in docs:
        window = ctx.windowed_text(d)
        for f in ctx.fields:
            gt = d.fields.get(f, [])
            empty += int(ctx.score_field([], gt, f))
            ceiling += int(ctx.score_field([s for s in gt if s in window], gt, f))
            n += 1
    return {"n_pairs": n, "predict_nothing": empty / n, "window_ceiling": ceiling / n, "max_chars": dp["max_chars"]}


def plot_time_vs_accuracy(runs: list[dict], ref: dict, path: Path) -> None:
    panels = [
        ("mean_wall_seconds", "mean per-document latency (s, summed over calls)"),
        ("elapsed_seconds", "elapsed time for the whole validation set (s, concurrent)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, label) in zip(axes, panels):
        for i, r in enumerate(runs):
            ax.scatter(r[key], r["accuracy"], color=CATEGORICAL[i % len(CATEGORICAL)], s=70, zorder=3, label=r["policy"])
        ax.axhline(ref["predict_nothing"], color="#52514e", linestyle="--", linewidth=1, label="predict nothing")
        ax.axhline(ref["window_ceiling"], color="#52514e", linestyle=":", linewidth=1, label=f"window ceiling ({ref['max_chars']} chars)")
        ax.set_xlabel(label)
        ax.set_xlim(left=0)
        _style(ax, "both")
    axes[0].set_ylabel(ACCURACY_LABEL, fontsize=8)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("CUAD: accuracy vs. time", loc="left")
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def analyze(ids: list[str], results_root: Path, out_dir: Path) -> list[dict]:
    runs = load_runs(ids, results_root)
    if not runs:
        raise SystemExit(f"No finished Abacus runs among {ids} under {results_root}.")
    ref = reference_accuracies(results_root, runs[0]["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in runs:
        r["predict_nothing_accuracy"], r["window_ceiling_accuracy"] = ref["predict_nothing"], ref["window_ceiling"]
    write_summary(runs, out_dir / "summary.csv")
    write_usage(runs, out_dir / "usage.csv")
    plot_time_vs_accuracy(runs, ref, out_dir / "time_vs_accuracy.png")
    plot_model_usage(runs, out_dir / "model_usage.png")
    print(f"predict-nothing={ref['predict_nothing']:.3f} window-ceiling={ref['window_ceiling']:.3f} over {ref['n_pairs']} pairs")
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
