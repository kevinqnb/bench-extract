"""Descriptive plots of the VRDU dataset (data/vrdu/data.json).

    uv run python -m experiments.vrdu_analysis --corpus ad-buy-form

Writes PNGs plus `summary.json` to `--out-dir` (default
experiments/results/analysis/vrdu/<corpus>/, gitignored). This is a
descriptive script, not a benchmark run, so it has no experiment config; the
only dataset-specific numbers it uses (which fields exist, their match
functions, the OCR window) are read from dataset-configs/vrdu.py's CORPORA.

Figures:
    doc_structure.png    pages per document; OCR length per document vs. the prompt window
    field_coverage.png   share of documents annotated for each field
    field_spans.png      annotated spans per document, per field (repeated mentions)
    gt_location.png     where each field's ground truth sits relative to the OCR window
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bench_extract.datasets.vrdu import VRDUDocument, load_documents  # noqa: E402
from experiments.config import DATASET_REGISTRY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# dataviz reference palette (light mode): sequential/categorical blue, orange, neutral gray.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a9a8a2"


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


def _fig(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _nonempty(values: list[str]) -> list[str]:
    return [v.strip() for v in values if v.strip()]


def plot_doc_structure(docs: list[VRDUDocument], max_chars: int, max_pages: int, path: Path) -> None:
    pages = Counter(len(d.pages) for d in docs)
    chars = np.array([len(d.ocr_text) for d in docs])
    fig, (a1, a2) = _fig(1, 2, figsize=(11, 4))

    xs = sorted(pages)
    a1.bar(xs, [pages[x] for x in xs], color=BLUE, width=0.8)
    a1.axvline(max_pages + 0.5, color=ORANGE, linewidth=2)
    a1.text(max_pages + 0.6, a1.get_ylim()[1] * 0.95, f"window: first {max_pages} pages", color=INK_2, va="top", fontsize=9)
    a1.set_xlabel("pages per document")
    a1.set_ylabel("documents")
    a1.set_title("Document length (pages)", loc="left")
    _style(a1, "y")

    a2.hist(chars, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1)
    a2.axvline(max_chars, color=ORANGE, linewidth=2)
    over = float((chars > max_chars).mean())
    a2.text(max_chars, a2.get_ylim()[1] * 0.95, f" window: {max_chars:,} chars\n {over:.0%} of docs longer", color=INK_2, va="top", fontsize=9)
    a2.set_xlabel("OCR characters per document")
    a2.set_ylabel("documents")
    a2.set_title("OCR text length", loc="left")
    _style(a2, "y")

    fig.suptitle(f"{len(docs)} documents, median {int(np.median(chars)):,} OCR chars", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_field_coverage(docs, match_funcs: dict[str, str], path: Path) -> dict[str, float]:
    coverage = {f: sum(bool(_nonempty(d.fields.get(f, []))) for d in docs) / len(docs) for f in match_funcs}
    order = sorted(coverage, key=coverage.get)
    fig, ax = _fig(figsize=(8, 0.42 * len(order) + 1.5))
    ax.barh(order, [coverage[f] for f in order], color=BLUE, height=0.65)
    for i, f in enumerate(order):
        ax.text(coverage[f] + 0.01, i, f"{coverage[f]:.0%}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(order)), [f"{f}  ({match_funcs[f]})" for f in order])
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("share of documents with the field annotated")
    ax.set_title("Field coverage", loc="left")
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return coverage


def plot_field_spans(docs, match_funcs: dict[str, str], path: Path) -> dict[str, float]:
    counts = {f: [len(_nonempty(d.fields.get(f, []))) for d in docs] for f in match_funcs}
    mean = {f: float(np.mean([c for c in cs if c > 0] or [0])) for f, cs in counts.items()}
    order = sorted(mean, key=mean.get)
    fig, ax = _fig(figsize=(8, 0.42 * len(order) + 1.5))
    ax.barh(order, [mean[f] for f in order], color=BLUE, height=0.65)
    for i, f in enumerate(order):
        ax.text(mean[f] + 0.05, i, f"{mean[f]:.1f}", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, max(mean.values()) * 1.1)
    ax.set_xlabel("mean annotated spans per document (documents where the field is present)")
    ax.set_title("Repeated mentions per field", loc="left")
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return mean


def plot_gt_location(docs, match_funcs: dict[str, str], max_chars: int, path: Path) -> dict[str, dict[str, float]]:
    """Per field, among documents where it is annotated: does any ground-truth
    string occur verbatim within the first `max_chars` OCR chars, only beyond
    it, or not verbatim at all (e.g. the OCR/format differs from the label)?"""
    shares: dict[str, dict[str, float]] = {}
    for f in match_funcs:
        tally = Counter()
        for d in docs:
            values = _nonempty(d.fields.get(f, []))
            if not values:
                continue
            hits = [i for i in (d.ocr_text.find(v) for v in values) if i >= 0]
            if not hits:
                tally["not found verbatim"] += 1
            elif min(hits) + 1 <= max_chars:  # start < max_chars; trailing overrun is negligible
                tally["inside window"] += 1
            else:
                tally["beyond window"] += 1
        n = sum(tally.values()) or 1
        shares[f] = {k: tally[k] / n for k in ("inside window", "beyond window", "not found verbatim")}

    order = sorted(shares, key=lambda f: shares[f]["inside window"])
    fig, ax = _fig(figsize=(9, 0.42 * len(order) + 2))
    left = np.zeros(len(order))
    for label, color in (("inside window", BLUE), ("beyond window", ORANGE), ("not found verbatim", GRAY)):
        w = np.array([shares[f][label] for f in order])
        # 2px surface gap between stacked segments
        ax.barh(order, w, left=left, color=color, height=0.65, label=label, edgecolor=SURFACE, linewidth=2)
        left += w
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of annotated documents")
    ax.set_title(f"Ground truth vs. the {max_chars:,}-char prompt window", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, labelcolor=INK_2)
    _style(ax, "x")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return shares


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, help="dataset-configs/vrdu.py CORPORA key, e.g. ad-buy-form")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = DATASET_REGISTRY["vrdu"].CORPORA[args.corpus]
    out_dir = args.out_dir or REPO_ROOT / "experiments" / "results" / "analysis" / "vrdu" / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(args.corpus)
    if not docs:
        raise SystemExit(f"No documents for corpus {args.corpus!r}; run `uv run data/download_vrdu.py`.")

    summary = {
        "corpus": args.corpus,
        "n_documents": len(docs),
        "max_chars": cfg.max_chars,
        "max_pages": cfg.max_pages,
        "field_coverage": None,
        "mean_spans_when_present": None,
        "gt_location": None,
    }
    plot_doc_structure(docs, cfg.max_chars, cfg.max_pages, out_dir / "doc_structure.png")
    summary["field_coverage"] = plot_field_coverage(docs, cfg.match_func_by_field, out_dir / "field_coverage.png")
    summary["mean_spans_when_present"] = plot_field_spans(docs, cfg.match_func_by_field, out_dir / "field_spans.png")
    summary["gt_location"] = plot_gt_location(docs, cfg.match_func_by_field, cfg.max_chars, out_dir / "gt_location.png")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote 4 figures + summary.json to {out_dir}")


if __name__ == "__main__":
    main()
