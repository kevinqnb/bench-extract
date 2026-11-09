"""CUAD's span-matching rule.

Ported from the official evaluator
(https://github.com/TheAtticusProject/cuad, `evaluate.py`: `get_jaccard` and
`compute_precision_recall`), unchanged in behaviour -- including its
quirks (`split(" ")` rather than `split()`, only `.,;:` stripped, "/" treated
as a separator). The upstream repo ships no LICENSE file for its code; the
data is CC BY 4.0.

The official metrics (AUPR, precision at 80/90% recall) sweep a confidence
threshold over scored model spans, which text-only LLM extraction does not
produce. What is reused is the per-(contract, category) match rule at a fixed
operating point: `score_prediction` is True iff `compute_precision_recall`
would count zero false negatives and zero false positives for that one
(contract, category) -- i.e. the prediction set and the ground-truth set
match one-to-one-ish under the Jaccard rule. A contract with no labelled
span for a category is correct iff nothing is predicted (a false positive
upstream).

Consumed only through experiments/dataset-configs/cuad.py's `score_field`.
"""

from __future__ import annotations

IOU_THRESH = 0.5  # evaluate.py's IOU_THRESH; the match is Jaccard >= this
# Categories (CUAD names, as in the qa ids) for which evaluate.py also accepts
# "ground truth is a substring of the prediction" (it tests `"Parties" in key`).
SUBSTRING_OK_CATEGORIES = frozenset({"Parties"})


def get_jaccard(gt: str, pred: str) -> float:
    for token in (".", ",", ";", ":"):
        gt = gt.replace(token, "")
        pred = pred.replace(token, "")
    gt = gt.lower().replace("/", " ")
    pred = pred.lower().replace("/", " ")
    gt_words = set(gt.split(" "))
    pred_words = set(pred.split(" "))
    return len(gt_words & pred_words) / len(gt_words | pred_words)


def _is_match(answer: str, pred: str, substr_ok: bool) -> bool:
    if get_jaccard(answer, pred) >= IOU_THRESH:
        return True
    return substr_ok and answer in pred


def score_prediction(predicted: list[str], ground_truth: list[str], category: str) -> bool:
    """True iff every ground-truth span is matched by some prediction (no false
    negative) and every prediction matches some ground-truth span (no false
    positive). Empty-string predictions are ignored, as in evaluate.py."""
    substr_ok = category in SUBSTRING_OK_CATEGORIES
    preds = [p for p in predicted if p != ""]
    if not ground_truth:
        return not preds
    return all(any(_is_match(a, p, substr_ok) for p in preds) for a in ground_truth) and all(
        any(_is_match(a, p, substr_ok) for a in ground_truth) for p in preds
    )
