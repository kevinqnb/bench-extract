"""Unified, cross-dataset accuracy measure (`params.scoring.measure: unified`).

The native measures (`datasets/matching.py` for VRDU, `datasets/cuad_matching.py`
for CUAD) are ports of each benchmark's own evaluator and stay the default.
This module is the opt-in alternative: one dataset-agnostic definition of
"correct" so accuracy numbers can be compared across datasets. Dataset
modules only *declare* things -- each field's type (`FIELD_TYPES`), and for
VRDU which fields form line-item records -- and this module does all the
scoring.

Per (document, scored field) the score is a soft F1 in [0, 1] (or None, see
below), computed in three layers:

1. **Value similarity** of one predicted string against one ground-truth
   string, by the field's declared type (`FIELD_TYPES`):
     date        exact match after parsing to (year, month, day); every
                 component the ground truth has must be present and equal in
                 the prediction. Month-first, no MM/DD swap. Unparsable on
                 either side -> falls back to `identifier`.
     quantity    exact match after parsing the (single) number to a Decimal
                 quantized to `quantity_decimals` places; "(x)" / "-x" are
                 negative. Unparsable / several numbers -> `identifier`.
     identifier  exact match after `normalize_text` with all whitespace removed.
     short_text  1 - normalized Levenshtein distance of the normalized strings
                 (rapidfuzz.distance.Levenshtein.normalized_similarity -- true
                 Levenshtein, not fuzz.ratio's Indel distance).
     long_text   token-level F1 of the normalized strings' token multisets.
   Ground truth is a list of *entities*, each a list of acceptable surface
   forms (VRDU annotates every on-page appearance of an unrepeated field as one
   entity); a prediction's similarity to an entity is the max over its forms.

2. **Record similarity** (line items, e.g. VRDU ad-buy-form's
   channel/program_desc/program_start_date/program_end_date/sub_amount rows,
   scored together as one record field like upstream VRDU does): the mean of
   the typed sub-field similarities, skipping sub-fields empty on both sides; a
   sub-field present on only one side scores 0. Models return records
   directly (VRDU's `line_items` extraction field is a list of row objects,
   cleaned by `coerce_records`), so row alignment is explicit.

3. **List matching**: a (n_pred x n_gt) similarity matrix, cells below
   `match_threshold` zeroed, then a maximum-weight one-to-one assignment
   (scipy.optimize.linear_sum_assignment). Soft F1: TP = sum of the assigned
   similarities, precision = TP / n_pred, recall = TP / n_gt.

Empty cases (after dropping predictions / surface forms that normalize to
""): exactly one side empty -> 0.0 (a hallucinated or a missed value);
**both empty -> None**, i.e. the cell is not scored at all and every
consumer leaves it out of its averages.
"""

from __future__ import annotations

import datetime
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional

import numpy as np
from dateutil import parser as date_parser
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

DATE = "date"
QUANTITY = "quantity"
IDENTIFIER = "identifier"
SHORT_TEXT = "short_text"
LONG_TEXT = "long_text"
FIELD_TYPES = (DATE, QUANTITY, IDENTIFIER, SHORT_TEXT, LONG_TEXT)
#: Pseudo-type reported for a record (line-item) field (accuracy_type_record).
RECORD = "record"

NATIVE = "native"
UNIFIED = "unified"
MEASURES = (NATIVE, UNIFIED)
_UNIFIED_KEYS = ("match_threshold", "quantity_decimals")


@dataclass(frozen=True)
class ScoringConfig:
    """Parsed `params.scoring`. Absent block => native. The unified measure's
    knobs have no code defaults -- they must be set in the experiment config."""

    measure: str = NATIVE
    match_threshold: Optional[float] = None
    quantity_decimals: Optional[int] = None

    @classmethod
    def from_params(cls, raw: Optional[dict]) -> "ScoringConfig":
        if raw is None:
            return cls()
        raw = dict(raw)
        measure = raw.pop("measure", None)
        if measure not in MEASURES:
            raise ValueError(f"params.scoring.measure must be one of {MEASURES}, got {measure!r}")
        if measure == NATIVE:
            if raw:
                raise ValueError(f"params.scoring: measure 'native' takes no other keys, got {sorted(raw)}")
            return cls()
        missing = [k for k in _UNIFIED_KEYS if k not in raw]
        unknown = sorted(set(raw) - set(_UNIFIED_KEYS))
        if missing or unknown:
            raise ValueError(f"params.scoring (unified): missing={missing} unknown={unknown}; expected exactly {list(_UNIFIED_KEYS)}")
        threshold = float(raw["match_threshold"])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"params.scoring.match_threshold must be in [0, 1], got {threshold}")
        decimals = raw["quantity_decimals"]
        if not isinstance(decimals, int) or decimals < 0:
            raise ValueError(f"params.scoring.quantity_decimals must be a non-negative int, got {decimals!r}")
        return cls(measure=UNIFIED, match_threshold=threshold, quantity_decimals=decimals)


# --- normalization ---------------------------------------------------------


def normalize_text(text: str) -> str:
    """NFKC, casefold, every punctuation/symbol character -> space, whitespace
    collapsed. `normalize_text(s).split()` therefore never yields an empty
    token (unlike CUAD's `split(" ")`)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def is_empty(text: str) -> bool:
    return not normalize_text(text)


# --- value similarity by type ----------------------------------------------

# Two different fill-in defaults: a component dateutil filled from the default
# (i.e. absent from the string) differs between the two parses. Fixed dates,
# never "today", so a score can't depend on when it was computed.
_DATE_DEFAULTS = (datetime.datetime(1904, 1, 1), datetime.datetime(1908, 2, 2))
_DATE_COMPONENTS = ("year", "month", "day")


def parse_date(text: str) -> Optional[dict[str, int]]:
    """{'year'?, 'month'?, 'day'?} with only the components actually present in
    `text`, or None if it doesn't parse as a date."""
    try:
        a, b = (date_parser.parse(text, default=d, fuzzy=True, dayfirst=False) for d in _DATE_DEFAULTS)
    except (ValueError, OverflowError, TypeError):
        return None
    parsed = {c: getattr(a, c) for c in _DATE_COMPONENTS if getattr(a, c) == getattr(b, c)}
    return parsed or None


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_NEGATIVE_PREFIX_RE = re.compile(r"(?:^|[^\w])[-(][^\w\s]?$")


def parse_quantity(text: str, decimals: int) -> Optional[Decimal]:
    """The single number in `text` (currency symbols, spaces, thousands
    separators ignored) quantized to `decimals` places, or None when there is
    no number, more than one, or a malformed thousands grouping."""
    text = unicodedata.normalize("NFKC", text).strip()
    matches = list(_NUMBER_RE.finditer(text))
    if len(matches) != 1:
        return None
    number = matches[0].group()
    if "," in number and not _THOUSANDS_RE.match(number):
        return None
    try:
        value = Decimal(number.replace(",", ""))
    except InvalidOperation:
        return None
    # Negative iff "-" or "(" is attached to the number, optionally through one
    # currency symbol ("-5", "-$5", "($5)"), and is not inside a word
    # ("M-F 5"). A free-standing separator ("Total - $600") stays positive.
    if _NEGATIVE_PREFIX_RE.search(text[: matches[0].start()]):
        value = -value
    return value.quantize(Decimal(1).scaleb(-decimals))


def identifier_similarity(pred: str, gt: str) -> float:
    a, b = normalize_text(pred).replace(" ", ""), normalize_text(gt).replace(" ", "")
    return float(bool(a) and a == b)


def date_similarity(pred: str, gt: str) -> float:
    p, g = parse_date(pred), parse_date(gt)
    if p is None or g is None:
        return identifier_similarity(pred, gt)
    return float(all(c in p and p[c] == g[c] for c in g))


def quantity_similarity(pred: str, gt: str, decimals: int) -> float:
    p, g = parse_quantity(pred, decimals), parse_quantity(gt, decimals)
    if p is None or g is None:
        return identifier_similarity(pred, gt)
    return float(p == g)


def short_text_similarity(pred: str, gt: str) -> float:
    return float(Levenshtein.normalized_similarity(normalize_text(pred), normalize_text(gt)))


def long_text_similarity(pred: str, gt: str) -> float:
    p, g = Counter(tokens(pred)), Counter(tokens(gt))
    overlap = sum((p & g).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * precision * recall / (precision + recall)


def value_similarity(field_type: str, pred: str, gt: str, config: ScoringConfig) -> float:
    if field_type == DATE:
        return date_similarity(pred, gt)
    if field_type == QUANTITY:
        return quantity_similarity(pred, gt, config.quantity_decimals)
    if field_type == IDENTIFIER:
        return identifier_similarity(pred, gt)
    if field_type == SHORT_TEXT:
        return short_text_similarity(pred, gt)
    if field_type == LONG_TEXT:
        return long_text_similarity(pred, gt)
    raise ValueError(f"Unknown field type {field_type!r}; choices: {FIELD_TYPES}")


# --- list matching ---------------------------------------------------------


def hungarian_soft_f1(similarity: np.ndarray, threshold: float) -> float:
    """`similarity` is (n_pred x n_gt), both > 0. Cells below `threshold` can't
    be matched; the max-weight one-to-one assignment's summed similarity is the
    soft true-positive count."""
    sim = np.where(similarity >= threshold, similarity, 0.0)
    rows, cols = linear_sum_assignment(sim, maximize=True)
    tp = float(sim[rows, cols].sum())
    if tp == 0.0:
        return 0.0
    n_pred, n_gt = sim.shape
    precision, recall = tp / n_pred, tp / n_gt
    return 2 * precision * recall / (precision + recall)


def score_entities(
    predicted: list[str], gt_entities: list[list[str]], field_type: str, config: ScoringConfig
) -> Optional[float]:
    """Soft F1 of a field's predicted values against its ground-truth entities
    (each a list of acceptable surface forms). None when both sides are empty."""
    preds = [p for p in predicted if not is_empty(p)]
    entities = [forms for forms in ([f for f in e if not is_empty(f)] for e in gt_entities) if forms]
    if not preds and not entities:
        return None
    if not preds or not entities:
        return 0.0
    sim = np.array(
        [[max(value_similarity(field_type, p, f, config) for f in forms) for forms in entities] for p in preds]
    )
    return hungarian_soft_f1(sim, config.match_threshold)


def coerce_records(value: Any, sub_fields: list[str]) -> list[dict[str, str]]:
    """A model's record-shaped output (e.g. VRDU `line_items`: a list of row
    objects) as clean records: non-dict entries dropped, every sub-field
    present as a string ("" when missing or null), unknown keys dropped."""
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [
        {f: ("" if row.get(f) is None else str(row.get(f))) for f in sub_fields}
        for row in value
        if isinstance(row, dict)
    ]


def record_similarity(pred: dict[str, str], gt: dict[str, str], sub_field_types: dict[str, str], config: ScoringConfig) -> float:
    scores = []
    for sub_field, field_type in sub_field_types.items():
        p, g = pred.get(sub_field, ""), gt.get(sub_field, "")
        p_empty, g_empty = is_empty(p), is_empty(g)
        if p_empty and g_empty:
            continue
        scores.append(0.0 if p_empty or g_empty else value_similarity(field_type, p, g, config))
    return sum(scores) / len(scores) if scores else 0.0


def score_records(
    predicted: list[dict[str, str]], gt_records: list[dict[str, str]], sub_field_types: dict[str, str], config: ScoringConfig
) -> Optional[float]:
    """Soft F1 of predicted line-item records against ground-truth records.
    None when both sides have no non-empty record."""
    nonempty = lambda r: any(not is_empty(r.get(f, "")) for f in sub_field_types)  # noqa: E731
    preds = [r for r in predicted if nonempty(r)]
    gts = [r for r in gt_records if nonempty(r)]
    if not preds and not gts:
        return None
    if not preds or not gts:
        return 0.0
    sim = np.array([[record_similarity(p, g, sub_field_types, config) for g in gts] for p in preds])
    return hungarian_soft_f1(sim, config.match_threshold)


# --- factory ---------------------------------------------------------------


def make_unified_scorer(
    field_types: dict[str, str], record_fields: dict[str, list[str]], config: ScoringConfig
) -> Callable[[Any, Any, str], Optional[float]]:
    """`score_field(predicted, ground_truth, scored_field)` for the unified
    measure. For a field in `record_fields` (record field -> sub-fields),
    predicted/ground_truth are lists of records ({sub_field: str}); otherwise
    predicted is a list[str] and ground_truth a list of entities (list[list[str]]).
    `field_types` must cover every plain field and every record sub-field."""
    if config.measure != UNIFIED:
        raise ValueError("make_unified_scorer needs a unified ScoringConfig")
    bad = {f: t for f, t in field_types.items() if t not in FIELD_TYPES}
    if bad:
        raise ValueError(f"Unknown field types {bad}; choices: {FIELD_TYPES}")
    for record_field, sub_fields in record_fields.items():
        missing = [f for f in sub_fields if f not in field_types]
        if missing:
            raise ValueError(f"record field {record_field!r}: sub-fields without a declared type: {missing}")
    sub_field_types = {r: {f: field_types[f] for f in subs} for r, subs in record_fields.items()}

    def score_field(predicted, ground_truth, field: str) -> Optional[float]:
        if field in sub_field_types:
            return score_records(predicted, ground_truth, sub_field_types[field], config)
        if field not in field_types:
            raise KeyError(f"No declared field type for {field!r}")
        return score_entities(predicted, ground_truth, field_types[field], config)

    return score_field


def unified_context_kwargs(
    fields: list[str],
    field_types: dict[str, str],
    record_fields: dict[str, list[str]],
    config: ScoringConfig,
    entity_ground_truth: Callable[[Any, str], list[list[str]]],
    record_ground_truth: Optional[Callable[[Any, str], list[dict[str, str]]]] = None,
) -> dict[str, Any]:
    """The unified-measure keyword arguments for a DatasetRunContext
    (scored_fields, extraction_field_of, prediction_for, ground_truth,
    score_field, scored_field_types, scoring_measure), shared by every dataset
    module. Scored fields are the extraction fields themselves; a record field
    is scored as a list of records.

    fields:              extraction fields (record fields included)
    field_types:         FIELD_TYPES entry for every plain field and every record sub-field
    record_fields:       record extraction field -> its sub-fields ({} if none)
    entity_ground_truth: (doc, field) -> list of entities (each a list of surface forms)
    record_ground_truth: (doc, record field) -> list of records; required iff record_fields
    """
    unknown_records = set(record_fields) - set(fields)
    if unknown_records:
        raise ValueError(f"record fields not among the extraction fields: {sorted(unknown_records)}")
    if record_fields and record_ground_truth is None:
        raise ValueError("record_fields given without a record_ground_truth accessor")
    expected = {f for f in fields if f not in record_fields} | {f for subs in record_fields.values() for f in subs}
    if set(field_types) != expected:
        raise ValueError(
            "field types must cover exactly the plain extraction fields and the record sub-fields -- "
            f"missing={sorted(expected - set(field_types))} extra={sorted(set(field_types) - expected)}"
        )

    def prediction_for(field_values: dict[str, list], scored_field: str):
        if scored_field in record_fields:
            return coerce_records(field_values.get(scored_field), record_fields[scored_field])
        return [str(v) for v in (field_values.get(scored_field) or []) if v is not None]

    def ground_truth(doc: Any, scored_field: str):
        if scored_field in record_fields:
            return record_ground_truth(doc, scored_field)
        return entity_ground_truth(doc, scored_field)

    return {
        "scored_fields": list(fields),
        "extraction_field_of": {f: f for f in fields},
        "prediction_for": prediction_for,
        "ground_truth": ground_truth,
        "score_field": make_unified_scorer(field_types, record_fields, config),
        "scored_field_types": {f: (RECORD if f in record_fields else field_types[f]) for f in fields},
        "scoring_measure": UNIFIED,
    }
