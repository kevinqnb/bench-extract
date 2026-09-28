"""VRDU's fuzzy field-matching functions.

Ported from the official evaluator
(https://github.com/google-research/google-research/tree/master/vrdu,
`match_utils.py`; Apache License 2.0, Copyright 2026 The Google Research
Authors: https://www.apache.org/licenses/LICENSE-2.0) and simplified to
operate on plain extracted/ground-truth strings -- this project's
extraction models produce text only, not the original `Entity` type's
bounding box + character-segment metadata, so those fields are dropped. The
matching algorithms themselves (date decoding, price tolerance, edit
distance, ...) are unchanged from the original.

Edit distance uses rapidfuzz instead of the original's `editdistance`
package (equivalent Levenshtein distance), to avoid a second fuzzy-matching
dependency.

These functions define what "correct" means for every routing/cascade/
scheduling method's training labels (see profiling.py) -- adapted from the
paper's own evaluator rather than reinvented, so accuracy numbers here stay
comparable to VRDU's published results at the per-field level.
"""

from __future__ import annotations

import datetime
import re

from rapidfuzz.distance import Levenshtein


def remove_redundant_whitespace(text: str) -> str:
    """' abc\\ndef   ghi\\t' -> 'abc def ghi'."""
    return " ".join(part.strip() for part in text.strip().split())


def match_by_alpha_numeric_text(a: str, b: str) -> bool:
    """Equal after stripping everything but letters/digits, e.g. 'Xy_Z1 2@3' == 'XyZ123'."""
    strip = lambda s: re.sub(r"[^0-9a-zA-Z]", "", s)
    return strip(a) == strip(b)


def match_by_non_whitespace_text(a: str, b: str) -> bool:
    """Equal after stripping whitespace, e.g. 'X y Z\\t1 2 3' == 'XyZ123'."""
    strip = lambda s: re.sub(r"\s", "", s)
    return strip(a) == strip(b)


def match_by_numeric_text(a: str, b: str) -> bool:
    """Equal after stripping everything but digits, e.g. 'Xy_Z1 2@3' == '1xx2yy3zz'."""
    strip = lambda s: re.sub(r"[^0-9]", "", s)
    return strip(a) == strip(b)


def match_by_value(a: str, b: str, diff: float = 0.01) -> bool:
    """Numerically equal within `diff` after stripping non-numeric characters
    (keeping '.'), e.g. '$3.14' == '3.1415926'."""
    to_num = lambda s: re.sub(r"[^0-9.]", "", s)
    try:
        return abs(float(to_num(a)) - float(to_num(b))) <= diff
    except ValueError:
        return False


def match_by_edit_distance(a: str, b: str, threshold: int = 3) -> bool:
    return Levenshtein.distance(a, b) <= threshold


# Patterns tried in order to decode a date string; the first one that parses wins.
_DATE_PATTERNS = [
    "%m/%d/%y",   # 07/01/22
    "%m/%d/%Y",   # 07/01/2022
    "%m/%d",      # 07/01 (year defaults to 1900)
    "%b%d/%y",    # Jul01/22
    "%m-%d-%Y",   # 07-01-2022
    "%m-%d-%y",   # 07-01-22
    "%B%d,%Y",    # July01,2022
    "%Y/%m/%d",   # 2022/07/01
]


def decode_date(date_string: str) -> dict[str, int] | None:
    """Extracts {'year', 'month', 'day'} from a date string, or None if no
    known pattern matches."""
    proc = re.sub(r"[^0-9a-zA-Z/\-,]", "", date_string)
    for pattern in _DATE_PATTERNS:
        try:
            parsed = datetime.datetime.strptime(proc, pattern).date()
            return {"year": parsed.year, "month": parsed.month, "day": parsed.day}
        except ValueError:
            continue
    return None


class Match:
    """Base for a specific {Type}Match. `match(extracted_text, labeled_texts)`
    returns True if `extracted_text` matches any one of `labeled_texts` --
    a field may appear multiple times in a document (e.g. a repeated
    line-item), and the model only needs to have extracted one of them."""

    @classmethod
    def match(cls, extracted_text: str, labeled_texts: list[str]) -> bool:
        raise NotImplementedError


class DateMatch(Match):
    """Two dates match if they share year/month/day, or share the year with
    month/day swapped (MM/DD vs. DD/MM format ambiguity)."""

    @classmethod
    def match(cls, extracted_text: str, labeled_texts: list[str]) -> bool:
        extracted = remove_redundant_whitespace(extracted_text)
        extracted_date = decode_date(extracted)
        for labeled_text in labeled_texts:
            labeled = remove_redundant_whitespace(labeled_text)
            labeled_date = decode_date(labeled)
            if not extracted_date or not labeled_date:
                if match_by_alpha_numeric_text(extracted, labeled):
                    return True
                continue
            if (
                extracted_date["year"] == labeled_date["year"]
                and extracted_date["month"] == labeled_date["month"]
                and extracted_date["day"] == labeled_date["day"]
            ):
                return True
            if (
                extracted_date["year"] == labeled_date["year"]
                and extracted_date["day"] == labeled_date["month"]
                and extracted_date["month"] == labeled_date["day"]
            ):
                return True
        return False


class PriceMatch(Match):
    """Numeric value within tolerance, e.g. '$3.14' == '3.14'."""

    @classmethod
    def match(cls, extracted_text: str, labeled_texts: list[str]) -> bool:
        extracted = remove_redundant_whitespace(extracted_text)
        return any(match_by_value(extracted, remove_redundant_whitespace(t)) for t in labeled_texts)


class AddressMatch(Match):
    """Small edit distance (<=3 by default)."""

    @classmethod
    def match(cls, extracted_text: str, labeled_texts: list[str]) -> bool:
        extracted = remove_redundant_whitespace(extracted_text)
        return any(match_by_edit_distance(extracted, remove_redundant_whitespace(t)) for t in labeled_texts)


class NumericalStringMatch(Match):
    """Equal after keeping only digits."""

    @classmethod
    def match(cls, extracted_text: str, labeled_texts: list[str]) -> bool:
        extracted = remove_redundant_whitespace(extracted_text)
        return any(match_by_numeric_text(extracted, remove_redundant_whitespace(t)) for t in labeled_texts)


class GeneralStringMatch(Match):
    """Equal after keeping only alphanumerics."""

    @classmethod
    def match(cls, extracted_text: str, labeled_texts: list[str]) -> bool:
        extracted = remove_redundant_whitespace(extracted_text)
        return any(match_by_alpha_numeric_text(extracted, remove_redundant_whitespace(t)) for t in labeled_texts)


class NameMatch(Match):
    """Equal after stripping whitespace."""

    @classmethod
    def match(cls, extracted_text: str, labeled_texts: list[str]) -> bool:
        extracted = remove_redundant_whitespace(extracted_text)
        return any(match_by_non_whitespace_text(extracted, remove_redundant_whitespace(t)) for t in labeled_texts)


class DefaultMatch(Match):
    """Strict equality after whitespace normalization. Used for any field
    whose meta.json match-function name isn't one of the above."""

    @classmethod
    def match(cls, extracted_text: str, labeled_texts: list[str]) -> bool:
        extracted = remove_redundant_whitespace(extracted_text)
        return any(extracted == remove_redundant_whitespace(t) for t in labeled_texts)


MATCH_FUNCS: dict[str, type[Match]] = {
    "DateMatch": DateMatch,
    "PriceMatch": PriceMatch,
    "AddressMatch": AddressMatch,
    "NumericalStringMatch": NumericalStringMatch,
    "GeneralStringMatch": GeneralStringMatch,
    "NameMatch": NameMatch,
    "DefaultMatch": DefaultMatch,
}
