import math

import pytest

from bench_extract.scoring import unified as u

CFG = u.ScoringConfig.from_params({"measure": "unified", "match_threshold": 0.5, "quantity_decimals": 2})


# --- config ----------------------------------------------------------------


def test_scoring_config_absent_is_native():
    assert u.ScoringConfig.from_params(None).measure == u.NATIVE
    assert u.ScoringConfig.from_params({"measure": "native"}).measure == u.NATIVE


@pytest.mark.parametrize(
    "raw",
    [
        {"measure": "fuzzy"},
        {"measure": "native", "match_threshold": 0.5},
        {"measure": "unified", "match_threshold": 0.5},  # quantity_decimals missing: no code default
        {"measure": "unified", "match_threshold": 0.5, "quantity_decimals": 2, "extra": 1},
        {"measure": "unified", "match_threshold": 1.5, "quantity_decimals": 2},
        {"measure": "unified", "match_threshold": 0.5, "quantity_decimals": -1},
    ],
)
def test_scoring_config_rejects_bad_blocks(raw):
    with pytest.raises(ValueError):
        u.ScoringConfig.from_params(raw)


# --- normalization ---------------------------------------------------------


def test_normalize_text_has_no_empty_tokens():
    # CUAD's evaluator split(" ") turns double spaces / newlines into "" tokens.
    assert u.tokens("The  laws of\nNew York.") == ["the", "laws", "of", "new", "york"]
    assert u.normalize_text("  Ｍichael Bloomberg 2020, Inc\n") == "michael bloomberg 2020 inc"
    assert u.is_empty(" -- \n") and not u.is_empty("x")


# --- types -----------------------------------------------------------------


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        ("12/30/19 ", "2019-12-30", 1.0),
        ("December 30, 2019", "12/30/19", 1.0),
        ("September 7, 1999", "7th day of September, 1999.", 1.0),
        ("12/31/19", "12/30/19", 0.0),
        ("03/02/20", "02/03/20", 0.0),  # no MM/DD swap
        ("07/01/2022", "07/01", 1.0),  # GT has no year: only month/day must agree
        ("March 2020", "03/27/20", 0.0),  # prediction missing a component GT has
        ("TBD", "tbd", 1.0),  # unparsable on both sides -> identifier fallback
    ],
)
def test_date_similarity(pred, gt, expected):
    assert u.date_similarity(pred, gt) == expected


def test_parse_date_never_uses_today():
    assert u.parse_date("07/01") == {"month": 7, "day": 1}
    assert u.parse_date("Fox 9 AM News") is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("$5,625.00\n", "5625.00"),
        ("5625", "5625.00"),
        ("($600.00)", "-600.00"),
        ("-$5", "-5.00"),
        ("Total - $600", "600.00"),  # free-standing dash is a separator
        ("1,5", None),  # malformed thousands grouping
        ("$600 x 2", None),  # more than one number
        ("n/a", None),
    ],
)
def test_parse_quantity(text, expected):
    got = u.parse_quantity(text, 2)
    assert (str(got) if got is not None else None) == expected


def test_quantity_similarity_is_exact_after_normalization():
    assert u.quantity_similarity("$5,625.00", "5625", 2) == 1.0
    assert u.quantity_similarity("$5,625.01", "5625", 2) == 0.0
    assert u.quantity_similarity("$5,625.004", "5625", 2) == 1.0  # quantized to 2 places


def test_identifier_similarity():
    assert u.identifier_similarity("950658\n", "950 658") == 1.0
    assert u.identifier_similarity("12", "13") == 0.0  # Levenshtein would give 0.5
    assert u.identifier_similarity("", "") == 0.0


def test_short_text_is_true_levenshtein_not_indel():
    assert math.isclose(u.short_text_similarity("kitten", "sitting"), 4 / 7)  # fuzz.ratio gives 0.615
    assert u.short_text_similarity("Acme Corp.", "ACME CORP") == 1.0


def test_long_text_token_f1():
    # pred tokens {the, laws, of, new, york}; gt {laws, of, the, state, of, new, york}
    p, r = 5 / 5, 5 / 7
    assert math.isclose(u.long_text_similarity("the laws of New York.", "laws of the State of New York"), 2 * p * r / (p + r))
    assert u.long_text_similarity("foo", "bar") == 0.0


def test_value_similarity_unknown_type():
    with pytest.raises(ValueError):
        u.value_similarity("vibes", "a", "a", CFG)


# --- list matching ---------------------------------------------------------


def test_empty_cases():
    assert u.score_entities([], [], u.SHORT_TEXT, CFG) is None  # both empty: not scored
    assert u.score_entities(["  "], [[""]], u.SHORT_TEXT, CFG) is None  # empty after normalization
    assert u.score_entities(["Acme"], [], u.SHORT_TEXT, CFG) == 0.0  # hallucinated
    assert u.score_entities([], [["Acme"]], u.SHORT_TEXT, CFG) == 0.0  # missed


def test_entity_with_many_surface_forms_counts_once():
    gt = [["Michael Bloomberg 2020, Inc\n", "Michael Bloomberg 2020, Inc\n", "MICHAEL BLOOMBERG 2020 INC"]]
    assert u.score_entities(["Michael Bloomberg 2020, Inc"], gt, u.SHORT_TEXT, CFG) == 1.0


def test_duplicate_entities_are_a_multiset():
    # two distinct GT entities with the same value need two predictions
    f1 = u.score_entities(["$600"], [["$600"], ["$600"]], u.QUANTITY, CFG)
    assert math.isclose(f1, 2 * 1.0 * 0.5 / 1.5)
    assert u.score_entities(["$600", "600.00"], [["$600"], ["$600"]], u.QUANTITY, CFG) == 1.0


def test_hungarian_is_one_to_one_and_optimal():
    # greedy (pred 0 -> best gt 0) would leave pred 1 unmatched; optimal pairs crosswise
    import numpy as np

    sim = np.array([[0.9, 0.8], [0.85, 0.0]])
    assert math.isclose(u.hungarian_soft_f1(sim, 0.5), (0.8 + 0.85) / 2)


def test_threshold_is_inclusive_and_soft():
    import numpy as np

    assert u.hungarian_soft_f1(np.array([[0.5]]), 0.5) == 0.5
    assert u.hungarian_soft_f1(np.array([[0.49]]), 0.5) == 0.0


def test_rectangular_precision_recall():
    # 3 predictions, 1 correct entity: P = 1/3, R = 1
    f1 = u.score_entities(["Acme", "Widget", "Gadget"], [["Acme"]], u.SHORT_TEXT, CFG)
    assert math.isclose(f1, 2 * (1 / 3) * 1 / (1 / 3 + 1))


# --- records ---------------------------------------------------------------

TYPES = {"channel": u.IDENTIFIER, "program_desc": u.SHORT_TEXT, "sub_amount": u.QUANTITY}


def test_coerce_records_tolerates_messy_model_json():
    raw = [{"channel": 9, "sub_amount": None, "extra": "x"}, "not a row", None, {"sub_amount": "$1"}]
    assert u.coerce_records(raw, ["channel", "sub_amount"]) == [
        {"channel": "9", "sub_amount": ""},
        {"channel": "", "sub_amount": "$1"},
    ]
    assert u.coerce_records({"channel": "3"}, ["channel"]) == [{"channel": "3"}]  # a lone row object
    assert u.coerce_records(None, ["channel"]) == [] and u.coerce_records("9", ["channel"]) == []


def test_record_similarity_skips_both_empty_subfields():
    gt = {"channel": "9", "sub_amount": "$600"}  # no program_desc on this row
    assert u.record_similarity({"channel": "9", "program_desc": "", "sub_amount": "600"}, gt, TYPES, CFG) == 1.0
    # a sub-field only one side has scores 0
    assert u.record_similarity({"channel": "9", "program_desc": "News", "sub_amount": "600"}, gt, TYPES, CFG) == 2 / 3


def test_score_records():
    gt = [{"channel": "9", "program_desc": "News at 6", "sub_amount": "$600"}, {"channel": "3", "program_desc": "Prime", "sub_amount": "$50"}]
    perfect = [{"channel": "3", "program_desc": "prime", "sub_amount": "50.00"}, {"channel": "9", "program_desc": "News at 6", "sub_amount": "600"}]
    assert u.score_records(perfect, gt, TYPES, CFG) == 1.0
    assert u.score_records([], [], TYPES, CFG) is None
    assert u.score_records([{"channel": "", "program_desc": "", "sub_amount": ""}], [], TYPES, CFG) is None
    assert u.score_records([], gt, TYPES, CFG) == 0.0


# --- factory ---------------------------------------------------------------


def test_unified_context_kwargs_record_fields_and_validation():
    kwargs = u.unified_context_kwargs(
        fields=["advertiser", "line_items"],
        field_types={"advertiser": u.SHORT_TEXT, "channel": u.IDENTIFIER, "sub_amount": u.QUANTITY},
        record_fields={"line_items": ["channel", "sub_amount"]},
        config=CFG,
        entity_ground_truth=lambda doc, f: [],
        record_ground_truth=lambda doc, g: [],
    )
    assert kwargs["scored_fields"] == ["advertiser", "line_items"]
    assert kwargs["extraction_field_of"] == {"advertiser": "advertiser", "line_items": "line_items"}
    assert kwargs["scored_field_types"] == {"advertiser": u.SHORT_TEXT, "line_items": u.RECORD}
    rows = [{"channel": "9", "sub_amount": "$1", "x": "y"}]
    assert kwargs["prediction_for"]({"line_items": rows}, "line_items") == [{"channel": "9", "sub_amount": "$1"}]
    assert kwargs["prediction_for"]({"advertiser": ["Acme", None, 7]}, "advertiser") == ["Acme", "7"]
    with pytest.raises(ValueError, match="cover exactly"):
        u.unified_context_kwargs(["a", "b"], {"a": u.SHORT_TEXT}, {}, CFG, lambda d, f: [])
    with pytest.raises(ValueError, match="cover exactly"):  # a record field itself takes no type
        u.unified_context_kwargs(["r"], {"r": u.SHORT_TEXT, "s": u.DATE}, {"r": ["s"]}, CFG, lambda d, f: [], lambda d, f: [])
    with pytest.raises(ValueError, match="not among the extraction fields"):
        u.unified_context_kwargs(["a"], {"a": u.SHORT_TEXT}, {"r": ["s"]}, CFG, lambda d, f: [], lambda d, f: [])
    with pytest.raises(ValueError, match="Unknown field types"):
        u.unified_context_kwargs(["a"], {"a": "vibes"}, {}, CFG, lambda d, f: [])
