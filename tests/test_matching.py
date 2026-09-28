from bench_extract.datasets import matching


def test_date_match_exact():
    assert matching.DateMatch.match("01/02/2024", ["01/02/2024"])


def test_date_match_format_variant():
    # Same date, different textual format -- decode_date should normalize both.
    assert matching.DateMatch.match("January 02, 2024", ["01/02/24"])


def test_date_match_day_month_swap_same_year():
    assert matching.DateMatch.match("02/01/2024", ["01/02/2024"])


def test_date_match_wrong_date():
    assert not matching.DateMatch.match("03/04/2024", ["01/02/2024"])


def test_price_match_within_tolerance():
    assert matching.PriceMatch.match("$1,250.00", ["1250.00"])


def test_price_match_wrong_value():
    assert not matching.PriceMatch.match("$1,250.00", ["999.00"])


def test_general_string_match_ignores_punctuation():
    assert matching.GeneralStringMatch.match("Acme Corp.", ["Acme Corp"])


def test_general_string_match_wrong_text():
    assert not matching.GeneralStringMatch.match("Acme Corp", ["Widget Inc"])


def test_numerical_string_match_strips_non_digits():
    assert matching.NumericalStringMatch.match("Reg# 4821", ["4821"])


def test_numerical_string_match_wrong_number():
    assert not matching.NumericalStringMatch.match("4821", ["4822"])


def test_address_match_small_edit_distance():
    # "123 Main St." vs "123 Main St" -- edit distance 1, within the default threshold of 3.
    assert matching.AddressMatch.match("123 Main St", ["123 Main St."])


def test_address_match_too_different():
    assert not matching.AddressMatch.match("123 Main St", ["456 Oak Ave"])


def test_match_checks_any_of_multiple_ground_truth_occurrences():
    # A repeated field: any one occurrence matching is enough.
    assert matching.GeneralStringMatch.match("Acme Corp", ["Widget Inc", "Acme Corp"])


def test_match_empty_ground_truth_is_no_match():
    assert not matching.GeneralStringMatch.match("Acme Corp", [])


def test_score_prediction_correct_and_incorrect():
    assert matching.score_prediction(["Acme Corp"], ["Acme Corp"], "GeneralStringMatch")
    assert not matching.score_prediction(["Widget Inc"], ["Acme Corp"], "GeneralStringMatch")


def test_score_prediction_empty_prediction_is_incorrect():
    assert not matching.score_prediction([], ["Acme Corp"], "GeneralStringMatch")
