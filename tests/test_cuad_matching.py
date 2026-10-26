from bench_extract.datasets import cuad_matching as m


def test_get_jaccard_ignores_case_punctuation_and_slash():
    assert m.get_jaccard("Governing Law, NY.", "governing law ny") == 1.0
    assert m.get_jaccard("a/b", "a b") == 1.0


def test_empty_ground_truth_correct_only_with_no_predictions():
    assert m.score_prediction([], [], "Insurance")
    assert m.score_prediction([""], [], "Insurance")  # empty string is not a prediction
    assert not m.score_prediction(["some clause"], [], "Insurance")


def test_missing_prediction_is_false_negative():
    assert not m.score_prediction([], ["clause text here"], "Insurance")


def test_extra_prediction_is_false_positive():
    gt = ["clause text here"]
    assert m.score_prediction(["clause text here"], gt, "Insurance")
    assert not m.score_prediction(["clause text here", "totally unrelated words"], gt, "Insurance")


def test_all_ground_truth_spans_must_be_matched():
    gt = ["first clause alpha", "second clause beta"]
    assert not m.score_prediction(["first clause alpha"], gt, "Insurance")
    assert m.score_prediction(["second clause beta", "first clause alpha"], gt, "Insurance")


def test_jaccard_threshold_is_half():
    # 2 shared words / 4 in union = 0.5 -> match; 1/3 words... -> no match
    assert m.score_prediction(["a b c"], ["a b d"], "Insurance")
    assert not m.score_prediction(["a x y"], ["a b d"], "Insurance")


def test_parties_accepts_ground_truth_substring_of_prediction():
    gt = ["Acme"]
    pred = ["This agreement is between Acme Corporation of Delaware and others in a long sentence"]
    assert m.score_prediction(pred, gt, "Parties")
    assert not m.score_prediction(pred, gt, "Insurance")
