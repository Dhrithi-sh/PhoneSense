from dhvani.eval.sanity import cross_form_agreement, kendall_tau, overlap, system_agreement


def test_overlap():
    assert overlap(["a", "b", "c", "d"], ["b", "a", "x", "y"], k=4) == 0.5
    assert overlap([], ["a"]) is None


def test_kendall_tau_same_and_reversed():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert kendall_tau(["a"], ["a"]) is None


def test_cross_form_agreement_uses_the_hindi_form_as_reference():
    queries = [("R01_hi", "R01", "hindi", ""), ("R01_en", "R01", "english", ""), ("R01_hinglish", "R01", "hinglish", "")]
    run = {"R01_hi": ["a", "b"], "R01_en": ["a", "x"], "R01_hinglish": ["a", "b"]}
    assert cross_form_agreement(run, queries, k=2) == {"english": 0.5, "hinglish": 1.0}


def test_system_agreement_pairs():
    runs = {"x": {"q": ["a", "b"]}, "y": {"q": ["a", "b"]}, "z": {"q": ["c", "d"]}}
    rows = {(a, b): (ov, tau) for a, b, ov, tau in system_agreement(runs, k=2)}
    assert rows[("x", "y")] == (1.0, 1.0)
    assert rows[("x", "z")][0] == 0.0
