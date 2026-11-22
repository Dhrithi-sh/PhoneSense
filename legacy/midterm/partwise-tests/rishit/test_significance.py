from dhvani.eval.significance import compare, randomization_test, t_test


def test_identical_systems_are_not_significant():
    a = {f"q{i}": i / 10 for i in range(10)}
    assert randomization_test(a, a, permutations=500) == 1.0
    assert t_test(a, a)[1] == 1.0


def test_a_clear_consistent_gain_is_significant():
    a = {f"q{i}": 0.5 + 0.01 * i for i in range(20)}
    b = {q: v - 0.2 for q, v in a.items()}
    assert randomization_test(a, b, permutations=2000) < 0.01
    assert t_test(a, b)[1] < 0.01 or t_test(a, b)[0] == float("inf")


def test_noisy_small_difference_is_not_significant():
    a = {"q1": 0.5, "q2": 0.2, "q3": 0.9, "q4": 0.4}
    b = {"q1": 0.4, "q2": 0.3, "q3": 0.8, "q4": 0.5}
    assert randomization_test(a, b, permutations=2000) > 0.3
    assert t_test(a, b)[1] > 0.3


def test_t_test_p_value_matches_a_known_case():
    # differences 1, 2, 3, 4, 5: t = 4.243, df = 4, two-sided p = 0.0132
    a = {f"q{i}": float(i) for i in range(1, 6)}
    b = {q: 0.0 for q in a}
    t, p = t_test(a, b)
    assert abs(t - 4.2426) < 1e-3 and abs(p - 0.0132) < 1e-3


def test_compare_uses_only_shared_queries():
    c = compare({"q1": 1.0, "q2": 0.0}, {"q1": 0.5}, permutations=100)
    assert c["queries"] == 1 and c["difference"] == 0.5
