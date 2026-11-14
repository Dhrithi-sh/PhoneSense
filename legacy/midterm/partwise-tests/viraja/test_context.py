from dhvani.query.context import best_path, correct


def _cooccur_map(pairs):
    """Return a symmetric cooccur(a, b) backed by a dict of counts."""
    counts = {}
    for (a, b), c in pairs.items():
        counts[(a, b)] = c
        counts[(b, a)] = c
    return lambda a, b: counts.get((a, b), 0)


def test_context_flips_the_per_word_best():
    # Position 0: A is individually stronger than B.
    # Position 1: X and Y tie on weight.
    # But B and Y co-occur a lot, so the joint-best path is (B, Y).
    lattice = [[("A", 0.6), ("B", 0.4)], [("X", 0.5), ("Y", 0.5)]]
    cooccur = _cooccur_map({("B", "Y"): 50, ("A", "X"): 1})
    assert best_path(lattice, cooccur, lam=1.0) == ["B", "Y"]


def test_no_cooccurrence_falls_back_to_weights():
    lattice = [[("A", 0.9), ("B", 0.1)], [("X", 0.8), ("Y", 0.2)]]
    cooccur = lambda a, b: 0  # noqa: E731 - nothing co-occurs
    assert best_path(lattice, cooccur) == ["A", "X"]


def test_single_word_query_returns_its_best_candidate():
    assert best_path([[("A", 0.3), ("B", 0.7)]], lambda a, b: 0) == ["B"]


def test_correct_moves_chosen_variant_to_front():
    query = {"tokens": [
        {"expansions": [("A", 0.6, "exact"), ("B", 0.4, "phonetic")]},
        {"expansions": [("X", 0.5, "exact"), ("Y", 0.5, "phonetic")]},
    ]}
    cooccur = _cooccur_map({("B", "Y"): 50})
    correct(query, cooccur, lam=1.0)
    assert query["tokens"][0]["expansions"][0][0] == "B"
    assert query["tokens"][1]["expansions"][0][0] == "Y"
