from dhvani.query.editdist import distance, learn_costs, levenshtein


def test_levenshtein_known_values():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("mausam", "mausam") == 0
    assert levenshtein("", "abc") == 3


# A tiny training set where the o→a substitution and a dropped 'a' are common.
TRAIN = (
    [("mausam", "mosam")] * 5
    + [("barish", "baarish")] * 5
    + [("kamal", "kamal")] * 5
    + [("dilli", "dilli")] * 5
)


def test_learned_cost_makes_matches_cheap():
    table, default = learn_costs(TRAIN, iterations=3)
    # Identical characters cost ~0; the default (unseen edit) is clearly dearer.
    assert distance("kamal", "kamal", table, default) < 1e-9
    assert default > 0.5


def test_learned_distance_prefers_seen_variation():
    table, default = learn_costs(TRAIN, iterations=3)
    # A spelling variation the model has seen should cost less than an
    # equally-long but unseen garble.
    seen = distance("mausam", "mosam", table, default)
    garble = distance("mausam", "xyzqm", table, default)
    assert seen < garble


def test_save_and_load_roundtrip(tmp_path):
    from dhvani.query.editdist import load_costs, save_costs

    table, default = learn_costs(TRAIN, iterations=2)
    path = tmp_path / "costs.json"
    save_costs(path, table, default)
    table2, default2 = load_costs(path)
    assert default2 == default
    assert distance("mausam", "mosam", table2, default2) == distance("mausam", "mosam", table, default)
