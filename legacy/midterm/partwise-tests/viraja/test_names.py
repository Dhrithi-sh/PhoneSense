from dhvani.query import match as M
from dhvani.query.editdist import learn_costs
from dhvani.query.kgram import KGramIndex
from dhvani.query.names import evaluate, load_names

# Small cost table so the 'learned' matcher has something to use in the test.
_TRAIN = [("laxmi", "lakshmi")] * 3 + [("sidharth", "siddharth")] * 3 + [("x", "x")] * 3
_COSTS = learn_costs(_TRAIN, iterations=2)


def test_fifty_names_load_with_variants():
    names = load_names()
    assert len(names) == 50
    for canonical, variants in names:
        assert variants
        assert canonical not in variants  # a query must differ from the answer


def test_a_clear_variant_retrieves_its_canonical():
    names = load_names()
    index = KGramIndex([c for c, _ in names], k=2)
    # "laxmi" should surface "lakshmi" at the top for the phonetic matchers.
    for matcher in ("dhvani", "learned", "levenshtein"):
        top = M.rank("laxmi", index, matcher=matcher, costs=_COSTS, k=3)
        assert top[0][0] == "lakshmi", f"{matcher} ranked {top[0][0]} first"


def test_evaluate_returns_all_matchers_with_sane_scores():
    names = load_names()
    results = evaluate(names, _COSTS)
    assert set(results) == set(M.MATCHERS)
    for _matcher, (acc, mrr) in results.items():
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= mrr <= 1.0
    # Phonetic matching should get a clear majority of names right.
    assert max(acc for acc, _mrr in results.values()) > 0.6
