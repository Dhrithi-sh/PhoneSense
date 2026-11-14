from dhvani.query.build import build_query
from dhvani.query.editdist import learn_costs
from dhvani.query.expand import expand_query
from dhvani.query.kgram import KGramIndex

VOCAB = ["मौसम", "बारिश", "तापमान", "क्रिकेट", "चुनाव", "बाजार"]
TRAIN = [("mausam", "mosam")] * 5 + [("baarish", "barish")] * 5 + [("mausam", "mausam")] * 5


def _index_and_costs():
    return KGramIndex(VOCAB, k=2), learn_costs(TRAIN, iterations=3)


def test_build_query_stays_exact_without_an_index():
    q = build_query("mosam")
    assert q["tokens"][0]["expansions"] == [("mosam", 1.0, "exact")]


def test_build_query_adds_phonetic_variants_with_index():
    index, costs = _index_and_costs()
    q = build_query("mosam", index=index, costs=costs)
    exps = q["tokens"][0]["expansions"]
    # Exact stays first; a phonetic मौसम is appended.
    assert exps[0] == ("mosam", 1.0, "exact")
    phonetic = [(t, s) for t, _w, s in exps if s == "phonetic"]
    assert ("मौसम", "phonetic") in phonetic


def test_expand_query_does_not_duplicate_terms():
    index, costs = _index_and_costs()
    q = build_query("barish", index=index, costs=costs)
    terms = [t for t, _w, _s in q["tokens"][0]["expansions"]]
    assert len(terms) == len(set(terms))


def test_alias_forces_the_correct_devanagari_for_hard_names():
    # delhi -> दिल्ली (silent-h) and iyer -> अय्यर (homophone of एयर) are the
    # curated hard cases: the alias must be the top phonetic expansion.
    idx = KGramIndex(["दिल्ली", "देल्ही", "अय्यर", "एयर", "मौसम"], k=2)
    _c = learn_costs([("x", "x")] * 3, iterations=2)
    for query, want in (("delhi", "दिल्ली"), ("iyer", "अय्यर")):
        exps = build_query(query, index=idx, costs=_c)["tokens"][0]["expansions"]
        phonetic = [t for t, _w, s in exps if s == "phonetic"]
        assert phonetic[0] == want, f"{query} -> {phonetic}"


def test_alias_ignored_when_not_in_vocabulary():
    # Don't invent a term the corpus doesn't have.
    idx = KGramIndex(["मौसम"], k=2)
    exps = build_query("delhi", index=idx, costs=learn_costs([("x", "x")] * 3))["tokens"][0]["expansions"]
    assert "दिल्ली" not in [t for t, _w, _s in exps]


def test_english_words_are_not_phonetically_expanded():
    # "farmers" is English -> no Hindi homophone -> must not get phonetic junk.
    index, costs = _index_and_costs()
    q = build_query("farmers", index=index, costs=costs)
    assert q["tokens"][0]["expansions"] == [("farmers", 1.0, "exact")]


def test_expand_is_idempotent():
    index, costs = _index_and_costs()
    q = build_query("mosam", index=index, costs=costs)
    before = list(q["tokens"][0]["expansions"])
    expand_query(q, index, costs)  # running again adds nothing new
    assert q["tokens"][0]["expansions"] == before
