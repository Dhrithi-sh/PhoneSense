from dhvani.rank.difficulty import clarity, predict, word_idf
from dhvani.rank.parser import parse_and_rank
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex


def _run(text):
    idx = SampleIndex.load()
    q = exact_query(text)
    return q, parse_and_rank(q, idx, k=10), idx


def test_common_words_only_is_low_confidence():
    q, r, idx = _run("में के")
    p = predict(q, r, idx)
    assert p["low_confidence"] and "every query word is very common" in p["reasons"]


def test_specific_query_is_fine():
    q, r, idx = _run("कोहली शतक")
    p = predict(q, r, idx)
    assert not p["low_confidence"] and p["specificity"] >= 1.0


def test_word_idf_uses_the_most_common_strong_spelling():
    idx = SampleIndex.load()
    token = {"surface": "x", "expansions": [("zzz_not_there", 1.0, "exact"), ("में", 0.6, "phonetic"), ("कोहली", 0.1, "phonetic")]}
    import math
    assert abs(word_idf(token, idx) - math.log10(idx.N / idx.df("में"))) < 1e-12


def test_nothing_matched_is_flagged():
    q, r, idx = _run("zzzz")
    assert predict(q, r, idx)["low_confidence"]


def test_clarity_is_positive_for_a_focused_query():
    q, r, idx = _run("कोहली शतक")
    assert clarity(r, idx) > 0
