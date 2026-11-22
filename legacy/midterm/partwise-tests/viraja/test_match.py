from dhvani.query import match as M
from dhvani.query.editdist import learn_costs
from dhvani.query.kgram import KGramIndex

VOCAB = ["मौसम", "बारिश", "तापमान", "क्रिकेट", "चुनाव", "बाजार", "स्कूल"]

# Enough signal for the learned matcher to behave sensibly in the test.
TRAIN = [("mausam", "mosam")] * 5 + [("baarish", "barish")] * 5 + [("mausam", "mausam")] * 5


def _costs():
    return learn_costs(TRAIN, iterations=3)


def test_all_four_matchers_find_the_right_word():
    idx = KGramIndex(VOCAB, k=2)
    costs = _costs()
    for matcher in M.MATCHERS:
        top = M.rank("mosam", idx, matcher=matcher, costs=costs, k=3)
        assert top, f"{matcher} returned nothing"
        assert top[0][0] == "मौसम", f"{matcher} ranked {top[0][0]} first"


def test_weighted_variants_shape_and_weights():
    idx = KGramIndex(VOCAB, k=2)
    variants = M.weighted_variants("mosam", idx, _costs(), k=5)
    assert variants, "expected at least one variant"
    # Format-4 expansion entries: (term, weight, "phonetic").
    for term, weight, source in variants:
        assert isinstance(term, str)
        assert 0.0 <= weight <= 1.0
        assert source == "phonetic"
    assert abs(sum(w for _t, w, _s in variants) - 1.0) < 1e-9
    # The best match carries the largest weight.
    assert variants[0][0] == "मौसम"
    assert variants[0][1] == max(w for _t, w, _s in variants)


def test_iyer_reaches_ayyar_even_when_kgram_pool_misses_it():
    # Distractors out-compete अय्यर on k-gram overlap; with a small pool the
    # k-gram step alone would drop it. Phonetic-code candidates rescue it so the
    # matcher ranks अय्यर top instead of a wrong word like श्रेयस.
    vocab = ["अय्यर", "श्रेयस", "मेयर", "लेयर", "बायर", "सायर", "वीर", "तीर"]
    idx = KGramIndex(vocab, k=2)
    costs = learn_costs([("iyer", "ayyar")] * 3 + [("x", "x")] * 3, iterations=2)
    for matcher in ("dhvani", "learned", "levenshtein"):
        top = M.rank("iyer", idx, matcher=matcher, costs=costs, k=1, pool=3)
        assert top[0][0] == "अय्यर", f"{matcher} ranked {top[0][0]} first"


def test_self_match_in_vocab_does_not_starve_phonetic_variants():
    # "iyer" is also an English word in the index. The exact self-match must not
    # eat all the weight: the Devanagari homophones should still carry real
    # weight, and the name अय्यर should rank above spelling near-misses.
    vocab = ["iyer", "अय्यर", "एयर", "ईयर", "बायर", "मेयर"]
    idx = KGramIndex(vocab, k=2)
    costs = learn_costs([("iyer", "ayyar")] * 3 + [("x", "x")] * 3, iterations=2)
    variants = M.weighted_variants("iyer", idx, costs, k=5)
    assert "iyer" not in [t for t, _w, _s in variants]      # self excluded
    top_term, top_weight, _ = variants[0]
    assert top_term == "अय्यर"                               # name ranks first
    assert top_weight > 0.2                                  # and carries real weight
    byr = dict((t, w) for t, w, _s in variants)
    assert byr.get("बायर", 0) < 0.05                         # near-miss stays tiny


def test_df_prefers_the_common_homophone():
    # "modi" sounds like मोदी, मॉड and मोड; corpus frequency (df) should put the
    # common one (the name) first instead of a rare look-alike.
    vocab = ["मोदी", "मॉड", "मोड"]
    idx = KGramIndex(vocab, k=2, df={"मोदी": 500, "मॉड": 2, "मोड": 3})
    costs = learn_costs([("modi", "modi")] * 3 + [("x", "x")] * 3, iterations=2)
    variants = M.weighted_variants("modi", idx, costs, k=3)
    assert variants and variants[0][0] == "मोदी"


def test_phonetic_variants_are_devanagari_only():
    # A Roman look-alike in the vocab must never be offered as an expansion.
    idx = KGramIndex(["मौसम", "mosam", "mausam"], k=2)
    costs = learn_costs([("mosam", "mausam")] * 3 + [("x", "x")] * 3, iterations=2)
    variants = M.weighted_variants("mosam", idx, costs, k=5)
    assert all(any("ऀ" <= ch <= "ॿ" for ch in t) for t, _w, _s in variants)
    assert "मौसम" in [t for t, _w, _s in variants]


def test_english_word_rejects_loose_lookalike():
    # "weather" reads as English and भाथर is only a coincidental same-code word,
    # too far in spelling to be a transliteration, so nothing should expand.
    idx = KGramIndex(["भाथर", "मौसम"], k=2, df={"भाथर": 5, "मौसम": 100})
    costs = learn_costs([("x", "x")] * 3, iterations=2)
    assert M.weighted_variants("weather", idx, costs, k=3) == []


def test_unknown_matcher_raises():
    idx = KGramIndex(VOCAB, k=2)
    try:
        M.rank("mosam", idx, matcher="nope", costs=_costs())
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown matcher")
