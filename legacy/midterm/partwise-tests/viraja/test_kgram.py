from dhvani.query.kgram import KGramIndex, kgrams

# A small Devanagari vocabulary (weather words + unrelated distractors).
VOCAB = ["मौसम", "बारिश", "तापमान", "क्रिकेट", "चुनाव", "बाजार", "स्कूल"]


def test_kgrams_have_boundary_markers():
    g = kgrams("kal", k=2)
    assert "$k" in g and "l$" in g


def test_devanagari_vocab_is_reachable_from_roman_query():
    idx = KGramIndex(VOCAB, k=2)
    cands = [term for term, _score in idx.candidates("mosam", limit=5)]
    # मौसम romanises to "mausam", which shares most k-grams with "mosam".
    assert "मौसम" in cands
    assert cands[0] == "मौसम"


def test_candidates_are_sorted_by_jaccard_desc():
    idx = KGramIndex(VOCAB, k=2)
    scores = [j for _term, j in idx.candidates("barish", limit=10)]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 < scores[0] <= 1.0


def test_unrelated_query_returns_few_or_no_candidates():
    idx = KGramIndex(VOCAB, k=2)
    # "zzzz" shares no k-grams with any weather/news term.
    assert idx.candidates("zzzz", limit=5) == []


def test_phonetic_candidates_catch_same_sounding_distant_spellings():
    # iyer / अय्यर share Dhvani-code 26 but almost no k-grams.
    idx = KGramIndex(["अय्यर", "श्रेयस", "मौसम"], k=2)
    assert "अय्यर" in idx.phonetic_candidates("iyer")
    assert "श्रेयस" not in idx.phonetic_candidates("iyer")  # different code


def test_phonetic_candidates_also_use_soundex():
    # delhi / दिल्ली differ in Dhvani-code over the silent "h" but share Soundex,
    # so Soundex keying is the second net that surfaces दिल्ली.
    idx = KGramIndex(["दिल्ली", "मौसम"], k=2)
    assert "दिल्ली" in idx.phonetic_candidates("delhi")
