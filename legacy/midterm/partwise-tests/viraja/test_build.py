from dhvani.query.build import build_query


def test_query_object_shape_matches_format():
    q = build_query("kal ka mosam")
    assert q["raw"] == "kal ka mosam"
    assert [t["surface"] for t in q["tokens"]] == ["kal", "ka", "mosam"]
    for tok in q["tokens"]:
        assert set(tok) == {"surface", "script", "lang", "expansions"}
        assert set(tok["lang"]) == {"hi", "hinglish", "en"}


def test_exact_expansion_only_for_now():
    q = build_query("मौसम")
    (term, weight, source), = q["tokens"][0]["expansions"]
    assert (term, weight, source) == ("मौसम", 1.0, "exact")


def test_script_and_lang_are_filled_in():
    q = build_query("कल weather")
    hindi_tok, english_tok = q["tokens"]
    assert hindi_tok["script"] == "devanagari"
    assert hindi_tok["lang"]["hi"] == 1.0
    assert english_tok["script"] == "roman"
    assert english_tok["lang"]["en"] == 0.9


def test_punctuation_is_dropped_by_tokenizer():
    q = build_query("kal, ka mosam?")
    assert [t["surface"] for t in q["tokens"]] == ["kal", "ka", "mosam"]
