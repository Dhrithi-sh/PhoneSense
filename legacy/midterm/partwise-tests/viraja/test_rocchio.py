from dhvani.query.rocchio import expand_query, rocchio, top_terms


def test_rocchio_pulls_in_relevant_terms():
    # Query is about {मौसम}; the "relevant" docs are full of बारिश.
    q = {"मौसम": 1.0}
    rel = [{"मौसम": 1.0, "बारिश": 1.0}, {"मौसम": 0.5, "बारिश": 1.0, "अलर्ट": 1.0}]
    out = rocchio(q, rel)
    assert out["मौसम"] > out["बारिश"] > 0          # query term stays strongest
    assert "अलर्ट" in out                           # a feedback term was pulled in


def test_rocchio_clips_negative_weights():
    q = {"a": 1.0}
    out = rocchio(q, relevant=[], nonrelevant=[{"b": 5.0}], gamma=1.0)
    assert "b" not in out                            # negative weight truncated to 0
    assert out["a"] == 1.0


def test_top_terms_excludes_query_terms():
    weights = {"a": 1.0, "b": 0.9, "c": 0.8}
    assert top_terms(weights, 2, exclude={"a"}) == [("b", 0.9), ("c", 0.8)]


def test_expand_query_appends_prf_tokens():
    query = {"raw": "मौसम", "tokens": [
        {"surface": "मौसम", "script": "devanagari",
         "lang": {"hi": 1.0, "hinglish": 0.0, "en": 0.0},
         "expansions": [("मौसम", 1.0, "exact")]},
    ]}
    rel = [{"मौसम": 1.0, "बारिश": 1.0, "अलर्ट": 1.0}]
    expand_query(query, rel, k=5)
    sources = {s for tok in query["tokens"] for _t, _w, s in tok["expansions"]}
    assert "prf" in sources
    # The original token is untouched; new PRF terms are separate tokens.
    assert query["tokens"][0]["expansions"] == [("मौसम", 1.0, "exact")]
    prf_terms = {tok["surface"] for tok in query["tokens"][1:]}
    assert "बारिश" in prf_terms and "मौसम" not in prf_terms
