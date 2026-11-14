from dhvani.rank.filters import make_filter
from dhvani.rank.parser import and_docs, parse_and_rank, phrase_docs, stage_matches
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex


def tiny_index():
    articles = [
        {"doc_id": "a", "headline": "दिल्ली में बारिश", "body": "तेज बारिश का अलर्ट", "date": "2026-10-05T10:00:00+05:30"},
        {"doc_id": "b", "headline": "बारिश में दिल्ली", "body": "दिल्ली की खबर", "date": "2026-10-05T10:00:00+05:30"},
        {"doc_id": "c", "headline": "दिल्ली मौसम", "body": "आज धूप", "date": "2026-10-05T10:00:00+05:30"},
        {"doc_id": "d", "headline": "पटना खबर", "body": "कुछ नहीं", "date": "2026-10-05T10:00:00+05:30"},
    ]
    return SampleIndex(articles)


def alts(*words):
    return [{w} for w in words]


def test_phrase_needs_order_and_adjacency():
    idx = tiny_index()
    # "दिल्ली में" is a phrase in a's headline; b has "में दिल्ली", the wrong order.
    assert phrase_docs(idx, alts("दिल्ली", "में")) == {"a"}
    assert phrase_docs(idx, alts("में", "दिल्ली")) == {"b"}


def test_phrase_must_stay_in_one_zone():
    idx = tiny_index()
    # a's headline ends with बारिश and its body starts with तेज, so
    # "बारिश तेज" must not count as a phrase across zones.
    assert phrase_docs(idx, alts("बारिश", "तेज")) == set()


def test_and_needs_every_word():
    idx = tiny_index()
    assert and_docs(idx, alts("दिल्ली", "बारिश")) == {"a", "b"}
    assert and_docs(idx, alts("दिल्ली", "पटना")) == set()


def test_stages_go_from_strict_to_loose():
    idx = tiny_index()
    stages = dict(stage_matches(idx, exact_query("दिल्ली में बारिश")))
    assert stages["phrase"] == {"a"}
    # b has the words, but "बारिश में दिल्ली" contains neither "दिल्ली में"
    # nor "में बारिश" in order, so it isn't a sub-phrase match.
    assert stages["sub-phrase"] == {"a"}
    assert "b" in stages["all words"]
    assert stages["any word"] >= stages["all words"] >= stages["phrase"]


def test_phrase_match_ranks_first():
    idx = tiny_index()
    results = parse_and_rank(exact_query("दिल्ली में बारिश"), idx, k=3, ranker="lnc")
    assert results[0][0] == "a"
    assert results[0][2]["stage"] == "phrase"


def test_cascade_stops_once_it_has_k():
    idx = tiny_index()
    # Two articles contain both words, so with k = 2 we never need "any word".
    results = parse_and_rank(exact_query("दिल्ली बारिश"), idx, k=2, ranker="lnc")
    assert {d for d, _, _ in results} == {"a", "b"}
    assert "any word" not in results[0][2]["stages_run"]


def test_cascade_falls_back_to_free_text():
    idx = tiny_index()
    # Nothing has both दिल्ली and पटना, so the parser falls back to any word.
    results = parse_and_rank(exact_query("दिल्ली पटना"), idx, k=4, ranker="bm25")
    stages = {d: e["stage"] for d, _, e in results}
    assert stages["d"] == "any word"
    assert "any word" in results[0][2]["stages_run"]


def test_term_scores_stay_separate_from_stage_info():
    idx = tiny_index()
    for ranker in ("lnc", "bm25", "net"):
        _, _, explain = parse_and_rank(exact_query("दिल्ली बारिश"), idx, k=1, ranker=ranker)[0]
        assert set(explain["terms"]) == {"दिल्ली", "बारिश"}
        assert "stage" in explain and "stage" not in explain["terms"]


def test_single_word_skips_phrase_stages():
    idx = tiny_index()
    names = [s for s, _ in stage_matches(idx, exact_query("दिल्ली"))]
    assert "phrase" not in names and "sub-phrase" not in names


def test_works_with_filters_and_sample_index():
    idx = SampleIndex.load()
    f = make_filter(idx, states=["delhi"])
    results = parse_and_rank(exact_query("बारिश का अलर्ट"), idx, k=3, doc_filter=f)
    assert results and all(idx.meta[d]["state"] == "delhi" for d, _, _ in results)
    assert results[0][0] == "jagran_1001"          # has the exact phrase in its headline
    assert results[0][2]["stage"] == "phrase"


def test_feedback_tokens_do_not_tighten_the_and_stage():
    idx = tiny_index()
    q = exact_query("दिल्ली बारिश")
    q["tokens"].append({"surface": "अलर्ट", "script": "devanagari",
                        "lang": {"hi": 1.0, "hinglish": 0.0, "en": 0.0},
                        "expansions": [("अलर्ट", 0.3, "prf")]})
    stages = dict(stage_matches(idx, q))
    # Both a and b have दिल्ली and बारिश; only a has अलर्ट. The feedback word
    # mustn't knock b out of "all words".
    assert stages["all words"] == {"a", "b"}
    assert "a" in stages["any word"]
