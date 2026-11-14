from datetime import date

from dhvani.rank.kal import apply_kal, is_future_tense, kal_boost, kal_intent
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.vsm import search


def test_no_kal_word_means_no_intent():
    assert kal_intent(exact_query("दिल्ली बारिश")) is None


def test_english_words_say_it_outright():
    assert kal_intent(exact_query("weather tomorrow")) == "tomorrow"
    assert kal_intent(exact_query("match yesterday")) == "yesterday"


def test_weather_kal_means_tomorrow():
    assert kal_intent(exact_query("कल का मौसम")) == "tomorrow"
    assert kal_intent(exact_query("kal ka mausam")) == "tomorrow"


def test_past_cues_mean_yesterday():
    assert kal_intent(exact_query("कल का मैच हुआ")) == "yesterday"
    assert kal_intent(exact_query("kal ka result")) == "yesterday"


def test_no_cue_defaults_to_yesterday():
    assert kal_intent(exact_query("कल संसद")) == "yesterday"


def test_future_tense_detection():
    assert is_future_tense("मुंबई में कल बारिश होगी और मौसम साफ रहेगा")
    assert not is_future_tense("कल मैच में भारत ने जीत हासिल की थी और टीम खुश हुई")


def test_yesterday_boosts_the_day_before():
    today = date(2026, 10, 5)
    assert kal_boost("yesterday", {"date": "2026-10-04T10:00:00+05:30"}, "", today) == 1.0
    assert kal_boost("yesterday", {"date": "2026-10-02T10:00:00+05:30"}, "", today) == 0.0


def test_tomorrow_boosts_fresh_forecasts():
    today = date(2026, 10, 5)
    forecast = "कल तेज बारिश होगी, अलर्ट जारी"
    report = "कल बारिश हुई थी"
    fresh = {"date": "2026-10-05T08:00:00+05:30"}
    assert kal_boost("tomorrow", fresh, forecast, today) == 1.0
    assert kal_boost("tomorrow", fresh, report, today) < 1.0


def test_queries_without_kal_are_untouched():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    results = search(q, idx, k=5)
    assert apply_kal(results, q, idx) == results


def test_kal_explain_and_reorder_on_sample():
    idx = SampleIndex.load()
    q = exact_query("कल बारिश")
    reranked = apply_kal(search(q, idx, k=5), q, idx)
    assert all("kal" in e for _, _, e in reranked)
    assert reranked[0][2]["kal"]["intent"] == "tomorrow"
    scores = [s for _, s, _ in reranked]
    assert scores == sorted(scores, reverse=True)
    # The Delhi rain alert (5 Oct, a forecast) should come out on top.
    assert reranked[0][0] == "jagran_1001"


def test_date_alone_cannot_lift_an_unrelated_article():
    # aajtak_5003 (petrol prices, published the day before) only shares "का"
    # with the query. A flat bonus put it at #1; with the multiplicative boost
    # its tiny relevance score keeps it from taking the top spot.
    idx = SampleIndex.load()
    q = exact_query("कल का मैच हुआ")
    reranked = apply_kal(search(q, idx, k=5), q, idx)
    assert reranked[0][0] != "aajtak_5003"
    assert next(e for d, _, e in reranked if d == "aajtak_5003")["kal"]["boost"] == 1.0


def test_kal_keeps_the_parser_stage_order():
    from dhvani.rank.parser import STAGES, parse_and_rank
    idx = SampleIndex.load()
    q = exact_query("कल बारिश")
    reranked = apply_kal(parse_and_rank(q, idx, k=5), q, idx)
    ranks = [STAGES.index(e["stage"]) for _, _, e in reranked]
    assert ranks == sorted(ranks)


def test_kal_keeps_term_scores_separate_for_plain_rankers():
    idx = SampleIndex.load()
    q = exact_query("कल बारिश")
    _, _, explain = apply_kal(search(q, idx, k=3), q, idx)[0]
    assert "kal" in explain and "kal" not in explain["terms"]
    assert all(isinstance(v, float) for v in explain["terms"].values())
