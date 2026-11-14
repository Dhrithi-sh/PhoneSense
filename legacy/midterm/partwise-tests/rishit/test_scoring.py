from datetime import datetime

from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.scoring import rank, recency, smallest_window
from dhvani.rank.vsm import search


def test_smallest_window_adjacent_words():
    assert smallest_window({"a": [3], "b": [4]}) == 2


def test_smallest_window_picks_tightest_span():
    # a at 0 and 10, b at 7, c at 9: best span is 7..10 = 4
    assert smallest_window({"a": [0, 10], "b": [7], "c": [9]}) == 4


def test_smallest_window_needs_two_terms():
    assert smallest_window({"a": [1, 2]}) is None
    assert smallest_window({"a": [1], "b": []}) is None


def test_recency_today_is_one_and_older_is_lower():
    now = datetime.fromisoformat("2026-10-05T12:00:00+05:30")
    assert abs(recency("2026-10-05T12:00:00+05:30", now) - 1.0) < 1e-9
    assert recency("2026-10-03T12:00:00+05:30", now) < recency("2026-10-04T12:00:00+05:30", now)
    assert recency(None, now) == 0.0


def test_explain_has_every_part():
    idx = SampleIndex.load()
    doc_id, net, explain = rank(exact_query("दिल्ली बारिश"), idx, k=1)[0]
    for key in ("cosine", "terms", "zone", "proximity", "recency", "net"):
        assert key in explain
    assert abs(explain["net"] - net) < 1e-12


def test_zero_weights_give_plain_cosine_order():
    idx = SampleIndex.load()
    q = exact_query("मौसम बारिश")
    plain = [d for d, _, _ in search(q, idx, k=5)]
    net = [d for d, _, _ in rank(q, idx, k=5, weights={"zone": 0, "prox": 0, "recency": 0})]
    assert plain == net


def test_headline_match_scores_higher_zone():
    idx = SampleIndex.load()
    results = {d: e for d, _, e in rank(exact_query("मौसम"), idx, k=10)}
    # nbt_2001 has मौसम in the headline, amarujala_3001 only in the body
    assert results["nbt_2001"]["zone"] > results["amarujala_3001"]["zone"]


def test_adjacent_query_words_get_full_proximity():
    idx = SampleIndex.load()
    results = {d: e for d, _, e in rank(exact_query("कोहली शतक"), idx, k=10)}
    # "कोहली का शतक" in the jagran_1002 headline puts the words 3 apart,
    # "कोहली के शतक" in aajtak_5002 too, so both get 2/3
    assert abs(results["jagran_1002"]["proximity"] - 2 / 3) < 1e-9
    assert abs(results["aajtak_5002"]["proximity"] - 2 / 3) < 1e-9


def test_recency_breaks_a_tie_between_similar_stories():
    idx = SampleIndex.load()
    # The two copies of the rail line wire story have the same body; the
    # original is older, so with only recency switched on the copy wins.
    ranked = [d for d, _, _ in rank(exact_query("रेल लाइन उद्घाटन"), idx, k=2,
                                    weights={"zone": 0, "prox": 0, "recency": 1.0})]
    assert set(ranked) == {"aajtak_5004", "amarujala_3004"}
