import math

from dhvani.rank.bm25 import bm25_scores, doc_lengths, idf, search_bm25
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex


def tiny_index():
    # Three short articles with known lengths: 4, 2 and 6 words.
    articles = [
        {"doc_id": "a", "headline": "बारिश", "body": "बारिश बारिश दिल्ली", "date": None},
        {"doc_id": "b", "headline": "दिल्ली", "body": "मौसम", "date": None},
        {"doc_id": "c", "headline": "मौसम", "body": "मौसम बारिश पटना पटना पटना", "date": None},
    ]
    return SampleIndex(articles)


def test_doc_lengths_add_up_both_zones():
    assert doc_lengths(tiny_index()) == {"a": 4, "b": 2, "c": 6}


def test_idf_stays_positive_for_common_words():
    # A word in every document would get a negative classic idf.
    assert idf(df=3, n=3) > 0
    assert idf(df=1, n=3) > idf(df=2, n=3)


def test_hand_computed_score():
    # Query बारिश on article a: tf = 3, dl = 4, avgdl = 4, df = 2, N = 3.
    idx = tiny_index()
    scores, _ = bm25_scores(exact_query("बारिश"), idx)
    expected_idf = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    k1, b = 1.2, 0.75
    expected = expected_idf * 3 * (k1 + 1) / (3 + k1 * (1 - b + b * 4 / 4))
    assert abs(scores["a"] - expected) < 1e-12


def test_tf_saturates():
    # Going from 1 to 2 occurrences should help more than 9 to 10.
    k1, b, avgdl, dl = 1.2, 0.75, 100, 100

    def part(tf):
        return tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    assert part(2) - part(1) > part(10) - part(9)
    assert part(1000) < k1 + 1  # never goes past the k1 + 1 ceiling


def test_longer_article_scores_lower_at_same_tf():
    articles = [
        {"doc_id": "short", "headline": "", "body": "मौसम आज", "date": None},
        {"doc_id": "long", "headline": "", "body": "मौसम " + "खबर " * 20, "date": None},
        {"doc_id": "other", "headline": "", "body": "क्रिकेट", "date": None},
    ]
    scores, _ = bm25_scores(exact_query("मौसम"), SampleIndex(articles))
    assert scores["short"] > scores["long"]


def test_b_zero_ignores_length():
    articles = [
        {"doc_id": "short", "headline": "", "body": "मौसम आज", "date": None},
        {"doc_id": "long", "headline": "", "body": "मौसम " + "खबर " * 20, "date": None},
        {"doc_id": "other", "headline": "", "body": "क्रिकेट", "date": None},
    ]
    scores, _ = bm25_scores(exact_query("मौसम"), SampleIndex(articles), b=0.0)
    assert abs(scores["short"] - scores["long"]) < 1e-12


def test_headline_boost_favours_headline_matches():
    idx = SampleIndex.load()
    plain = dict((d, s) for d, s, _ in search_bm25(exact_query("मौसम"), idx, k=10))
    boosted = dict((d, s) for d, s, _ in search_bm25(exact_query("मौसम"), idx, k=10, headline_boost=2.0))
    # nbt_2001 has मौसम in its headline, amarujala_3001 doesn't
    assert boosted["nbt_2001"] - plain["nbt_2001"] > boosted["amarujala_3001"] - plain["amarujala_3001"]


def test_sample_queries_rank_sensibly():
    idx = SampleIndex.load()
    top = [d for d, _, _ in search_bm25(exact_query("कोहली शतक"), idx, k=2)]
    assert set(top) == {"jagran_1002", "aajtak_5002"}
    top = [d for d, _, _ in search_bm25(exact_query("दिल्ली बारिश"), idx, k=2)]
    assert set(top) == {"jagran_1001", "jagran_1004"}


def test_unknown_word_gives_nothing():
    assert search_bm25(exact_query("xyzabc"), SampleIndex.load()) == []
