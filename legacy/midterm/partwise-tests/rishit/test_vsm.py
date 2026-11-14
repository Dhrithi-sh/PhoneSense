import math

from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.vsm import query_vector, search


class LectureIndex:
    """The worked lnc.ltc example from Lecture 6.

    Document: "car insurance auto insurance". Query: "best car insurance".
    N = 1,000,000 and the df values are the ones on the slide.
    """

    N = 1_000_000
    _df = {"auto": 5000, "best": 50000, "car": 10000, "insurance": 1000}
    _tf = {"auto": 1, "car": 1, "insurance": 2}

    def __init__(self):
        self.doc_norm = {"d1": math.sqrt(sum((1 + math.log10(tf)) ** 2 for tf in self._tf.values()))}

    def df(self, term):
        return self._df.get(term, 0)

    def postings(self, term, zone):
        if zone != "body" or term not in self._tf:
            return []
        return [("d1", self._tf[term], [])]


def test_lecture6_lnc_ltc_example_scores_about_0_8():
    idx = LectureIndex()
    results = search(exact_query("best car insurance"), idx)
    doc_id, score, explain = results[0]
    assert doc_id == "d1"
    # The slide gives 0 + 0 + 0.27 + 0.53 = 0.8
    assert abs(score - 0.80) < 0.01
    assert abs(explain["car"] - 0.27) < 0.01
    assert abs(explain["insurance"] - 0.53) < 0.01


def test_lecture6_query_weights_match_slide():
    qvec = query_vector(exact_query("best car insurance"), LectureIndex())
    # Normalised query weights on the slide: best 0.34, car 0.52, insurance 0.78
    assert abs(qvec["best"] - 0.34) < 0.01
    assert abs(qvec["car"] - 0.52) < 0.01
    assert abs(qvec["insurance"] - 0.78) < 0.01


def test_hindi_query_on_sample_index():
    idx = SampleIndex.load("none")
    results = search(exact_query("दिल्ली बारिश"), idx, k=5)
    top_ids = [doc_id for doc_id, _, _ in results]
    assert top_ids[0] in {"jagran_1001", "jagran_1004"}
    assert {"jagran_1001", "jagran_1004"} <= set(top_ids)
    scores = [s for _, s, _ in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 1.0001 for s in scores)


def test_k_limits_results():
    idx = SampleIndex.load("none")
    assert len(search(exact_query("मौसम बारिश दिल्ली"), idx, k=3)) == 3


def test_unknown_words_give_no_results():
    idx = SampleIndex.load("none")
    assert search(exact_query("xyzabc"), idx) == []


def test_stop_words_barely_matter():
    # का appears in many sample articles, so its idf is low and it shouldn't
    # change who comes first.
    idx = SampleIndex.load("none")
    with_stop = search(exact_query("बारिश का अलर्ट"), idx, k=1)[0][0]
    without = search(exact_query("बारिश अलर्ट"), idx, k=1)[0][0]
    assert with_stop == without
