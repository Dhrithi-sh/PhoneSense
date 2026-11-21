from dhvani.rank.diversify import diversify, text_vector
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.vsm import search


class Idx:
    def __init__(self, articles):
        self.articles = articles


def _results(*ids):
    return [(d, 1.0 - i * 0.1, {"t": 1.0}) for i, d in enumerate(ids)]


def test_text_vectors_are_unit_length():
    v = text_vector({"headline": "दिल्ली में बारिश", "body": "बारिश बारिश अलर्ट"})
    assert abs(sum(w * w for w in v.values()) - 1.0) < 1e-9


def test_lambda_one_keeps_the_original_order():
    idx = SampleIndex.load()
    base = search(exact_query("दिल्ली बारिश"), idx, k=5)
    assert [d for d, _, _ in diversify(base, idx, lam=1.0)] == [d for d, _, _ in base]


def test_a_near_copy_is_pushed_down():
    idx = Idx({"a": {"headline": "रेल लाइन उद्घाटन", "body": "प्रधानमंत्री रेल"},
               "b": {"headline": "रेल लाइन उद्घाटन", "body": "प्रधानमंत्री रेल"},
               "c": {"headline": "कोहली शतक", "body": "क्रिकेट मैच"}})
    out = [d for d, _, _ in diversify(_results("a", "b", "c"), idx, lam=0.5)]
    assert out == ["a", "c", "b"]


def test_stages_stay_in_order():
    idx = Idx({"a": {"headline": "x"}, "b": {"headline": "x"}, "c": {"headline": "y"}})
    base = [("a", 0.9, {"terms": {}, "stage": 0}), ("b", 0.8, {"terms": {}, "stage": 0}),
            ("c", 0.95, {"terms": {}, "stage": 1})]
    out = diversify(base, idx, lam=0.1)
    assert [e["stage"] for _, _, e in out] == [0, 0, 1]
    assert all("mmr" in e for _, _, e in out)


def test_cut_to_k():
    idx = SampleIndex.load()
    base = search(exact_query("दिल्ली बारिश"), idx, k=5)
    assert len(diversify(base, idx, k=2)) == 2
