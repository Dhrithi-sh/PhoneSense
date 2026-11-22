"""Dense re-ranking, tested with a tiny fake encoder so no model is needed."""

import math

from dhvani.rank.dense import DenseIndex, dense_query_text, dense_rerank
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.vsm import search


class FakeEncoder:
    """Vectors from a few marker words, so similarity is easy to predict."""

    name = "fake"
    MARKERS = ("बारिश", "मौसम", "शतक", "चुनाव")

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += len(texts)
        out = []
        for text in texts:
            v = [float(text.count(m)) for m in self.MARKERS] + [0.1]
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v])
        return out


def _query(raw, expansions):
    tokens = [{"surface": s, "script": "roman", "lang": {}, "expansions": e} for s, e in expansions]
    return {"raw": raw, "tokens": tokens}


def test_query_text_adds_confident_devanagari_spellings_only():
    q = _query("kisan bhukamp delhi", [
        ("kisan", [("kisan", 1.0, "exact"), ("किसने", 0.14, "phonetic")]),
        ("bhukamp", [("bhukamp", 1.0, "exact"), ("भूकंप", 0.67, "phonetic")]),
        ("delhi", [("delhi", 1.0, "exact"), ("दिल्ली", 0.1, "xling")]),
    ])
    assert dense_query_text(q) == "kisan bhukamp delhi भूकंप दिल्ली"


def test_hindi_query_text_is_unchanged():
    assert dense_query_text(exact_query("दिल्ली बारिश")) == "दिल्ली बारिश"


def test_article_vectors_are_cached(tmp_path):
    idx = SampleIndex.load()
    enc = FakeEncoder()
    DenseIndex(idx, enc, cache_dir=tmp_path)
    assert enc.calls == idx.N
    enc2 = FakeEncoder()
    DenseIndex(idx, enc2, cache_dir=tmp_path)
    assert enc2.calls == 0


def test_alpha_zero_keeps_the_first_stage_order():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    base = search(q, idx, k=8)
    out = dense_rerank(base, q, DenseIndex(idx, FakeEncoder(), cache_dir=None), alpha=0.0)
    assert [d for d, _, _ in out] == [d for d, _, _ in base]


def test_alpha_one_orders_by_dense_similarity():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    out = dense_rerank(search(q, idx, k=8), q, DenseIndex(idx, FakeEncoder(), cache_dir=None), alpha=1.0)
    cosines = [e["dense"]["cosine"] for _, _, e in out]
    assert cosines == sorted(cosines, reverse=True)
    assert all({"cosine", "first_stage", "alpha", "score"} <= set(e["dense"]) for _, _, e in out)


def test_parser_stages_stay_in_order():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    # the parser keeps term scores under "terms" and adds the stage next to them
    base = [(d, s, {"terms": e, "stage": 0 if i < 2 else 1}) for i, (d, s, e) in enumerate(search(q, idx, k=6))]
    out = dense_rerank(base, q, DenseIndex(idx, FakeEncoder(), cache_dir=None), alpha=1.0)
    assert [e["stage"] for _, _, e in out] == [0, 0] + [1] * (len(base) - 2)


def test_results_past_the_depth_keep_their_place():
    idx = SampleIndex.load()
    q = exact_query("दिल्ली बारिश")
    base = search(q, idx, k=8)
    out = dense_rerank(base, q, DenseIndex(idx, FakeEncoder(), cache_dir=None), depth=3)
    assert out[3:] == base[3:]
    assert {d for d, _, _ in out[:3]} == {d for d, _, _ in base[:3]}
