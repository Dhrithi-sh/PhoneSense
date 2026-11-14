from dhvani.rank.fusion import RRF_C, dense_list, rrf, search_rrf
from dhvani.rank.parser import parse_and_rank
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex


def _list(*ids):
    return [(d, 1.0 / (i + 1), {"t": 1.0}) for i, d in enumerate(ids)]


def test_rrf_formula():
    out = rrf({"a": _list("x", "y"), "b": _list("y", "z")}, k=None)
    scores = {d: s for d, s, _ in out}
    assert abs(scores["y"] - (1 / (RRF_C + 2) + 1 / (RRF_C + 1))) < 1e-12
    assert abs(scores["x"] - 1 / (RRF_C + 1)) < 1e-12


def test_agreement_beats_a_single_first_place():
    # y is second in both lists, x and z are first in only one each
    out = rrf({"a": _list("x", "y"), "b": _list("z", "y")}, k=3)
    assert out[0][0] == "y"


def test_explain_has_ranks_in_every_list():
    out = rrf({"a": _list("x", "y"), "b": _list("y")}, k=None)
    explain = dict((d, e) for d, _, e in out)
    assert explain["x"]["rrf"] == {"a": 1, "b": None}
    assert explain["y"]["rrf"] == {"a": 2, "b": 1}
    assert explain["x"]["terms"] == {"t": 1.0}


def test_search_rrf_fuses_the_three_sparse_rankers():
    idx = SampleIndex.load()
    results = search_rrf(exact_query("दिल्ली बारिश"), idx, k=5)
    assert results
    assert set(results[0][2]["rrf"]) == {"lnc.ltc", "BM25", "net"}
    scores = [s for _, s, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_list_orders_candidates_by_cosine():
    class FakeDense:
        def similarity(self, query, doc_ids):
            return {d: float(len(d)) for d in doc_ids}
    out = dense_list(exact_query("x"), FakeDense(), ["ab", "abcd", "abc"])
    assert [d for d, _, _ in out] == ["abcd", "abc", "ab"]


def test_dense_joins_the_fusion_when_given():
    class FakeDense:
        def similarity(self, query, doc_ids):
            return {d: 0.5 for d in doc_ids}
    idx = SampleIndex.load()
    results = search_rrf(exact_query("दिल्ली बारिश"), idx, k=3, dense=FakeDense())
    assert "dense" in results[0][2]["rrf"]


def test_parser_works_with_rrf():
    idx = SampleIndex.load()
    results = parse_and_rank(exact_query("दिल्ली बारिश"), idx, k=3, ranker="rrf")
    assert results and all("stage" in e and "rrf" in e for _, _, e in results)
