"""T16: phone graph builders."""

import numpy as np
import pytest

from phonesense import io
from phonesense.graph.build_graph import build_graph, co_review_edges, graph_stats, knn_edges
from phonesense.graph.pagerank import Graph, load_graph, pagerank
from phonesense.schemas import Review, make_review_id
from phonesense.trust.toy import make_toy_reviews

TS = 1_700_000_000_000


def rv(user_id: str, doc_id: str) -> Review:
    return Review(make_review_id(user_id, doc_id, TS), doc_id, user_id, 4, "", "", TS, 0, True)


def test_co_review_needs_min_shared_and_is_symmetric():
    reviews = [rv("u1", "a"), rv("u1", "b"), rv("u2", "a"), rv("u2", "b"), rv("u2", "c"),
               rv("u4", "zzz"), rv("u4", "a")]  # phone outside the subset is ignored
    assert co_review_edges(reviews, "abcde", min_shared=2, cap=5) == [("a", "b", 0.4),
                                                                      ("b", "a", 0.4)]
    assert len(co_review_edges(reviews, "abcde", min_shared=1)) == 6  # a-b, a-c, b-c


def test_co_review_caps_weight_and_skips_heavy_reviewers():
    reviews = [rv(f"u{i}", d) for i in range(9) for d in "ab"]
    assert co_review_edges(reviews, "ab") == [("a", "b", 1.0), ("b", "a", 1.0)]
    heavy = [rv("bot", d) for d in "abcde"] + [rv("bot2", d) for d in "abcde"]
    assert co_review_edges(heavy, "abcde", min_shared=1, max_user_phones=4) == []


def test_knn_picks_nearest_and_never_self():
    ids = ["a", "b", "c", "d"]
    vecs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    edges = knn_edges(ids, vecs, k=1)
    assert [(s, d) for s, d, _ in edges] == [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")]
    assert all(s != d for s, d, _ in knn_edges(ids, vecs, k=3))
    assert len(knn_edges(ids, vecs, k=10)) <= 4 * 3  # k is capped at n - 1


def test_knn_chunking_gives_the_same_edges():
    rng = np.random.default_rng(0)
    ids = [f"p{i:02d}" for i in range(25)]
    vecs = rng.normal(size=(25, 6))
    small, big = knn_edges(ids, vecs, k=4, chunk=7), knn_edges(ids, vecs, k=4, chunk=1000)
    assert [e[:2] for e in small] == [e[:2] for e in big]
    assert np.allclose([e[2] for e in small], [e[2] for e in big])


def test_knn_drops_non_positive_similarity_and_checks_shape():
    vecs = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    assert knn_edges(["a", "b", "c"], vecs, k=2) == []
    with pytest.raises(ValueError):
        knn_edges(["a"], np.zeros((2, 3)))


def test_build_graph_merges_types_and_reports_stats():
    reviews = [rv("u1", "a"), rv("u1", "b"), rv("u2", "a"), rv("u2", "b")]
    vecs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1], [0, 0, 1], [1, 1, 1]])
    emb = (["a", "b", "c", "d", "e", "not-a-phone"], vecs)
    edges, graph, stats = build_graph("abcde", reviews, emb, knn_k=1)
    assert graph.doc_ids == ["a", "b", "c", "d", "e"]
    pairs = {(e.src_doc_id, e.dst_doc_id): e.weight for e in edges}
    assert len(pairs) == len(edges) and all(s != d for s, d in pairs)
    assert set(pairs) == {("a", "b"), ("b", "a"), ("c", "d"), ("d", "c"), ("e", "d")}
    cos_ab = 0.9 / np.hypot(0.9, 0.1)
    assert pairs[("a", "b")] == pytest.approx(0.4 + cos_ab)  # co-review + kNN, summed
    assert stats["types"]["co_review"]["edges"] == 2
    assert stats["types"]["knn"]["edges"] == 5
    assert stats["merged"] == {"edges": 5, "dangling_phones": 0, "isolated_phones": 0,
                               "largest_component": 3}


def test_build_graph_with_nothing_but_phones():
    edges, graph, stats = build_graph("abc")
    assert edges == [] and stats["types"] == {}
    assert stats["merged"] == {"edges": 0, "dangling_phones": 3, "isolated_phones": 3,
                               "largest_component": 1}
    assert abs(sum(pagerank(graph).values()) - 1.0) < 1e-9


def test_graph_stats_on_an_empty_graph():
    stats = graph_stats(Graph.from_edges([], []), {})
    assert stats["merged"]["edges"] == 0 and stats["merged"]["largest_component"] == 0


def test_toy_graph_roundtrips_through_the_contract_file(tmp_path):
    reviews = make_toy_reviews()
    doc_ids = {r.doc_id for r in reviews}
    edges, graph, _ = build_graph(doc_ids, reviews)
    path = io.write_graph_edges(tmp_path / "graph_edges.tsv", edges)
    back = load_graph(path, doc_ids)
    assert back.doc_ids == graph.doc_ids
    assert pagerank(back) == pagerank(graph)
