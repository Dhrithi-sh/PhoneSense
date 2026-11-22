"""T17: PageRank and personalized PageRank. networkx is only a reference here."""

import math

import numpy as np
import pytest

from phonesense import io
from phonesense.graph.pagerank import (
    Graph,
    load_graph,
    pagerank,
    pagerank_scores,
    personalized_pagerank,
    power_iteration,
    scale_unit,
)
from phonesense.schemas import GraphEdge

TOY_IDS = ["p0", "p1", "p2", "p3", "p4", "p5"]
TOY_EDGES = [("p0", "p1"), ("p0", "p2"), ("p1", "p2"), ("p2", "p0"), ("p3", "p2"),
             ("p4", "p5")]  # p5 is dangling


@pytest.fixture
def toy_graph() -> Graph:
    return Graph.from_edges(TOY_IDS, TOY_EDGES)


def nx_toy():
    nx = pytest.importorskip("networkx")
    g = nx.DiGraph()
    g.add_nodes_from(TOY_IDS)
    g.add_edges_from(TOY_EDGES)
    return nx, g


def test_sums_to_one_and_positive(toy_graph):
    pr = pagerank(toy_graph)
    assert math.isclose(sum(pr.values()), 1.0, abs_tol=1e-9)
    assert all(v > 0 for v in pr.values())
    assert set(pr) == set(TOY_IDS)


def test_cycle_is_uniform():
    g = Graph.from_edges("abcd", [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    assert all(math.isclose(v, 0.25, abs_tol=1e-9) for v in pagerank(g).values())


def test_dangling_chain_keeps_mass():
    pi, _ = power_iteration(3, [0, 1], [1, 2])  # node 2 has no out-links
    assert math.isclose(pi.sum(), 1.0, abs_tol=1e-9)
    assert pi[2] > pi[1] > pi[0]


def test_damping_zero_returns_teleport():
    pi, _ = power_iteration(3, [0, 1], [1, 2], damping=0.0, teleport=[1, 1, 2])
    assert np.allclose(pi, [0.25, 0.25, 0.5])


def test_iteration_bound_from_lecture():
    # error shrinks by the damping factor each round: Err(t) <= d * Err(t-1)
    rng = np.random.default_rng(0)
    n, m, d, tol = 2000, 12000, 0.85, 1e-8
    _, it = power_iteration(n, rng.integers(0, n, m), rng.integers(0, n, m), damping=d, tol=tol)
    assert it <= math.ceil(math.log(tol) / math.log(d)) + 5


def test_bad_inputs():
    with pytest.raises(ValueError):
        power_iteration(2, [0], [1], damping=1.5)
    with pytest.raises(ValueError):
        power_iteration(2, [0], [1], teleport=[0, 0])
    with pytest.raises(ValueError):
        Graph.from_edges("ab", [("a", "b", 0.0)])
    with pytest.raises(ValueError):
        Graph.from_edges("aa", [])


def test_duplicate_edges_merge_and_self_loops_drop():
    g = Graph.from_edges("ab", [("a", "b"), ("a", "b", 2.0), ("a", "a")])
    assert g.src.tolist() == [0] and g.dst.tolist() == [1] and g.w.tolist() == [3.0]


def test_weights_split_rank_proportionally():
    g = Graph.from_edges("abc", [("a", "b", 3.0), ("a", "c", 1.0), ("b", "a"), ("c", "a")])
    pr = pagerank(g)
    assert pr["b"] > pr["c"]


def test_matches_networkx(toy_graph):
    nx, g = nx_toy()
    ref = nx.pagerank(g, alpha=0.85, tol=1e-13, max_iter=1000)
    ours = pagerank(toy_graph, tol=1e-13)
    assert max(abs(ours[i] - ref[i]) for i in TOY_IDS) < 1e-9


def test_personalized_matches_networkx(toy_graph):
    nx, g = nx_toy()
    seeds = {"p3": 2.0, "p4": 1.0}
    ref = nx.pagerank(g, alpha=0.85, personalization=seeds, tol=1e-13, max_iter=1000)
    ours = personalized_pagerank(toy_graph, seeds, tol=1e-13)
    assert max(abs(ours[i] - ref[i]) for i in TOY_IDS) < 1e-9


def test_personalized_favours_the_seed_side(toy_graph):
    glob = pagerank(toy_graph)
    ppr = personalized_pagerank(toy_graph, {"p4": 1.0})
    assert math.isclose(sum(ppr.values()), 1.0, abs_tol=1e-9)
    assert ppr["p4"] > glob["p4"] and ppr["p5"] > glob["p5"]
    assert ppr["p0"] == 0.0  # unreachable from the seed, and teleport never lands there
    with pytest.raises(KeyError):
        personalized_pagerank(toy_graph, {"nope": 1.0})


def test_scale_unit():
    s = scale_unit({"a": 0.1, "b": 0.4, "c": 0.4, "d": 0.1})
    assert s == {"a": 0.5, "b": 1.0, "c": 1.0, "d": 0.5}
    assert scale_unit({"a": 1.0, "b": 4.0}, method="max") == {"a": 0.25, "b": 1.0}
    with pytest.raises(ValueError):
        scale_unit({"a": 1.0}, method="rank")


def test_fixture_graph_loads(fixtures):
    g = load_graph(fixtures / "graph_edges.tsv", doc_ids=["B0TEST0004"])
    assert g.doc_ids == ["B0TEST0001", "B0TEST0002", "B0TEST0003", "B0TEST0004"]
    assert len(g.src) == 3


def test_pagerank_scores_roundtrip(toy_graph, tmp_path):
    path = tmp_path / "graph_edges.tsv"
    io.write_graph_edges(path, [GraphEdge(toy_graph.doc_ids[s], toy_graph.doc_ids[d], w)
                                for s, d, w in zip(toy_graph.src, toy_graph.dst, toy_graph.w,
                                                   strict=True)])
    scores = pagerank_scores(path=path)
    assert set(scores) == set(TOY_IDS)
    assert all(0.0 < v <= 1.0 for v in scores.values())
    pr = pagerank(toy_graph)
    assert max(scores, key=scores.get) == max(pr, key=pr.get)
    with pytest.raises(FileNotFoundError):
        pagerank_scores(path=tmp_path / "missing.tsv")
