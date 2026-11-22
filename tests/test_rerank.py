"""T18: prior reranking of run files."""

import pytest

from phonesense import io
from phonesense.graph.pagerank import Graph, load_graph
from phonesense.graph.rerank import (
    minmax,
    mix,
    ppr_rerank,
    rerank_with_prior,
    reranked_run_name,
)
from phonesense.schemas import validate_run_name

RUN = {"q01": [("a", 10.0), ("b", 6.0), ("c", 2.0)], "q02": [("c", 0.9), ("a", 0.1)]}


def order(ranking):
    return [d for d, _ in ranking]


def test_minmax():
    assert minmax([("a", 10.0), ("b", 6.0), ("c", 2.0)]) == [1.0, 0.5, 0.0]
    assert minmax([("a", -3.0), ("b", -3.0)]) == [1.0, 1.0]
    assert minmax([]) == []


def test_lambda_zero_keeps_order_and_one_follows_prior():
    prior = {"a": 0.0, "b": 0.5, "c": 1.0}
    assert {q: order(r) for q, r in rerank_with_prior(RUN, prior, 0.0).items()} == \
        {q: order(r) for q, r in RUN.items()}
    assert order(rerank_with_prior(RUN, prior, 1.0)["q01"]) == ["c", "b", "a"]


def test_mixing_is_the_documented_formula():
    out = dict(rerank_with_prior(RUN, {"a": 0.0, "b": 1.0, "c": 0.0}, 0.6)["q01"])
    assert out["a"] == pytest.approx(0.4 * 1.0)
    assert out["b"] == pytest.approx(0.4 * 0.5 + 0.6)
    assert out["c"] == pytest.approx(0.0)


def test_missing_prior_defaults_to_the_median():
    out = dict(rerank_with_prior(RUN, {"a": 0.2, "b": 0.4, "x": 0.9}, 1.0)["q01"])
    assert out["c"] == pytest.approx(0.4)
    out = dict(rerank_with_prior(RUN, {"a": 0.2}, 1.0, missing=0.0)["q01"])
    assert out["b"] == 0.0


def test_ties_break_by_doc_id_and_input_is_unchanged():
    run = {"q01": [("b", 1.0), ("a", 1.0)]}
    assert order(rerank_with_prior(run, {}, 0.5)["q01"]) == ["a", "b"]
    assert run == {"q01": [("b", 1.0), ("a", 1.0)]}
    with pytest.raises(ValueError):
        mix(RUN["q01"], {}, 1.5, 0.0)


def test_reranked_run_name_is_valid():
    name = reranked_run_name("bm25_k1.2_b0.75_dev", "pagerank0.2")
    assert name == "bm25_k1.2_b0.75+pagerank0.2_dev"
    assert validate_run_name(name) == name


def test_ppr_lifts_neighbours_of_the_top_result():
    # a is the top hit; c is linked from a, b is not linked at all
    graph = Graph.from_edges("abc", [("a", "c"), ("c", "a")])
    run = {"q01": [("a", 3.0), ("b", 2.0), ("c", 1.0)]}
    assert order(ppr_rerank(run, graph, 0.0)["q01"]) == ["a", "b", "c"]
    assert order(ppr_rerank(run, graph, 0.7, n_seeds=1)["q01"]) == ["a", "c", "b"]


def test_ppr_keeps_queries_outside_the_graph():
    graph = Graph.from_edges("xy", [("x", "y")])
    assert order(ppr_rerank(RUN, graph, 0.9)["q01"]) == ["a", "b", "c"]


def test_fixture_run_roundtrip(fixtures, tmp_path):
    run = io.read_run(fixtures / "sample.run")
    graph = load_graph(fixtures / "graph_edges.tsv")
    out = ppr_rerank(run, graph, 0.5)
    name = reranked_run_name("sample_dev", "ppr0.5")
    path = io.write_rankings(tmp_path / f"{name}.run", out, name)
    back = io.read_run(path)
    assert {q: order(r) for q, r in back.items()} == {q: order(r) for q, r in out.items()}
    assert {q: set(order(r)) for q, r in back.items()} == {q: set(order(r)) for q, r in run.items()}
