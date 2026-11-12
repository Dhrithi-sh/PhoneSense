"""Reviews, graph files and results."""

import json

import pytest

from phonesense import io
from phonesense.schemas import GraphEdge, Results, Review, SchemaError, make_review_id

TS = 1700000000000
REVIEW = {"review_id": make_review_id("U1", "B0TEST0001", TS),
          "doc_id": "B0TEST0001", "user_id": "U1", "rating": 4, "title": "t", "text": "x",
          "timestamp": TS, "helpful_votes": 0, "verified": True}


def test_review_defaults_and_id():
    r = Review.from_dict(REVIEW)
    assert r.is_synthetic is False
    assert len(r.review_id) == 16


@pytest.mark.parametrize("patch", [{"rating": 0}, {"rating": 6}, {"rating": 4.0},
                                   {"timestamp": TS // 1000},  # seconds, not ms
                                   {"verified": 1}, {"review_id": "abc"},
                                   {"helpful_votes": -1}, {"is_synthetic": "no"}])
def test_review_rejects(patch):
    with pytest.raises(SchemaError):
        Review.from_dict({**REVIEW, **patch})


def test_synthetic_reviews_need_opt_in(tmp_path):
    r = Review.from_dict({**REVIEW, "is_synthetic": True})
    p = io.write_reviews(tmp_path / "r.jsonl", [r])
    with pytest.raises(SchemaError, match="synthetic"):
        io.read_reviews(p)
    assert io.read_reviews(p, allow_synthetic=True) == [r]


def test_reviews_round_trip(tmp_path, fixtures):
    reviews = io.read_reviews(fixtures / "reviews.jsonl")
    assert io.read_reviews(io.write_reviews(tmp_path / "r.jsonl", reviews)) == reviews


def test_graph_round_trip_and_rejects(tmp_path, fixtures):
    edges = io.read_graph_edges(fixtures / "graph_edges.tsv")
    assert io.read_graph_edges(io.write_graph_edges(tmp_path / "g.tsv", edges)) == edges
    for bad in [GraphEdge("a", "a", 1.0), GraphEdge("a", "b", 0.0), GraphEdge("a", "b", -1)]:
        with pytest.raises(SchemaError):
            io.write_graph_edges(tmp_path / "bad.tsv", [bad])
    (tmp_path / "nohdr.tsv").write_text("a\tb\t1.0\n")
    with pytest.raises(SchemaError, match="header"):
        io.read_graph_edges(tmp_path / "nohdr.tsv")


def test_trust_round_trip_and_range(tmp_path, fixtures):
    trust = io.read_reviewer_trust(fixtures / "reviewer_trust.tsv")
    assert io.read_reviewer_trust(io.write_reviewer_trust(tmp_path / "t.tsv", trust)) == trust
    with pytest.raises(SchemaError):
        io.write_reviewer_trust(tmp_path / "bad.tsv", {"u": 1.5})


def test_results_round_trip_and_rules(tmp_path, fixtures):
    r = io.read_results(fixtures / "sample_dev.json")
    assert io.read_results(io.write_results(tmp_path / "r.json", r)) == r
    base = json.loads((fixtures / "sample_dev.json").read_text())
    bad_cases = [
        {**base, "per_query": {}},                          # per-query required
        {**base, "split": "test"},                          # split must match run name
        {**base, "per_query": {"q01": {"P@10": 0.1}}},      # same metric keys
        {k: v for k, v in base.items() if k != "qrels"},    # missing key
    ]
    for bad in bad_cases:
        with pytest.raises(SchemaError):
            Results.from_dict(bad)
