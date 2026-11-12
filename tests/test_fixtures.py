"""Every fixture loads and validates through schemas/io."""

from phonesense import io
from phonesense.schemas import build_text, run_name_split


def test_phones(fixtures):
    phones = io.read_phones(fixtures / "phones.jsonl")
    assert [p.doc_id for p in phones] == ["B0TEST0001", "B0TEST0002", "B0TEST0003"]
    assert phones[2].specs.ram_gb is None and phones[2].brand is None
    for p in phones:  # fixtures follow the canonical text recipe
        assert p.text == build_text(p.fields)


def test_reviews(fixtures):
    reviews = io.read_reviews(fixtures / "reviews.jsonl")
    assert len(reviews) == 4 and not any(r.is_synthetic for r in reviews)
    phone_ids = {p.doc_id for p in io.read_phones(fixtures / "phones.jsonl")}
    assert {r.doc_id for r in reviews} <= phone_ids


def test_queries(fixtures):
    qs = io.read_queries(fixtures / "queries.jsonl")
    assert [q.qid for q in qs] == ["q01", "q02", "q03"]
    dev = io.read_queries(fixtures / "queries.jsonl", split="dev")
    assert [q.qid for q in dev] == ["q01", "q02"]
    assert qs[1].filters is None and qs[2].intent_notes is None


def test_qrels(fixtures):
    qrels = io.read_qrels(fixtures / "sample_qrels.txt")
    assert qrels["q01"] == {"B0TEST0001": 3, "B0TEST0002": 0, "B0TEST0003": 1}


def test_run(fixtures):
    run = io.read_run(fixtures / "sample.run")
    assert run["q01"][0] == ("B0TEST0001", 12.5)
    assert run_name_split(io.read_run_entries(fixtures / "sample.run")[0].run_name) == "dev"


def test_graph_files(fixtures):
    edges = io.read_graph_edges(fixtures / "graph_edges.tsv")
    assert len(edges) == 3 and edges[2].weight == 0.5
    trust = io.read_reviewer_trust(fixtures / "reviewer_trust.tsv")
    assert trust == {"AE7K2QLXDZ4P": 0.92, "AF3NW9RTY81B": 0.5, "AG5HXB2MPQ7C": 0.0}


def test_results(fixtures):
    r = io.read_results(fixtures / "sample_dev.json")
    assert r.split == "dev" and set(r.per_query) == {"q01", "q02"}
