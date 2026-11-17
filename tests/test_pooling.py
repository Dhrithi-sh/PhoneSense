"""Pooling: union of top-depth documents, system attribution, determinism, file round trip."""

import pytest

from phonesense.llm.pooling import pool_runs, read_pool, write_pool

RUNS = {
    "bm25_dev": {"q01": [("d1", 3.0), ("d2", 2.0), ("d3", 1.0)], "q02": [("d9", 1.0)]},
    "dense_dev": {"q01": [("d2", 0.9), ("d4", 0.8), ("d5", 0.1)]},
}


def test_union_of_top_depth_with_attribution():
    pool = pool_runs(RUNS, depth=2)
    assert pool == {
        "q01": {"d1": ["bm25_dev"], "d2": ["bm25_dev", "dense_dev"], "d4": ["dense_dev"]},
        "q02": {"d9": ["bm25_dev"]},
    }


def test_depth_larger_than_a_run_takes_the_whole_run():
    assert set(pool_runs(RUNS, depth=50)["q01"]) == {"d1", "d2", "d3", "d4", "d5"}


def test_only_requested_queries_are_pooled():
    assert list(pool_runs(RUNS, depth=2, qids=["q02"])) == ["q02"]


def test_pool_is_independent_of_run_dict_order():
    reversed_runs = dict(reversed(list(RUNS.items())))
    assert pool_runs(RUNS, 2) == pool_runs(reversed_runs, 2)


def test_bad_depth_is_rejected():
    with pytest.raises(ValueError):
        pool_runs(RUNS, depth=0)


def test_file_round_trip(tmp_path):
    pool = pool_runs(RUNS, depth=2)
    path = write_pool(tmp_path / "sub" / "pool_dev.tsv", pool)
    assert path.read_text().splitlines()[0] == "qid\tdoc_id\tsystems"
    assert read_pool(path) == pool


def test_read_rejects_a_wrong_header(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("qid\tdoc\n")
    with pytest.raises(ValueError):
        read_pool(bad)
