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


# ---------------------------------------------------------------------------- deepening


def test_queries_without_relevant_lists_only_judged_queries_with_no_grade_2():
    from phonesense.llm.pooling import queries_without_relevant

    qrels = {"q01": {"a": 3, "b": 0}, "q02": {"a": 1, "b": 0}, "q03": {"a": 2}, "q04": {}}
    assert queries_without_relevant(qrels) == ["q02"]  # q04 has no judgments: not judged yet
    assert queries_without_relevant(qrels, threshold=1) == []
    assert queries_without_relevant({}) == []


def test_deepen_pool_changes_only_the_listed_queries_and_keeps_old_pairs():
    from phonesense.llm.pooling import deepen_pool

    runs = {"s": {"q01": [("a", 3.0), ("b", 2.0), ("c", 1.0)], "q02": [("x", 1.0), ("y", 0.5)]}}
    shallow = pool_runs(runs, depth=1)
    assert shallow == {"q01": {"a": ["s"]}, "q02": {"x": ["s"]}}
    deep = deepen_pool(shallow, runs, depth=3, qids=["q01"])
    assert set(deep["q01"]) == {"a", "b", "c"} and set(deep["q01"]) >= set(shallow["q01"])
    assert deep["q02"] == shallow["q02"]  # untouched
    assert deepen_pool(shallow, runs, depth=3, qids=[]) == shallow
    assert shallow == {"q01": {"a": ["s"]}, "q02": {"x": ["s"]}}  # the input is not modified
