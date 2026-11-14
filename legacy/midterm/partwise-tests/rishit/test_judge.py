from dhvani.eval.judge import (
    all_judgments,
    build_pool,
    progress,
    read_need_descriptions,
    read_person,
    read_pool,
    save_judgment,
    write_pool,
)

QUERIES = [("R01_hi", "R01", "hindi", "x"), ("R01_en", "R01", "english", "x"), ("R02_hi", "R02", "hindi", "y")]


def test_pool_merges_forms_and_runs_per_need():
    runs = {"a": {"R01_hi": ["d1", "d2"], "R01_en": ["d2", "d3"], "R02_hi": ["d9"]},
            "b": {"R01_hi": ["d4", "d1"]}}
    pool = build_pool(runs, QUERIES, k=10)
    assert sorted(pool["R01"]) == ["d1", "d2", "d3", "d4"]      # each article once per need
    assert pool["R02"] == ["d9"]


def test_pool_respects_k():
    pool = build_pool({"a": {"R01_hi": ["d1", "d2", "d3"]}}, QUERIES, k=2)
    assert pool["R01"] == ["d1", "d2"]


def test_pool_round_trips(tmp_path):
    pool = {"R01": ["d1", "d2"], "R02": ["d9"]}
    write_pool(pool, tmp_path / "pool.tsv")
    assert read_pool(tmp_path / "pool.tsv") == pool


def test_saving_overwrites_the_same_pair(tmp_path):
    save_judgment("rishit", "R01", "d1", 2, tmp_path)
    save_judgment("rishit", "R01", "d1", 1, tmp_path)
    save_judgment("rishit", "R01", "d2", 0, tmp_path)
    assert read_person("rishit", tmp_path) == {"R01": {"d1": 1, "d2": 0}}
    lines = (tmp_path / "qrels_rishit.txt").read_text().splitlines()
    assert lines == ["R01 0 d1 1", "R01 0 d2 0"]          # formats.md judgment format


def test_bad_grade_is_refused(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        save_judgment("rishit", "R01", "d1", 3, tmp_path)


def test_everyones_files_are_merged_keeping_the_higher_grade(tmp_path):
    save_judgment("rishit", "R01", "d1", 1, tmp_path)
    save_judgment("viraja", "R01", "d1", 2, tmp_path)
    save_judgment("viraja", "V01", "d5", 0, tmp_path)
    assert all_judgments(tmp_path) == {"R01": {"d1": 2}, "V01": {"d5": 0}}


def test_progress_counts_judged_articles():
    assert progress({"R01": ["d1", "d2", "d3"]}, {"R01": {"d1": 2}}) == {"R01": (1, 3)}


def test_need_descriptions_come_from_the_needs_files():
    d = read_need_descriptions()
    assert d["R01"].startswith("Earthquake") and "V01" in d


def test_merging_pools_keeps_everything_old_and_adds_new():
    from dhvani.eval.judge import merge_pools
    merged = merge_pools({"R01": ["d1", "d2"]}, {"R01": ["d2", "d3"], "R02": ["d9"]})
    assert merged == {"R01": ["d1", "d2", "d3"], "R02": ["d9"]}


def test_runner_pushes_listing_pages_down_by_default():
    from dhvani.eval.experiments import run_one
    from dhvani.rank.sample_index import SampleIndex
    q = [("X_hi", "X", "hindi", "दिल्ली बारिश")]
    idx = SampleIndex.load()
    assert run_one(idx, q, "net", k=3) == run_one(idx, q, "net", k=3, demote_listings=False)   # sample has no listings
