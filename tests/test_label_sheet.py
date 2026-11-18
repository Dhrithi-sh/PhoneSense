"""Hand-labelling sheets: sampling, disjoint shares, the sheet contents and the import."""

import csv

import pytest

from phonesense.io import read_phones, read_queries
from phonesense.llm.label_sheet import read_sheet, sample_pairs, share_of, write_sheet

POOL = {
    "q01": {"d1": ["a"], "d2": ["a"], "d3": ["b"]},
    "q02": {"d1": ["a"], "d4": ["b"]},
    "q03": {"d2": ["a"]},
}


def test_sample_is_deterministic_sorted_and_spread_over_queries():
    a, b = sample_pairs(POOL, 3, seed=7), sample_pairs(POOL, 3, seed=7)
    assert a == b and a == sorted(a)
    assert {q for q, _ in a} == {"q01", "q02", "q03"}  # one pair per query first
    assert all(d in POOL[q] for q, d in a)


def test_sample_never_exceeds_the_pool_and_changes_with_the_seed():
    assert len(sample_pairs(POOL, 100, seed=1)) == 6
    draws = {tuple(sample_pairs(POOL, 3, seed=s)) for s in range(20)}
    assert len(draws) > 1
    with pytest.raises(ValueError):
        sample_pairs(POOL, 0, seed=1)


def test_shares_are_disjoint_and_cover_the_sample():
    pairs = sample_pairs(POOL, 6, seed=3)
    shares = [share_of(pairs, i, 3) for i in (1, 2, 3)]
    assert sorted(p for s in shares for p in s) == pairs
    assert len({p for s in shares for p in s}) == len(pairs)
    assert [len(s) for s in shares] == [2, 2, 2]
    assert share_of(pairs, 1, 1) == pairs


@pytest.mark.parametrize("index, n", [(0, 3), (4, 3), (1, 0)])
def test_bad_share_is_rejected(index, n):
    with pytest.raises(ValueError):
        share_of([("q01", "d1")], index, n)


@pytest.fixture
def world(fixtures):
    queries = {q.qid: q for q in read_queries(fixtures / "queries.jsonl")}
    phones = {p.doc_id: p for p in read_phones(fixtures / "phones.jsonl")}
    return queries, phones


def test_sheet_has_empty_grades_and_no_llm_or_system_info(tmp_path, world):
    queries, phones = world
    pairs = [("q01", "B0TEST0001"), ("q02", "B0TEST0002")]
    path = write_sheet(tmp_path / "out" / "s.csv", pairs, queries, phones)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["grade"] for r in rows] == ["", ""]
    assert rows[0]["qid"] == "q01" and rows[0]["doc_id"] == "B0TEST0001"
    assert queries["q01"].text in rows[0]["query"] and "0 = not relevant" in rows[0]["query"]
    assert "Nova X5" in rows[0]["phone"]
    assert "avg_rating" not in rows[0]["phone"] and "bm25" not in path.read_text().lower()


def _fill(path, grades):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row, g in zip(rows, grades, strict=True):
        row["grade"] = g
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def test_read_sheet_returns_grades_and_skips_blanks(tmp_path, world):
    queries, phones = world
    path = write_sheet(tmp_path / "s.csv", [("q01", "B0TEST0001"), ("q02", "B0TEST0002"),
                                            ("q03", "B0TEST0003")], queries, phones)
    _fill(path, ["3", "", " 1 "])
    assert read_sheet(path) == {"q01": {"B0TEST0001": 3}, "q03": {"B0TEST0003": 1}}


@pytest.mark.parametrize("bad", ["4", "-1", "2.5", "two", "10"])
def test_read_sheet_rejects_bad_grades_with_the_row_number(tmp_path, world, bad):
    queries, phones = world
    path = write_sheet(tmp_path / "s.csv", [("q01", "B0TEST0001")], queries, phones)
    _fill(path, [bad])
    with pytest.raises(ValueError, match="row 2"):
        read_sheet(path)


def test_read_sheet_needs_the_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("qid,doc\nq01,d1\n")
    with pytest.raises(ValueError, match="missing column"):
        read_sheet(bad)
