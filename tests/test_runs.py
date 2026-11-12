import pytest

from phonesense import io
from phonesense.schemas import MAX_RUN_DEPTH, RunEntry, SchemaError


def entries(rows, name="t_dev"):
    return [RunEntry(q, d, r, s, name) for q, d, r, s in rows]


def test_round_trip_lossless(tmp_path, fixtures):
    original = io.read_run_entries(fixtures / "sample.run")
    out = io.write_run(tmp_path / "x.run", original)
    assert io.read_run_entries(out) == original
    # awkward floats survive exactly
    tricky = entries([("q01", "a", 1, 0.1 + 0.2), ("q01", "b", 2, 1e-300),
                      ("q01", "c", 3, -3.141592653589793)])
    assert io.read_run_entries(io.write_run(tmp_path / "y.run", tricky)) == tricky


def test_rejects_duplicate_doc(tmp_path):
    with pytest.raises(SchemaError, match="duplicate"):
        io.write_run(tmp_path / "x.run", entries([("q01", "a", 1, 2.0), ("q01", "a", 2, 1.0)]))


def test_rejects_unsorted_scores(tmp_path):
    with pytest.raises(SchemaError, match="sorted"):
        io.write_run(tmp_path / "x.run", entries([("q01", "a", 1, 1.0), ("q01", "b", 2, 2.0)]))


@pytest.mark.parametrize("ranks", [(1, 3), (2, 3), (1, 1)])
def test_rejects_rank_gaps(tmp_path, ranks):
    rows = [("q01", "a", ranks[0], 2.0), ("q01", "b", ranks[1], 1.0)]
    with pytest.raises(SchemaError, match="rank"):
        io.write_run(tmp_path / "x.run", entries(rows))


def test_rejects_too_many_docs(tmp_path):
    rows = [("q01", f"d{i}", i, 1000.0 - i) for i in range(1, MAX_RUN_DEPTH + 2)]
    with pytest.raises(SchemaError, match=str(MAX_RUN_DEPTH)):
        io.write_run(tmp_path / "x.run", entries(rows))
    io.write_run(tmp_path / "ok.run", entries(rows[:MAX_RUN_DEPTH]))


def test_rejects_mixed_run_names_and_noncontiguous(tmp_path):
    mixed = entries([("q01", "a", 1, 1.0)], "a_dev") + entries([("q02", "a", 1, 1.0)], "b_dev")
    with pytest.raises(SchemaError, match="run_name"):
        io.write_run(tmp_path / "x.run", mixed)
    split = entries([("q01", "a", 1, 1.0), ("q02", "a", 1, 1.0), ("q01", "b", 2, 0.5)])
    with pytest.raises(SchemaError, match="contiguous"):
        io.write_run(tmp_path / "x.run", split)


@pytest.mark.parametrize("name", ["bm25", "bm25 dev", "bm25_train", "_dev"])
def test_rejects_bad_run_names(name):
    with pytest.raises(SchemaError):
        RunEntry("q01", "a", 1, 1.0, name).validate()


@pytest.mark.parametrize("text", [
    "q01 Q0 a 1 1.0\n",            # five fields
    "q01 X a 1 1.0 t_dev\n",       # no Q0
    "q01 Q0 a 1 nan t_dev\n",      # nan score
    "query1 Q0 a 1 1.0 t_dev\n",   # bad qid
    "q01 Q0 a 0 1.0 t_dev\n",      # rank 0
])
def test_reader_rejects_bad_lines(tmp_path, text):
    p = tmp_path / "bad.run"
    p.write_text(text)
    with pytest.raises(SchemaError):
        io.read_run_entries(p)


def test_rankings_to_entries_sorts_ties_and_truncates():
    es = io.rankings_to_entries({"q02": [("b", 1.0), ("a", 1.0), ("c", 5.0)]}, "t_dev", k=2)
    assert [(e.doc_id, e.rank) for e in es] == [("c", 1), ("a", 2)]
    with pytest.raises(SchemaError, match="duplicate"):
        io.rankings_to_entries({"q01": [("a", 1.0), ("a", 0.5)]}, "t_dev")
