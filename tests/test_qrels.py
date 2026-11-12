import pytest

from phonesense import io
from phonesense.schemas import Qrel, SchemaError


def test_all_grades_parse(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text("".join(f"q01 0 d{g} {g}\n" for g in range(4)))
    assert io.read_qrels(p) == {"q01": {"d0": 0, "d1": 1, "d2": 2, "d3": 3}}
    assert [q.relevant for q in io.read_qrels_entries(p)] == [False, False, True, True]


@pytest.mark.parametrize("line", ["q01 0 d1 4", "q01 0 d1 -1", "q01 0 d1 2.0", "q01 0 d1 x",
                                  "q01 1 d1 2", "q01 0 d1", "q1 0 d1 2"])
def test_rejects_bad_lines(tmp_path, line):
    p = tmp_path / "q.txt"
    p.write_text(line + "\n")
    with pytest.raises(SchemaError):
        io.read_qrels(p)


def test_rejects_duplicates(tmp_path):
    p = tmp_path / "q.txt"
    p.write_text("q01 0 d1 2\nq01 0 d1 3\n")
    with pytest.raises(SchemaError, match="duplicate"):
        io.read_qrels(p)


def test_write_round_trip(tmp_path, fixtures):
    qrels = io.read_qrels(fixtures / "sample_qrels.txt")
    assert io.read_qrels(io.write_qrels(tmp_path / "q.txt", qrels)) == qrels
    with pytest.raises(SchemaError):
        io.write_qrels(tmp_path / "bad.txt", [Qrel("q01", "d1", 5)])
