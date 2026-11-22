"""scripts/label_cli.py: answering, invalid keys, undo, skip, quit and resume (scripted input)."""

import importlib.util
from pathlib import Path

import pytest

from phonesense.io import read_phones, read_queries
from phonesense.llm.label_sheet import read_sheet, write_sheet


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "label_cli.py"
    spec = importlib.util.spec_from_file_location("label_cli_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def sheet(tmp_path, fixtures):
    queries = {q.qid: q for q in read_queries(fixtures / "queries.jsonl")}
    phones = {p.doc_id: p for p in read_phones(fixtures / "phones.jsonl")}
    pairs = [("q01", "B0TEST0001"), ("q02", "B0TEST0002"), ("q03", "B0TEST0003")]
    return write_sheet(tmp_path / "s.csv", pairs, queries, phones)


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def grades(path):
    script = _load_script()
    return [r["grade"] for r in script.load(path)[1]]


def test_answers_invalid_keys_undo_and_skip(sheet):
    script = _load_script()
    shown = []
    # 2 -> pair 1; "x" is rejected; 3 -> pair 2; u undoes it; 1 -> pair 2 again; s skips pair 3
    n = script.run(sheet, scripted(["2", "x", "3", "u", "1", "s"]), shown.append)
    text = "\n".join(shown)
    assert n == 2 and grades(sheet) == ["2", "1", ""]
    assert "please type 0, 1, 2, 3, s, u or q" in text and "skipped pair(s) are still blank" in text
    assert read_sheet(sheet) == {"q01": {"B0TEST0001": 2}, "q02": {"B0TEST0002": 1}}


def test_the_display_shows_the_search_and_phone_but_not_the_rubric_cell_or_any_llm_grade(sheet):
    script = _load_script()
    shown = []
    script.run(sheet, scripted(["q"]), shown.append)
    text = "\n".join(shown)
    assert "SEARCH: phone under $300 with good battery and 5G" in text
    assert "Nova X5 5G Smartphone 128GB" in text
    assert "llm" not in text.lower().split("keys:")[1]  # nothing about LLM grades after the header


def test_quit_leaves_the_sheet_unchanged_and_resume_continues_where_it_stopped(sheet):
    script = _load_script()
    assert script.run(sheet, scripted(["q"]), lambda s: None) == 0
    assert grades(sheet) == ["", "", ""]
    script.run(sheet, scripted(["3", "q"]), lambda s: None)
    assert grades(sheet) == ["3", "", ""]
    shown = []
    script.run(sheet, scripted(["0", "2"]), shown.append)  # resumes with pair 2
    assert grades(sheet) == ["3", "0", "2"]
    assert "All done" in "\n".join(shown)
    assert script.run(sheet, scripted([]), lambda s: None) == 0  # nothing left: no prompt at all


def test_undo_with_nothing_to_undo_is_harmless(sheet):
    script = _load_script()
    shown = []
    script.run(sheet, scripted(["u", "q"]), shown.append)
    assert "nothing to undo" in "\n".join(shown) and grades(sheet) == ["", "", ""]


def test_saves_are_atomic_and_leave_no_temp_file(sheet):
    script = _load_script()
    script.run(sheet, scripted(["1", "q"]), lambda s: None)
    assert not sheet.with_suffix(".csv.tmp").exists()


def test_main_reports_a_missing_file_and_a_bad_sheet(tmp_path, capsys):
    script = _load_script()
    assert script.main([str(tmp_path / "nope.csv")]) == 1
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    assert script.main([str(bad)]) == 1
    assert "missing column" in capsys.readouterr().err
