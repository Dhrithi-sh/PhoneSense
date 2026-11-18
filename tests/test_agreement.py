"""Kappa and qrels merging: values checked by hand, then the kappa script on small files."""

import importlib.util
import json
from pathlib import Path

import pytest

from phonesense.io import read_qrels, write_qrels
from phonesense.llm.agreement import (
    agreement_report,
    binarise,
    cohen_kappa,
    merge_qrels,
    paired_grades,
    weighted_kappa,
)

LLM = {"q01": {"a": 3, "b": 2, "c": 0}, "q02": {"d": 1, "e": 3}}
HUMAN = {"q01": {"a": 3, "b": 1, "x": 2}, "q02": {"d": 1}, "q03": {"z": 2}}


def test_kappa_by_hand():
    """a = 1,1,0,0 and b = 1,0,0,0: p_o = 3/4; p_e = (2*1 + 2*3) / 16 = 1/2; kappa = 0.5."""
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0], labels=(0, 1)) == pytest.approx(0.5)


def test_kappa_extremes():
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert cohen_kappa([2, 2, 2], [2, 2, 2]) == 1.0  # one shared grade: 0/0, defined as perfect
    assert cohen_kappa([0, 1], [1, 0], labels=(0, 1)) == pytest.approx(-1.0)
    assert cohen_kappa([1, 1, 1], [1, 1, 0], labels=(0, 1)) == pytest.approx(0.0)  # chance level


def test_weighted_kappa_penalises_far_misses_more():
    a = [0, 1, 2, 3, 2, 1]
    near = [0, 1, 2, 2, 2, 1]  # one 3 -> 2 slip
    far = [0, 1, 2, 0, 2, 1]  # one 3 -> 0 slip
    assert weighted_kappa(a, near) > weighted_kappa(a, far)
    assert cohen_kappa(a, near) == pytest.approx(cohen_kappa(a, far), abs=0.15)
    assert weighted_kappa([0, 3], [3, 0]) == pytest.approx(-1.0)


def test_kappa_rejects_empty_and_mismatched_input():
    with pytest.raises(ValueError):
        cohen_kappa([], [])
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 2])


def test_binarise_uses_the_relevance_threshold():
    assert binarise([0, 1, 2, 3]) == [0, 0, 1, 1]
    assert binarise([0, 1, 2, 3], threshold=1) == [0, 1, 1, 1]


def test_paired_grades_only_keeps_doubly_judged_pairs():
    assert paired_grades(LLM, HUMAN) == [("q01", "a", 3, 3), ("q01", "b", 2, 1), ("q02", "d", 1, 1)]
    assert paired_grades(LLM, HUMAN, qids=["q02"]) == [("q02", "d", 1, 1)]


def test_report_counts_and_confusion():
    r = agreement_report(LLM, HUMAN)
    assert r["n_pairs"] == 3 and r["n_queries"] == 2
    assert r["exact_agreement"] == pytest.approx(2 / 3)
    conf = r["confusion_llm_rows_human_cols"]
    assert conf[3][3] == 1 and conf[1][1] == 1 and conf[2][1] == 1 and sum(map(sum, conf)) == 3
    with pytest.raises(ValueError):
        agreement_report(LLM, {"q09": {"z": 1}})


def test_merge_prefers_human_and_never_invents_pairs():
    merged = merge_qrels(LLM, HUMAN)
    assert merged["q01"] == {"a": 3, "b": 1, "c": 0, "x": 2}  # b replaced; x is human-only
    assert merged["q02"] == {"d": 1, "e": 3} and merged["q03"] == {"z": 2}
    assert LLM["q01"]["b"] == 2  # inputs untouched


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "kappa.py"
    spec = importlib.util.spec_from_file_location("kappa_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_writes_report_and_merged_qrels(tmp_path, capsys):
    write_qrels(tmp_path / "llm.txt", LLM)
    write_qrels(tmp_path / "human.txt", HUMAN)
    argv = ["--llm", str(tmp_path / "llm.txt"), "--human", str(tmp_path / "human.txt"),
            "--out", str(tmp_path / "kappa.json"), "--merge",
            "--merged-out", str(tmp_path / "qrels.txt")]
    assert _load_script().main(argv) == 0
    report = json.loads((tmp_path / "kappa.json").read_text())
    assert report["n_pairs"] == 3 and report["split"] == "all"
    assert read_qrels(tmp_path / "qrels.txt")["q01"]["b"] == 1
    assert "1 LLM grades replaced" in capsys.readouterr().out


def test_script_reports_missing_files(tmp_path):
    argv = ["--llm", str(tmp_path / "no.txt"), "--human", str(tmp_path / "x")]
    assert _load_script().main(argv) == 1
