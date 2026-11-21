"""Phase 5 experiment test — runs only when the other parts are present.

Needs Dhrithi's index, Rishit's ranker and metrics, so it skips on the Viraja
branch alone and runs once the branches are together (or pulled locally).
"""

import os

import pytest

pytest.importorskip("index.positional")
pytest.importorskip("dhvani.rank.vsm")
pytest.importorskip("dhvani.eval.metrics")

from dhvani.eval.metrics import read_qrels  # noqa: E402

from dhvani.query import experiment as E  # noqa: E402
from dhvani.query.editdist import load_costs  # noqa: E402


def _results():
    idx = E.build_sample_index("none")
    queries = E.load_queries(os.path.join(E.SAMPLE_DIR, "queries.tsv"))
    qrels = read_qrels(os.path.join(E.SAMPLE_DIR, "qrels.txt"))
    costs = load_costs(E.EDIT_COSTS)
    return E.run(idx, queries, qrels, costs)


def test_phonetic_expansion_recovers_hinglish_queries():
    results = _results()
    # Hinglish exact-match finds nothing (Roman vs Devanagari index);
    # phonetic expansion brings it back.
    assert results["hinglish"]["exact"][1] == 0.0
    assert results["hinglish"]["expanded"][1] > 0.5


def test_expansion_does_not_hurt_hindi():
    results = _results()
    assert results["hindi"]["expanded"][1] >= results["hindi"]["exact"][1]


def test_rocchio_runs_and_reports_both_arms():
    idx = E.build_sample_index("none")
    queries = E.load_queries(os.path.join(E.SAMPLE_DIR, "queries.tsv"))
    qrels = read_qrels(os.path.join(E.SAMPLE_DIR, "qrels.txt"))
    costs = load_costs(E.EDIT_COSTS)
    roc = E.run_rocchio(idx, queries, qrels, costs)
    assert set(roc) == {"no_rocchio", "rocchio"}
    # Rocchio should not collapse retrieval; on this sample it helps a little.
    assert roc["rocchio"][1] >= roc["no_rocchio"][1]
