"""End-to-end: a Viraja query object scored by Rishit's ranker.

Rishit's ``dhvani/rank/`` lives on his branch, not on mine, so this test skips
cleanly when it isn't importable (anyone on the Viraja branch) and runs once the
two parts are together (after a merge, or when his rank is pulled locally).
"""

import os

import pytest

vsm = pytest.importorskip("dhvani.rank.vsm")
from dhvani.rank.sample_index import SampleIndex  # noqa: E402

from dhvani.query.build import build_query  # noqa: E402
from dhvani.query.editdist import load_costs  # noqa: E402
from dhvani.query.kgram import KGramIndex  # noqa: E402

_COSTS = os.path.join(os.path.dirname(__file__), "..", "..", "dhvani", "query", "edit_costs.json")


def test_hindi_query_scores_through_the_ranker():
    idx = SampleIndex.load()
    results = vsm.search(build_query("मौसम"), idx)
    assert results, "ranker returned no documents for मौसम"
    top_sections = [idx.meta[doc_id]["section"] for doc_id, _s, _e in results[:3]]
    assert "weather" in top_sections


def test_hinglish_query_reaches_hindi_docs_via_phonetic_expansion():
    idx = SampleIndex.load()
    kg = KGramIndex(idx.vocab, k=2)
    costs = load_costs(_COSTS)
    # "mosam" has no exact match in the index; only the phonetic मौसम does.
    query = build_query("mosam", index=kg, costs=costs)
    results = vsm.search(query, idx)
    assert results, "phonetic expansion failed to reach any document"
    top_sections = [idx.meta[doc_id]["section"] for doc_id, _s, _e in results[:3]]
    assert "weather" in top_sections
