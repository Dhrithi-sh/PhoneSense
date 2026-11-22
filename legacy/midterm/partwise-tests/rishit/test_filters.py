from dhvani.rank.bm25 import search_bm25
from dhvani.rank.filters import field_values, make_filter
from dhvani.rank.query_stub import exact_query
from dhvani.rank.sample_index import SampleIndex
from dhvani.rank.scoring import rank
from dhvani.rank.vsm import search


def test_no_filters_means_no_filter():
    assert make_filter(SampleIndex.load()) is None


def test_source_filter():
    idx = SampleIndex.load()
    f = make_filter(idx, sources=["jagran"])
    results = search(exact_query("दिल्ली बारिश"), idx, k=10, doc_filter=f)
    assert results and all(idx.meta[d]["source"] == "jagran" for d, _, _ in results)


def test_filters_combine_as_and():
    idx = SampleIndex.load()
    f = make_filter(idx, sections=["weather"], states=["delhi"])
    results = rank(exact_query("मौसम बारिश"), idx, k=10, doc_filter=f)
    ids = {d for d, _, _ in results}
    assert ids == {"jagran_1001", "nbt_2001"}


def test_date_range():
    idx = SampleIndex.load()
    f = make_filter(idx, date_from="2026-10-05", date_to="2026-10-05")
    results = search_bm25(exact_query("बारिश मौसम"), idx, k=10, doc_filter=f)
    assert results and all(idx.meta[d]["date"].startswith("2026-10-05") for d, _, _ in results)


def test_filter_applies_before_top_k():
    # With k=1 the best Lucknow article should still come back, even though
    # Delhi articles outscore it overall.
    idx = SampleIndex.load()
    f = make_filter(idx, states=["uttar-pradesh"])
    top = search(exact_query("बारिश"), idx, k=1, doc_filter=f)
    assert top[0][0] == "amarujala_3001"


def test_field_values_for_dropdowns():
    idx = SampleIndex.load()
    assert field_values(idx, "source") == ["aajtak", "amarujala", "jagran", "livehindustan", "nbt"]
    assert "" not in field_values(idx, "state")
