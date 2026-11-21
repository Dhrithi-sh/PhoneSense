import pytest

from index.positional import Index
from index.query import search_both, search_query


def build_test_indexes():
    none = Index("none")
    light = Index("light")

    documents = [
        (
            "doc1",
            "दिल्ली बारिश",
            "आज दिल्ली में बारिश हुई",
        ),
        (
            "doc2",
            "दिल्ली मौसम",
            "दिल्ली में आज मौसम साफ है",
        ),
        (
            "doc3",
            "बारिश खबर",
            "मुंबई में बारिश हुई",
        ),
    ]

    for doc_id, headline, body in documents:
        metadata = {
            "source": "test",
            "date": "2026-10-06T00:00:00+05:30",
            "state": "test",
            "section": "weather",
            "dup_of": None,
        }

        none.add_document(
            doc_id,
            headline,
            body,
            metadata,
        )

        light.add_document(
            doc_id,
            headline,
            body,
            metadata,
        )

    return none, light


def test_search_query_none_mode():
    none, _ = build_test_indexes()

    results = search_query(
        none,
        "दिल्ली",
        mode="none",
    )

    assert set(results) == {"doc1", "doc2"}


def test_search_query_light_mode():
    _, light = build_test_indexes()

    results = search_query(
        light,
        "दिल्ली",
        mode="light",
    )

    assert set(results) == {"doc1", "doc2"}


def test_search_query_uses_multiple_terms():
    none, _ = build_test_indexes()

    results = search_query(
        none,
        "दिल्ली बारिश",
        mode="none",
    )

    assert results == ["doc1"]


def test_search_query_respects_zone():
    none, _ = build_test_indexes()

    results = search_query(
        none,
        "मौसम",
        mode="none",
        zone="headline",
    )

    assert results == ["doc2"]


def test_search_query_empty_query():
    none, _ = build_test_indexes()

    assert search_query(
        none,
        "",
        mode="none",
    ) == []


def test_search_query_invalid_mode():
    none, _ = build_test_indexes()

    with pytest.raises(
        ValueError,
        match="Unsupported analysis mode",
    ):
        search_query(
            none,
            "दिल्ली",
            mode="invalid",
        )


def test_search_query_mode_must_match_index():
    none, _ = build_test_indexes()

    with pytest.raises(
        ValueError,
        match="does not match index mode",
    ):
        search_query(
            none,
            "दिल्ली",
            mode="light",
        )


def test_search_query_invalid_zone():
    none, _ = build_test_indexes()

    with pytest.raises(
        ValueError,
        match="Unsupported zone",
    ):
        search_query(
            none,
            "दिल्ली",
            mode="none",
            zone="title",
        )


def test_search_both():
    none, light = build_test_indexes()

    results = search_both(
        none,
        light,
        "दिल्ली",
    )

    assert set(results.keys()) == {"none", "light"}

    assert set(results["none"]) == {"doc1", "doc2"}
    assert set(results["light"]) == {"doc1", "doc2"}