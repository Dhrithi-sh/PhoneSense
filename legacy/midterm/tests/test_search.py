from index.positional import Index

from index.search import (
    and_search,
    build_skip_pointers,
    intersect_postings,
    phrase_search,
    proximity_search,
)


def build_test_index():
    idx = Index("none")

    idx.add_document(
        "doc1",
        "दिल्ली में बारिश",
        "आज दिल्ली में बारिश हुई",
        {},
    )

    idx.add_document(
        "doc2",
        "दिल्ली मौसम अपडेट",
        "दिल्ली में आज तेज बारिश हुई",
        {},
    )

    idx.add_document(
        "doc3",
        "बारिश की खबर",
        "मुंबई में बारिश हुई",
        {},
    )

    return idx


def test_build_skip_pointers():
    postings = [
        ("doc1", 1, [0]),
        ("doc2", 1, [0]),
        ("doc3", 1, [0]),
        ("doc4", 1, [0]),
        ("doc5", 1, [0]),
        ("doc6", 1, [0]),
        ("doc7", 1, [0]),
        ("doc8", 1, [0]),
        ("doc9", 1, [0]),
    ]

    skips = build_skip_pointers(postings)

    assert skips == {
        0: 3,
        3: 6,
    }


def test_single_item_has_no_skip_pointer():
    postings = [
        ("doc1", 1, [0]),
    ]

    assert build_skip_pointers(postings) == {}


def test_intersect_postings():
    left = [
        ("doc1", 1, [0]),
        ("doc2", 1, [0]),
        ("doc3", 1, [0]),
        ("doc4", 1, [0]),
        ("doc5", 1, [0]),
    ]

    right = [
        ("doc2", 1, [0]),
        ("doc4", 1, [0]),
        ("doc6", 1, [0]),
    ]

    result = intersect_postings(left, right)

    assert result == [
        "doc2",
        "doc4",
    ]


def test_intersect_postings_with_no_overlap():
    left = [
        ("doc1", 1, [0]),
        ("doc2", 1, [0]),
    ]

    right = [
        ("doc3", 1, [0]),
        ("doc4", 1, [0]),
    ]

    assert intersect_postings(left, right) == []


def test_and_search():
    idx = build_test_index()

    result = and_search(
        idx,
        ["दिल्ली", "बारिश"],
        "body",
    )

    assert result == [
        "doc1",
        "doc2",
    ]


def test_and_search_returns_empty_when_term_missing():
    idx = build_test_index()

    result = and_search(
        idx,
        ["दिल्ली", "मुंबई"],
        "body",
    )

    assert result == []


def test_and_search_single_term():
    idx = build_test_index()

    result = and_search(
        idx,
        ["बारिश"],
        "body",
    )

    assert result == [
        "doc1",
        "doc2",
        "doc3",
    ]


def test_phrase_search():
    idx = build_test_index()

    result = phrase_search(
        idx,
        ["बारिश", "हुई"],
        "body",
    )

    assert result == [
        "doc1",
        "doc2",
        "doc3",
    ]


def test_phrase_search_respects_order():
    idx = build_test_index()

    result = phrase_search(
        idx,
        ["दिल्ली", "में", "बारिश"],
        "body",
    )

    assert result == [
        "doc1",
    ]


def test_phrase_search_does_not_match_non_consecutive_terms():
    idx = build_test_index()

    result = phrase_search(
        idx,
        ["दिल्ली", "बारिश"],
        "body",
    )

    assert result == []


def test_phrase_search_respects_zone():
    idx = build_test_index()

    result = phrase_search(
        idx,
        ["दिल्ली", "में"],
        "headline",
    )

    assert result == [
        "doc1",
    ]


def test_proximity_search_matches_nearby_terms():
    idx = build_test_index()

    result = proximity_search(
        idx,
        ["दिल्ली", "बारिश"],
        3,
        "body",
    )

    assert result == [
        "doc1",
    ]


def test_proximity_search_does_not_require_order():
    idx = build_test_index()

    result = proximity_search(
        idx,
        ["बारिश", "दिल्ली"],
        3,
        "body",
    )

    assert result == [
        "doc1",
    ]


def test_proximity_search_respects_distance():
    idx = build_test_index()

    result = proximity_search(
        idx,
        ["दिल्ली", "बारिश"],
        1,
        "body",
    )

    assert result == []


def test_proximity_search_can_match_exact_distance():
    idx = build_test_index()

    result = proximity_search(
        idx,
        ["दिल्ली", "बारिश"],
        4,
        "body",
    )

    assert result == [
        "doc1",
        "doc2",
    ]


def test_proximity_search_returns_empty_when_term_missing():
    idx = build_test_index()

    result = proximity_search(
        idx,
        ["दिल्ली", "मुंबई"],
        3,
        "body",
    )

    assert result == []


def test_proximity_search_empty_terms():
    idx = build_test_index()

    assert proximity_search(
        idx,
        [],
        3,
    ) == []


def test_proximity_search_rejects_negative_distance():
    idx = build_test_index()

    try:
        proximity_search(
            idx,
            ["दिल्ली", "बारिश"],
            -1,
        )
        assert False
    except ValueError:
        assert True