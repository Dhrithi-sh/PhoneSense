import math

from index.positional import Index
from index.scoring import idf, idf_table


def build_test_index():
    index = Index("none")

    index.add_document(
        "doc1",
        "बारिश",
        "बारिश मौसम",
        {
            "source": "test",
            "date": None,
            "state": None,
            "section": None,
            "dup_of": None,
        },
    )

    index.add_document(
        "doc2",
        "मौसम",
        "मौसम अच्छा",
        {
            "source": "test",
            "date": None,
            "state": None,
            "section": None,
            "dup_of": None,
        },
    )

    index.add_document(
        "doc3",
        "बारिश",
        "बारिश तेज",
        {
            "source": "test",
            "date": None,
            "state": None,
            "section": None,
            "dup_of": None,
        },
    )

    return index


def test_idf_for_term_present_in_all_documents():
    index = build_test_index()

    # मौसम appears in doc1 and doc2.
    # N = 3, df = 2.
    expected = math.log10(3 / 2)

    assert math.isclose(idf(index, "मौसम"), expected)


def test_idf_for_term_present_in_two_documents():
    index = build_test_index()

    # बारिश appears in doc1 and doc3.
    # N = 3, df = 2.
    expected = math.log10(3 / 2)

    assert math.isclose(idf(index, "बारिश"), expected)


def test_idf_for_missing_term():
    index = build_test_index()

    assert idf(index, "दिल्ली") == 0.0


def test_idf_table():
    index = build_test_index()

    result = idf_table(
        index,
        ["बारिश", "मौसम", "दिल्ली"],
    )

    assert math.isclose(result["बारिश"], math.log10(3 / 2))
    assert math.isclose(result["मौसम"], math.log10(3 / 2))
    assert result["दिल्ली"] == 0.0


def test_idf_for_term_in_one_document():
    index = build_test_index()

    # अच्छा appears only in doc2.
    # N = 3, df = 1.
    expected = math.log10(3)

    assert math.isclose(idf(index, "अच्छा"), expected)