"""The real queries file (queries/queries.jsonl) obeys CONTRACT.md 4.3 and the agreed mix."""

from collections import Counter

import pytest

from phonesense.config import get_path
from phonesense.io import read_queries

EXPECTED_TYPES = {"spec": 12, "aspect": 12, "model": 11, "vague": 10}
EXPECTED_TEST = {"spec": 5, "aspect": 5, "model": 4, "vague": 4}


def _type(q) -> str:
    assert q.intent_notes and q.intent_notes.startswith("type="), q.qid
    return q.intent_notes.split(".", 1)[0].removeprefix("type=")


@pytest.fixture(scope="module")
def queries():
    return read_queries(get_path("queries"))


def test_45_queries_with_sequential_ids(queries):
    assert [q.qid for q in queries] == [f"q{i:02d}" for i in range(1, 46)]


def test_type_mix(queries):
    assert Counter(_type(q) for q in queries) == EXPECTED_TYPES


def test_split_is_stratified_by_type(queries):
    test = Counter(_type(q) for q in queries if q.split == "test")
    assert test == EXPECTED_TEST
    assert len(read_queries(get_path("queries"), split="dev")) == 27


def test_only_spec_queries_have_filters(queries):
    for q in queries:
        assert (q.filters is not None) == (_type(q) == "spec"), q.qid


def test_texts_are_unique(queries):
    texts = [q.text.casefold() for q in queries]
    assert len(set(texts)) == len(texts)
