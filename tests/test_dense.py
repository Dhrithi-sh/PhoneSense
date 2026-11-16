"""Dense retriever: caching, determinism and the Retriever contract (no model download)."""

import numpy as np
import pytest

from phonesense.io import read_phones, read_queries
from phonesense.retrieval.dense import (
    DenseRetriever,
    HashingEncoder,
    collection_key,
)


class CountingEncoder(HashingEncoder):
    """HashingEncoder that counts how many texts it was asked to embed."""

    def __init__(self) -> None:
        super().__init__(dim=128)
        self.calls = 0

    def encode(self, texts):
        self.calls += len(texts)
        return super().encode(texts)


@pytest.fixture
def phones(fixtures):
    return read_phones(fixtures / "phones.jsonl")


@pytest.fixture
def query(fixtures):
    return read_queries(fixtures / "queries.jsonl")[1]  # "compact phone with great camera"


def test_vectors_are_unit_length(phones):
    mat = HashingEncoder().encode([p.text for p in phones])
    assert mat.shape[0] == len(phones)
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)


def test_search_is_deterministic_sorted_and_unique(phones, query):
    r = DenseRetriever(HashingEncoder(), cache_dir=None)
    r.fit(phones)
    a, b = r.search(query, 100), r.search(query, 100)
    assert a == b
    assert len(a) == len({d for d, _ in a}) == len(phones)
    scores = [s for _, s in a]
    assert scores == sorted(scores, reverse=True)
    assert {d for d, _ in r.search(query, 2)} <= {p.doc_id for p in phones}
    assert len(r.search(query, 2)) == 2


def test_ties_break_by_doc_id(phones):
    r = DenseRetriever(HashingEncoder(), cache_dir=None)
    r.fit(phones)
    ranking = r.score_text("zzzz qqqq")  # shares no token with any phone: all scores equal 0
    assert len({s for _, s in ranking}) == 1
    assert [d for d, _ in ranking] == sorted(d for d, _ in ranking)


def test_matching_text_ranks_first(phones):
    r = DenseRetriever(HashingEncoder(), cache_dir=None)
    r.fit(phones)
    target = phones[1]
    assert r.score_text(target.text, 1)[0][0] == target.doc_id


def test_cache_hit_skips_encoding(tmp_path, phones, query):
    first = DenseRetriever(CountingEncoder(), cache_dir=tmp_path)
    first.fit(phones)
    assert not first.cache_hit and first.encoder.calls == len(phones)
    expected = first.search(query, 10)

    second = DenseRetriever(CountingEncoder(), cache_dir=tmp_path)
    second.fit(list(reversed(phones)))  # input order must not matter
    assert second.cache_hit
    assert second.encoder.calls == 0
    assert second.search(query, 10) == expected  # only the query itself is encoded


def test_cache_is_rebuilt_when_a_text_changes(tmp_path, phones):
    DenseRetriever(CountingEncoder(), cache_dir=tmp_path).fit(phones)
    changed = list(phones)
    changed[0] = type(phones[0])(**{**phones[0].__dict__, "text": phones[0].text + " extra"})
    again = DenseRetriever(CountingEncoder(), cache_dir=tmp_path)
    again.fit(changed)
    assert not again.cache_hit


def test_collection_key_depends_on_model_and_text():
    base = collection_key("m", ["a", "b"], ["x", "y"])
    assert base == collection_key("m", ["a", "b"], ["x", "y"])
    assert base != collection_key("m2", ["a", "b"], ["x", "y"])
    assert base != collection_key("m", ["a", "b"], ["x", "z"])


def test_search_before_fit_raises(query):
    with pytest.raises(RuntimeError):
        DenseRetriever(HashingEncoder(), cache_dir=None).search(query)
