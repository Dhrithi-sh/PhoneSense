import pytest

from phonesense import io
from phonesense.retrieval.base import Retriever, run_retriever
from phonesense.schemas import Phone, Query


class WordOverlap(Retriever):
    """Trivial retriever: score = number of query words found in the phone text."""

    name = "dummy"

    def fit(self, phones: list[Phone]) -> None:
        self.docs = {p.doc_id: set(p.text.lower().split()) for p in phones}

    def search(self, query: Query, k: int = 100):
        words = set(query.text.lower().split())
        scored = [(d, float(len(words & toks))) for d, toks in self.docs.items()]
        return sorted(scored, key=lambda x: -x[1])[:k]


@pytest.fixture
def retriever(fixtures):
    r = WordOverlap()
    r.fit(io.read_phones(fixtures / "phones.jsonl"))
    return r


def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        Retriever()

    class OnlyFit(Retriever):
        def fit(self, phones):
            pass

    class OnlySearch(Retriever):
        def search(self, query, k=100):
            return []

    for cls in (OnlyFit, OnlySearch):
        with pytest.raises(TypeError):
            cls()


def test_run_retriever_writes_valid_run(tmp_path, fixtures, retriever):
    queries = io.read_queries(fixtures / "queries.jsonl", split="dev")
    path = run_retriever(retriever, queries, k=2, runs_dir=tmp_path)
    assert path == tmp_path / "dummy_dev.run"
    run = io.read_run(path)
    assert set(run) == {"q01", "q02"} and all(len(v) == 2 for v in run.values())


def test_run_retriever_split_checks(tmp_path, fixtures, retriever):
    queries = io.read_queries(fixtures / "queries.jsonl")
    assert run_retriever(retriever, queries, runs_dir=tmp_path).name == "dummy_all.run"
    with pytest.raises(ValueError, match="test"):
        run_retriever(retriever, queries, run_name="dummy_test", runs_dir=tmp_path)
    with pytest.raises(ValueError):
        run_retriever(retriever, queries, k=101, runs_dir=tmp_path)
