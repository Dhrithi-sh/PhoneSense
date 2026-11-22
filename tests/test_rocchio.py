"""Rocchio: the update formula by hand, then pseudo-feedback on a four-document collection."""

import pytest

from phonesense.index import Analyzer
from phonesense.retrieval.rocchio import (
    RocchioRetriever,
    centroid,
    rocchio_update,
    top_terms,
)
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Query


def q(text: str) -> Query:
    return Query(qid="q01", text=text, split="dev")


def test_centroid_is_the_componentwise_mean():
    assert centroid([{"a": 1.0, "b": 2.0}, {"b": 4.0}]) == {"a": 0.5, "b": 3.0}
    assert centroid([]) == {}


def test_update_hand_computed():
    """alpha=1, beta=0.75, gamma=0.15 on q={a:1}, rel={a:1,b:2},{b:4}, non={a:2}.

    centroid(rel) = {a: 0.5, b: 3},  centroid(non) = {a: 2}
      a: 1*1   + 0.75*0.5 - 0.15*2 = 1 + 0.375 - 0.3 = 1.075
      b: 1*0   + 0.75*3   - 0.15*0 = 2.25
    """
    out = rocchio_update(
        {"a": 1.0}, [{"a": 1, "b": 2}, {"b": 4}], [{"a": 2}], alpha=1, beta=0.75, gamma=0.15
    )
    assert out == {"a": pytest.approx(1.075), "b": pytest.approx(2.25)}


def test_update_drops_non_positive_components():
    # c: 0 + 0.75*0 - 0.15*4 = -0.6 (dropped);  a: 1 - 0.15*4 = 0.4 (kept)
    out = rocchio_update({"a": 1.0}, [{}], [{"a": 4.0, "c": 4.0}], 1, 0.75, 0.15)
    assert out == {"a": pytest.approx(0.4)}


def test_alpha_only_leaves_the_query_unchanged():
    assert rocchio_update({"a": 0.6, "b": 0.8}, [{"x": 5.0}], [{"y": 5.0}], 1, 0, 0) == {
        "a": 0.6,
        "b": 0.8,
    }


def test_top_terms_keeps_heaviest_and_breaks_ties_alphabetically():
    vec = {"z": 1.0, "a": 1.0, "m": 3.0, "k": 0.5}
    assert top_terms(vec, 2) == {"m": 3.0, "a": 1.0}
    assert top_terms(vec, 0) == {}


@pytest.fixture
def collection(make_phone):
    return [
        make_phone("d1", "camera lens zoom"),
        make_phone("d2", "camera lens zoom"),
        make_phone("d3", "lens zoom macro"),  # no "camera": only feedback can surface it
        make_phone("d4", "battery charger"),
    ]


def fitted(collection, **kw):
    r = RocchioRetriever(TfidfRetriever(analyzer=Analyzer(stem=False)), **kw)
    r.fit(collection)
    return r


def test_feedback_surfaces_a_document_without_the_query_term(collection):
    base = TfidfRetriever(analyzer=Analyzer(stem=False))
    base.fit(collection)
    assert [d for d, _ in base.search(q("camera"), 10)] == ["d1", "d2"]

    r = fitted(collection, m=2)
    ranked = [d for d, _ in r.search(q("camera"), 10)]
    assert ranked[:2] == ["d1", "d2"]
    assert "d3" in ranked and "d4" not in ranked


def test_expanded_query_contains_feedback_terms(collection):
    r = fitted(collection, m=2)
    expanded = r.expanded_query("camera")
    assert set(expanded) == {"camera", "lens", "zoom"}
    assert expanded["camera"] > expanded["lens"] > 0


def test_n_terms_caps_the_expansion(collection):
    assert len(fitted(collection, m=2, n_terms=1).expanded_query("camera")) == 1


def test_beta_zero_reproduces_plain_tfidf(collection):
    base = TfidfRetriever(analyzer=Analyzer(stem=False))
    base.fit(collection)
    r = fitted(collection, m=2, beta=0.0)
    got, want = r.search(q("camera"), 10), base.search(q("camera"), 10)
    assert [d for d, _ in got] == [d for d, _ in want]
    assert [s for _, s in got] == pytest.approx([s for _, s in want])


def test_unknown_query_returns_nothing(collection):
    assert fitted(collection).search(q("hovercraft"), 10) == []


def test_deterministic_and_sorted(collection):
    r = fitted(collection, m=2)
    a, b = r.search(q("camera"), 10), r.search(q("camera"), 10)
    assert a == b
    assert [s for _, s in a] == sorted((s for _, s in a), reverse=True)


def test_bad_parameters_are_rejected():
    with pytest.raises(ValueError):
        RocchioRetriever(m=0)
    with pytest.raises(ValueError):
        RocchioRetriever(beta=-1.0)
