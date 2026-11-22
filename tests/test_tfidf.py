import math
import random

import numpy as np
import pytest

from phonesense import io
from phonesense.index import Analyzer, InvertedIndex
from phonesense.retrieval.base import run_retriever
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Query, validate_run


def q(text: str) -> Query:
    return Query(qid="q01", text=text, split="dev")


@pytest.fixture
def three_docs(make_phone):
    return [
        make_phone("d1", "camera camera phone"),
        make_phone("d2", "phone battery"),
        make_phone("d3", "battery battery battery"),
    ]


def test_hand_computed_scores(three_docs):
    """The worked example from the tfidf.py docstring, computed by hand.

    N = 3, df(camera) = 1, df(phone) = 2, df(battery) = 2. Query "camera phone":
      q:  camera log10(3) = 0.4771213, phone log10(1.5) = 0.1760913, norm 0.5085792
          -> 0.9381454, 0.3462416
      d1: camera 1 + log10(2) = 1.3010300, phone 1; norm 1.6409385 -> 0.7928573, 0.6094074
      d2: phone 1, battery 1 -> 1/sqrt(2) each
      score(d1) = 0.9381454 * 0.7928573 + 0.3462416 * 0.6094074 = 0.9548176
      score(d2) = 0.3462416 * 0.7071068                         = 0.2448298
    """
    r = TfidfRetriever(analyzer=Analyzer(stem=False))
    r.fit(three_docs)
    res = r.search(q("camera phone"), k=10)
    assert [d for d, _ in res] == ["d1", "d2"]
    assert res[0][1] == pytest.approx(0.9548175559, abs=1e-6)
    assert res[1][1] == pytest.approx(0.2448297501, abs=1e-6)
    assert r.doc_norm(0) == pytest.approx(1.6409384661, abs=1e-6)
    w = r.query_weights("camera phone")
    assert w["camera"] == pytest.approx(0.9381453975, abs=1e-6)
    assert w["phone"] == pytest.approx(0.3462415531, abs=1e-6)
    # query tf > 1: (1 + log10 2) * log10(3/2) for battery, and a single-term query scores 1
    # for a doc whose only term it is
    assert r.search(q("battery battery"), k=1) == [("d3", pytest.approx(1.0))]


def _dense_lnc_ltc(texts: list[str], query: str) -> np.ndarray:
    """Independent brute-force lnc.ltc with dense numpy vectors (whitespace tokens)."""
    vocab = sorted({t for s in texts for t in s.split()})
    col = {t: i for i, t in enumerate(vocab)}
    tf = np.zeros((len(texts), len(vocab)))
    for i, s in enumerate(texts):
        for t in s.split():
            tf[i, col[t]] += 1
    with np.errstate(divide="ignore"):
        d = np.where(tf > 0, 1 + np.log10(tf), 0.0)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    d = np.divide(d, norms, out=np.zeros_like(d), where=norms > 0)
    n_docs = len(texts)
    df = (tf > 0).sum(axis=0)
    qtf = np.zeros(len(vocab))
    for t in query.split():
        if t in col:
            qtf[col[t]] += 1
    with np.errstate(divide="ignore"):
        qv = np.where(qtf > 0, (1 + np.log10(qtf)) * np.log10(n_docs / np.maximum(df, 1)), 0.0)
    qn = np.linalg.norm(qv)
    return d @ (qv / qn) if qn > 0 else np.zeros(n_docs)


@pytest.mark.parametrize("seed", range(3))
def test_matches_dense_bruteforce(make_phone, seed):
    rng = random.Random(seed)
    words = [f"t{i}" for i in range(25)]  # digits -> never stemmed, no stopwords
    texts = [
        " ".join(rng.choices(words, weights=range(25, 0, -1), k=rng.randint(1, 40)))
        for _ in range(30)
    ]
    phones = [make_phone(f"P{i:02d}", s) for i, s in enumerate(texts)]
    r = TfidfRetriever()
    r.fit(phones)
    for _ in range(10):
        query = " ".join(rng.choices(words + ["zz9"], k=rng.randint(1, 6)))
        expected = _dense_lnc_ltc(texts, query)
        got = dict(r.search(q(query), k=100))
        for i, score in enumerate(expected):
            if score > 0:
                assert got[f"P{i:02d}"] == pytest.approx(score, abs=1e-9)
            else:
                assert f"P{i:02d}" not in got
        ranked = [s for _, s in r.search(q(query), k=100)]
        assert ranked == sorted(ranked, reverse=True)


def test_matches_sklearn_if_available(make_phone):
    """Optional cross-check with scikit-learn (test-only; skipped if not installed)."""
    sk = pytest.importorskip("sklearn.feature_extraction.text")
    rng = random.Random(3)
    words = [f"t{i}" for i in range(15)]
    texts = [" ".join(rng.choices(words, k=rng.randint(2, 25))) for _ in range(30)]
    r = TfidfRetriever()
    r.fit([make_phone(f"P{i:02d}", s) for i, s in enumerate(texts)])
    cv = sk.CountVectorizer(token_pattern=r"\S+", lowercase=False)
    tf = cv.fit_transform(texts).toarray().astype(float)
    with np.errstate(divide="ignore"):
        d = np.where(tf > 0, 1 + np.log10(tf), 0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    query = "t1 t2 t2 t7"
    qtf = cv.transform([query]).toarray()[0].astype(float)
    df = (tf > 0).sum(0)
    with np.errstate(divide="ignore"):
        qv = np.where(qtf > 0, (1 + np.log10(qtf)) * np.log10(30 / df), 0)
    expected = d @ (qv / np.linalg.norm(qv))
    got = dict(r.search(q(query), k=100))
    for i, s in enumerate(expected):
        assert got.get(f"P{i:02d}", 0.0) == pytest.approx(s, abs=1e-9)


def test_ties_broken_by_doc_id(make_phone):
    phones = [make_phone(d, "camera phone") for d in ("Z9", "B2", "M5", "A1")]
    phones.append(make_phone("X0", "battery"))
    r = TfidfRetriever()
    r.fit(phones)
    res = r.search(q("camera"), k=10)
    assert [d for d, _ in res] == ["A1", "B2", "M5", "Z9"]
    assert len({s for _, s in res}) == 1
    assert r.search(q("camera"), k=2) == res[:2]
    assert r.search(q("camera"), k=10) == res  # deterministic across calls


def test_oov_and_empty_queries(three_docs):
    r = TfidfRetriever(analyzer=Analyzer(stem=False))
    r.fit(three_docs)
    assert r.search(q("zzzz qqqq")) == []
    assert r.search(q("the with of")) == []  # stopwords only
    assert r.search(q("!!!")) == []
    assert r.search(q("zzzz camera"), k=10)[0][0] == "d1"  # OOV term skipped


def test_term_in_every_doc_has_zero_idf(make_phone):
    r = TfidfRetriever()
    r.fit([make_phone("A", "phone camera"), make_phone("B", "phone battery")])
    assert r.search(q("phone")) == []
    assert [d for d, _ in r.search(q("phone camera"))] == ["A"]


def test_requires_fit():
    with pytest.raises(RuntimeError):
        TfidfRetriever().search(q("x"))


def test_field_and_shared_index(three_docs):
    idx = InvertedIndex.build(three_docs, Analyzer(stem=False))
    r = TfidfRetriever(field="title", index=idx)
    assert r.name == "tfidf_lnc.ltc.title"
    assert TfidfRetriever(field="text").name == "tfidf_lnc.ltc"
    assert r.search(q("camera phone"))[0][0] == "d1"
    with pytest.raises(ValueError):
        TfidfRetriever(
            field="title", index=InvertedIndex.build(three_docs, Analyzer(), fields=("text",))
        )


def test_run_file_validates(tmp_path, fixtures):
    phones = io.read_phones(fixtures / "phones.jsonl")
    queries = io.read_queries(fixtures / "queries.jsonl")
    r = TfidfRetriever()
    r.fit(phones)
    path = run_retriever(r, queries, k=100, runs_dir=tmp_path)
    assert path.name == "tfidf_lnc.ltc_all.run"
    entries = io.read_run_entries(path)  # validates format
    validate_run(entries)
    assert entries and all(math.isfinite(e.score) for e in entries)
    dev = run_retriever(
        r, io.read_queries(fixtures / "queries.jsonl", split="dev"), runs_dir=tmp_path
    )
    assert set(io.read_run(dev)) <= {"q01", "q02"}
