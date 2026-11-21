r"""TF-IDF retrieval with the SMART lnc.ltc scheme (T04).

Manning, Raghavan & Schuetze, *Introduction to Information Retrieval*, ch. 6 (sec. 6.4.3).

Formula. For a term t, document d and query q, with N documents and df_t the number of
documents containing t (in the indexed field, ``text`` by default):

* Document, **lnc**: ``w(t,d) = 1 + log10(tf(t,d))`` (0 if tf = 0), *no* idf, then cosine
  normalization ``w(t,d) / sqrt(sum_t' w(t',d)^2)``. Norms are precomputed in ``fit``.
* Query, **ltc**: ``w(t,q) = (1 + log10(tf(t,q))) * log10(N / df_t)``, then cosine
  normalization. Query terms that are not in the vocabulary are skipped.
* ``score(q, d) = sum_t w(t,q) * w(t,d)`` (cosine similarity of the two unit vectors).

The sum is computed term-at-a-time: for each query term with a non-zero weight we walk its
postings list and add ``w(t,q) * w(t,d)`` to an accumulator per document, so only documents
that share a term with the query are touched (no dense vectors). The top k are taken with a
heap; ties are broken by ``doc_id`` ascending so runs are deterministic.

Worked example (N = 3; the same numbers are checked in ``tests/test_tfidf.py``)::

    d1 = camera camera phone     d2 = phone battery     d3 = battery battery battery
    q  = camera phone            df: camera 1, phone 2, battery 2

    lnc, d1: camera 1 + log10 2 = 1.30103, phone 1.0; norm = sqrt(1.30103^2 + 1) = 1.640938
             -> camera 0.792857, phone 0.609407
    lnc, d2: battery 1, phone 1 -> 0.707107 each;  d3: battery -> 1.0
    ltc, q:  camera 1 * log10(3/1) = 0.477121, phone 1 * log10(3/2) = 0.176091;
             norm = 0.508579 -> camera 0.938145, phone 0.346242
    score(q,d1) = 0.938145 * 0.792857 + 0.346242 * 0.609407 = 0.954818
    score(q,d2) = 0.346242 * 0.707107                         = 0.244830
    score(q,d3) = 0  (no shared term, so d3 is not returned)

The tf weight is ``1 + log10(tf)``, not ``log10(1 + tf)``.

Adapted from our midterm project Dhvani-lingual (``dhvani/rank/vsm.py``): the zone loop,
query-expansion weights and per-term explanations were dropped; it now runs on the
:class:`~phonesense.index.InvertedIndex` and the shared ``Retriever`` API.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from phonesense.index import Analyzer, InvertedIndex
from phonesense.io import Ranking
from phonesense.retrieval.base import Retriever
from phonesense.schemas import Phone, Query


def _log_tf(tf: float) -> float:
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


def _default_field() -> str:
    from phonesense.config import get_config

    return get_config().get("retrieval", {}).get("tfidf_field", "text")


class TfidfRetriever(Retriever):
    """lnc.ltc cosine retriever over one field of the inverted index.

    Use ``fit(phones)`` to build a private index for ``field``, or pass a prebuilt
    (e.g. loaded) ``index`` to reuse it without re-analyzing the collection.
    """

    def __init__(
        self,
        field: str | None = None,
        analyzer: Analyzer | None = None,
        index: InvertedIndex | None = None,
    ) -> None:
        self.field = field or _default_field()
        self.name = "tfidf_lnc.ltc" if self.field == "text" else f"tfidf_lnc.ltc.{self.field}"
        self._analyzer = analyzer
        self.index: InvertedIndex | None = None
        self._norms: list[float] = []
        if index is not None:
            self.fit_index(index)

    # ------------------------------------------------------------------ fitting

    def fit(self, phones: list[Phone]) -> None:
        """Build an index over ``phones`` for ``self.field`` and precompute doc norms."""
        analyzer = self._analyzer if self._analyzer is not None else Analyzer.from_config()
        self.fit_index(InvertedIndex.build(phones, analyzer, fields=(self.field,)))

    def fit_index(self, index: InvertedIndex) -> None:
        """Use an existing index; precompute the lnc norm of every document."""
        if self.field not in index.fields:
            raise ValueError(f"index has no field {self.field!r} (fields: {index.fields})")
        self.index = index
        self._analyzer = index.analyzer
        sq = [0.0] * index.N
        for term in index.vocabulary(self.field):
            for p in index.postings(term, self.field):
                w = _log_tf(p.tf)
                sq[p.doc] += w * w
        self._norms = [math.sqrt(s) for s in sq]

    # ------------------------------------------------------------------ scoring

    def _require_index(self) -> InvertedIndex:
        if self.index is None:
            raise RuntimeError("call fit() or fit_index() before searching")
        return self.index

    def doc_norm(self, doc: int) -> float:
        """lnc normalization length of internal document ``doc``."""
        return self._norms[doc]

    def query_weights(self, text: str) -> dict[str, float]:
        """Normalized ltc query vector ``{term: weight}`` (OOV and zero-idf terms dropped)."""
        index = self._require_index()
        raw: dict[str, float] = {}
        for term, tf in Counter(self._analyzer.analyze(text)).items():
            df = index.df(term, self.field)
            if df == 0:
                continue
            w = _log_tf(tf) * math.log10(index.N / df)
            if w > 0:
                raw[term] = w
        norm = math.sqrt(sum(w * w for w in raw.values()))
        return {t: w / norm for t, w in raw.items()} if norm > 0 else {}

    def score_all(self, text: str) -> dict[int, float]:
        """Term-at-a-time accumulators ``{internal doc: score}`` for every matching doc."""
        index = self._require_index()
        acc: dict[int, float] = defaultdict(float)
        for term, wq in self.query_weights(text).items():
            for p in index.postings(term, self.field):
                acc[p.doc] += wq * _log_tf(p.tf) / self._norms[p.doc]
        return acc

    def score_text(self, text: str, k: int = 100) -> Ranking:
        """Top ``k`` ``(doc_id, score)`` for a free-text query, ties by doc_id ascending."""
        index = self._require_index()
        if k <= 0:
            return []
        acc = self.score_all(text)
        top = heapq.nsmallest(k, acc.items(), key=lambda kv: (-kv[1], index.doc_id(kv[0])))
        return [(index.doc_id(d), s) for d, s in top]

    def search(self, query: Query, k: int = 100) -> Ranking:
        """Rank documents for ``query.text`` (filters are ignored by plain TF-IDF)."""
        return self.score_text(query.text, k)
