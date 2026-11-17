r"""Rocchio pseudo-relevance feedback on top of TF-IDF lnc.ltc (owner: B).

Rocchio (Manning, Raghavan & Schuetze, *IIR*, sec. 9.1.1) moves the query vector towards the
centroid of documents believed relevant and away from the centroid of non-relevant ones::

    q' = alpha * q  +  beta * mean(D_rel)  -  gamma * mean(D_nonrel)

Negative components are dropped (a term cannot count against a document). An optional HyDE term
``+ delta * h`` adds the vector ``h`` of an LLM-written hypothetical listing (see
:mod:`phonesense.retrieval.hyde`); with ``delta = 0`` (the default) this is plain Rocchio.
There is no user to
judge documents, so we use *pseudo*-relevance feedback: the top ``m`` documents of the first
TF-IDF pass are assumed relevant. With the default ``gamma = 0`` no non-relevant set is used;
for ``gamma > 0`` the documents ranked ``m+1 .. m+m_nonrel`` stand in for it.

Vectors are in the same space as the first pass: queries are ltc (normalized, idf-weighted),
documents are lnc (``1 + log10 tf``, cosine-normalized, no idf). Only the ``n_terms`` heaviest
terms of the updated query are kept, because a long expansion drifts away from the topic and
makes the postings walk slower. The final ranking is ``score(d) = sum_t q'_t * lnc(t, d)``,
computed term-at-a-time over the postings of the original index.

Rankings are deterministic: score descending, ``doc_id`` ascending on ties.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from phonesense.io import Ranking
from phonesense.retrieval.base import Retriever
from phonesense.retrieval.tfidf import TfidfRetriever
from phonesense.schemas import Phone, Query

Vector = dict[str, float]


def centroid(vectors: Iterable[Mapping[str, float]]) -> Vector:
    """Component-wise mean of sparse vectors (empty input gives an empty vector)."""
    vecs = list(vectors)
    if not vecs:
        return {}
    total: Vector = defaultdict(float)
    for v in vecs:
        for term, w in v.items():
            total[term] += w
    return {t: w / len(vecs) for t, w in total.items()}


def rocchio_update(
    query: Mapping[str, float],
    relevant: Iterable[Mapping[str, float]],
    nonrelevant: Iterable[Mapping[str, float]] = (),
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.15,
    hypothetical: Mapping[str, float] | None = None,
    delta: float = 0.0,
) -> Vector:
    """Return ``alpha*q + beta*centroid(relevant) - gamma*centroid(nonrelevant) + delta*h``.

    ``h`` is the optional ``hypothetical`` vector (HyDE). Components that end up <= 0 are
    removed. The result is not re-normalized.
    """
    rel, non = centroid(relevant), centroid(nonrelevant)
    hyp = hypothetical or {}
    out: Vector = {}
    for term in set(query) | set(rel) | set(non) | set(hyp):
        w = (
            alpha * query.get(term, 0.0)
            + beta * rel.get(term, 0.0)
            - gamma * non.get(term, 0.0)
            + delta * hyp.get(term, 0.0)
        )
        if w > 0.0:
            out[term] = w
    return out


def top_terms(vec: Mapping[str, float], n: int) -> Vector:
    """Keep the ``n`` heaviest terms (ties broken alphabetically)."""
    if n <= 0:
        return {}
    best = heapq.nsmallest(n, vec.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(best)


def _log_tf(tf: float) -> float:
    return 1.0 + math.log10(tf) if tf > 0 else 0.0


class RocchioRetriever(Retriever):
    """Two-pass retriever: TF-IDF, Rocchio update from the top ``m``, TF-IDF again."""

    def __init__(
        self,
        base: TfidfRetriever | None = None,
        m: int = 10,
        alpha: float = 1.0,
        beta: float = 0.75,
        gamma: float = 0.0,
        n_terms: int = 20,
        m_nonrel: int | None = None,
        delta: float = 0.0,
    ) -> None:
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if min(alpha, beta, gamma, delta) < 0:
            raise ValueError("alpha, beta, gamma and delta must be non-negative")
        self.base = base if base is not None else TfidfRetriever()
        self.m, self.alpha, self.beta, self.gamma = m, alpha, beta, gamma
        self.delta = delta
        self.n_terms = n_terms
        self.m_nonrel = m if m_nonrel is None else m_nonrel
        suffix = f"_d{delta:g}" if delta > 0 else ""
        self.name = f"rocchio_{self.base.name}_m{m}_b{beta:g}_g{gamma:g}_t{n_terms}{suffix}"

    # ------------------------------------------------------------------ fitting

    def fit(self, phones: list[Phone]) -> None:
        self.base.fit(phones)

    # ------------------------------------------------------------------ vectors

    def doc_vector(self, doc: int) -> Vector:
        """lnc vector of internal document ``doc``: ``(1 + log10 tf) / norm``."""
        index = self.base._require_index()
        norm = self.base.doc_norm(doc)
        if norm == 0.0:
            return {}
        tv = index.term_vector(doc, self.base.field)
        return {t: _log_tf(tf) / norm for t, tf in tv.items()}

    def text_vector(self, text: str) -> Vector:
        """lnc-style vector of free text: ``1 + log10 tf``, cosine-normalized, terms not in the
        index dropped. This is the same space as :meth:`doc_vector`."""
        index = self.base._require_index()
        counts = Counter(self.base._analyzer.analyze(text))
        raw = {t: _log_tf(tf) for t, tf in counts.items() if index.df(t, self.base.field) > 0}
        norm = math.sqrt(sum(w * w for w in raw.values()))
        return {t: w / norm for t, w in raw.items()} if norm > 0 else {}

    def _first_pass(self, text: str, depth: int) -> list[int]:
        index = self.base._require_index()
        acc = self.base.score_all(text)
        top = heapq.nsmallest(depth, acc.items(), key=lambda kv: (-kv[1], index.doc_id(kv[0])))
        return [d for d, _ in top]

    def expanded_query(self, text: str, hypothetical: str | None = None) -> Vector:
        """The Rocchio-updated query vector for ``text`` (original weights if nothing matches).

        ``hypothetical`` is an LLM-written listing; with ``delta > 0`` its vector is added.
        """
        q0 = self.base.query_weights(text)
        depth = self.m + (self.m_nonrel if self.gamma > 0 else 0)
        ranked = self._first_pass(text, depth)
        rel = [self.doc_vector(d) for d in ranked[: self.m]]
        non = [self.doc_vector(d) for d in ranked[self.m : self.m + self.m_nonrel]]
        hyp = self.text_vector(hypothetical) if hypothetical and self.delta > 0 else None
        if not rel and not hyp:
            return q0
        updated = rocchio_update(
            q0, rel, non, self.alpha, self.beta, self.gamma, hypothetical=hyp, delta=self.delta
        )
        return top_terms(updated, self.n_terms)

    # ------------------------------------------------------------------ searching

    def score_vector(self, weights: Mapping[str, float], k: int = 100) -> Ranking:
        """Top ``k`` ``(doc_id, score)`` for a weighted term vector against lnc documents."""
        index = self.base._require_index()
        if k <= 0:
            return []
        acc: dict[int, float] = defaultdict(float)
        for term, w in weights.items():
            for p in index.postings(term, self.base.field):
                acc[p.doc] += w * _log_tf(p.tf) / self.base.doc_norm(p.doc)
        top = heapq.nsmallest(k, acc.items(), key=lambda kv: (-kv[1], index.doc_id(kv[0])))
        return [(index.doc_id(d), s) for d, s in top]

    def score_text(self, text: str, k: int = 100) -> Ranking:
        return self.score_vector(self.expanded_query(text), k)

    def search(self, query: Query, k: int = 100) -> Ranking:
        """Rank documents for ``query.text`` after pseudo-relevance feedback."""
        return self.score_text(query.text, k)
