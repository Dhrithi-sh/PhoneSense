"""Okapi BM25 and a field-weighted "BM25F-lite" (T05).

Okapi BM25 (Robertson & Zaragoza 2009; Manning et al., IIR sec. 11.4.3). For query q and
document d in one field, with N documents, df_t, tf(t,d), dl = |d| and avgdl:

    score(q, d) = sum_{t in q} w_q(t) * idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + K(d))
    K(d)        = k1 * (1 - b + b * dl / avgdl)
    idf(t)      = ln(1 + (N - df_t + 0.5) / (df_t + 0.5))      (never negative)
    w_q(t)      = 1 for every distinct query term                 (query_tf: once, default)
                  (k3 + 1) * qtf / (k3 + qtf), k3 = 8              (query_tf: k3)

* k1 controls tf saturation (k1 = 0: binary presence; large k1: close to raw tf).
* b controls length normalization (b = 0: none; b = 1: full, K(d) proportional to dl/avgdl).
* The "1 +" in the idf keeps it positive for terms in more than half the documents.

BM25F-lite combines per-field BM25 scores linearly:

    score_F(q, d) = sum_f w_f * score_f(q, d)

where score_f is BM25 on field f with that field's df, lengths and avgdl (the index stores
them per field), and w_f the field weight (default title 3, features 1.5, description 1). It
is "lite" because true BM25F merges weighted tfs *before* saturation. With a single field of
weight 1 it equals plain BM25 on that field.

Scoring is term-at-a-time with one accumulator per matching document; the top k come from a
heap with ties broken by doc_id ascending (as in TF-IDF), so runs are deterministic.

Worked example (k1 = 1.2, b = 0.75; checked in ``tests/test_bm25.py``)::

    d1 = camera camera phone (dl 3)   d2 = phone battery (dl 2)   d3 = battery x3 (dl 3)
    N = 3, avgdl = 8/3, query "camera phone"
    idf(camera) = ln(1 + 2.5/1.5) = 0.980829    idf(phone) = ln(1 + 1.5/2.5) = 0.470004
    K(d1) = 1.2 * (0.25 + 0.75 * 3/(8/3)) = 1.3125      K(d2) = 0.975
    d1: 0.980829 * 2 * 2.2 / (2 + 1.3125) + 0.470004 * 2.2 / (1 + 1.3125)
        = 1.302837 + 0.447139 = 1.749976
    d2: 0.470004 * 2.2 / (1 + 0.975) = 0.523548
    d3: 0 (no query term; not returned)

Adapted from our midterm project Dhvani-lingual (``dhvani/rank/bm25.py``): same formula and
idf. Changed: runs on :class:`~phonesense.index.InvertedIndex` per field (the midterm summed
zones with a headline boost before saturation), query weights are once/k3 instead of
expansion weights, and the field-weighted variant is new.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

from phonesense.index import Analyzer, InvertedIndex
from phonesense.io import Ranking
from phonesense.retrieval.base import Retriever
from phonesense.schemas import Phone, Query

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75
DEFAULT_K3 = 8.0
DEFAULT_FIELD_WEIGHTS: dict[str, float] = {"title": 3.0, "features": 1.5, "description": 1.0}
QUERY_TF_MODES = ("once", "k3")


def bm25_config() -> dict[str, Any]:
    """The ``bm25:`` section of config.yaml (empty dict if absent)."""
    from phonesense.config import get_config

    return dict(get_config().get("bm25") or {})


def idf(df: int, n: int) -> float:
    """Non-negative BM25 idf ``ln(1 + (N - df + 0.5) / (df + 0.5))``."""
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def top_k(acc: Mapping[int, float], index: InvertedIndex, k: int) -> Ranking:
    """Top k ``(doc_id, score)`` from accumulators, score desc, doc_id asc for ties."""
    if k <= 0:
        return []
    best = heapq.nsmallest(k, acc.items(), key=lambda kv: (-kv[1], index.doc_id(kv[0])))
    return [(index.doc_id(d), s) for d, s in best]


def _fmt(x: float) -> str:
    return f"{x:g}"


class BM25Retriever(Retriever):
    """Okapi BM25 over one field of the inverted index.

    ``fit(phones)`` builds a private index for ``field``; or pass a prebuilt ``index`` (e.g.
    the saved one) to reuse it. ``k1`` / ``b`` can be changed later with :meth:`set_params`
    without re-reading postings (used by the tuning grid).
    """

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        field: str = "text",
        query_tf: str | None = None,
        k3: float | None = None,
        analyzer: Analyzer | None = None,
        index: InvertedIndex | None = None,
        name: str | None = None,
    ) -> None:
        cfg = bm25_config()
        self.field = field
        self.query_tf = query_tf or cfg.get("query_tf", "once")
        if self.query_tf not in QUERY_TF_MODES:
            raise ValueError(f"query_tf must be one of {QUERY_TF_MODES}, got {self.query_tf!r}")
        self.k3 = float(k3 if k3 is not None else cfg.get("k3", DEFAULT_K3))
        self._name = name
        self._analyzer = analyzer
        self.index: InvertedIndex | None = None
        self._K: list[float] = []
        self.set_params(k1, b)
        if index is not None:
            self.fit_index(index)

    @property
    def name(self) -> str:  # type: ignore[override]
        if self._name:
            return self._name
        suffix = "" if self.field == "text" else f".{self.field}"
        return f"bm25{suffix}_k{_fmt(self.k1)}_b{_fmt(self.b)}"

    def set_params(self, k1: float, b: float) -> None:
        """Change k1 / b (recomputes the per-document length factor if fitted)."""
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
        self.k1, self.b = float(k1), float(b)
        if self.index is not None:
            self._compute_K()

    # ------------------------------------------------------------------ fitting

    def fit(self, phones: list[Phone]) -> None:
        analyzer = self._analyzer if self._analyzer is not None else Analyzer.from_config()
        self.fit_index(InvertedIndex.build(phones, analyzer, fields=(self.field,)))

    def fit_index(self, index: InvertedIndex) -> None:
        """Use an existing index (must contain ``self.field``)."""
        if self.field not in index.fields:
            raise ValueError(f"index has no field {self.field!r} (fields: {index.fields})")
        self.index = index
        self._analyzer = index.analyzer
        self._compute_K()

    def _compute_K(self) -> None:  # noqa: N802 (IR notation)
        assert self.index is not None
        avgdl = self.index.avg_doc_len(self.field)
        lens = self.index.doc_lens(self.field)
        if avgdl <= 0:
            self._K = [self.k1] * len(lens)
        else:
            self._K = [self.k1 * (1 - self.b + self.b * dl / avgdl) for dl in lens]

    # ------------------------------------------------------------------ scoring

    def _require_index(self) -> InvertedIndex:
        if self.index is None:
            raise RuntimeError("call fit() or fit_index() before searching")
        return self.index

    def query_weights(self, text: str) -> dict[str, float]:
        """``{term: w_q}`` for the analyzed query (OOV terms dropped)."""
        index = self._require_index()
        counts = Counter(self._analyzer.analyze(text))
        out = {}
        for term, qtf in counts.items():
            if index.df(term, self.field) == 0:
                continue
            out[term] = 1.0 if self.query_tf == "once" else (self.k3 + 1) * qtf / (self.k3 + qtf)
        return out

    def score_all(self, text: str, allowed: set[int] | None = None) -> dict[int, float]:
        """Accumulators ``{internal doc: score}`` for every doc sharing a query term.

        ``allowed`` (internal doc ids) restricts scoring to those docs (pre-filtering, T06);
        idf and lengths still come from the whole collection, so surviving docs keep exactly
        their unfiltered scores.
        """
        index = self._require_index()
        n, k1p1 = index.N, self.k1 + 1
        acc: dict[int, float] = defaultdict(float)
        for term, wq in self.query_weights(text).items():
            plist = index.postings(term, self.field)
            w = wq * idf(len(plist), n)
            for p in plist:
                if allowed is not None and p.doc not in allowed:
                    continue
                acc[p.doc] += w * p.tf * k1p1 / (p.tf + self._K[p.doc])
        return acc

    def score_text(self, text: str, k: int = 100, allowed: set[int] | None = None) -> Ranking:
        """Top k ``(doc_id, score)`` for a free-text query (optionally only ``allowed`` docs)."""
        return top_k(self.score_all(text, allowed), self._require_index(), k)

    def search(self, query: Query, k: int = 100) -> Ranking:
        """Rank documents for ``query.text`` (filters are ignored by plain BM25)."""
        return self.score_text(query.text, k)


class BM25FRetriever(Retriever):
    """Field-weighted BM25 ("BM25F-lite"): ``sum_f w_f * BM25_f``; see the module docstring.

    ``weights`` maps field -> weight (default from ``bm25.field_weights`` in config.yaml, else
    title 3 / features 1.5 / description 1). Fields with weight 0 are skipped.
    """

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        query_tf: str | None = None,
        k3: float | None = None,
        analyzer: Analyzer | None = None,
        index: InvertedIndex | None = None,
        name: str | None = None,
    ) -> None:
        cfg_w = bm25_config().get("field_weights")
        self.weights = {f: float(w) for f, w in (weights or cfg_w or DEFAULT_FIELD_WEIGHTS).items()}
        if any(w < 0 for w in self.weights.values()):
            raise ValueError(f"field weights must be >= 0, got {self.weights}")
        self._name = name
        self._analyzer = analyzer
        self.index: InvertedIndex | None = None
        self.parts = {
            f: BM25Retriever(k1, b, field=f, query_tf=query_tf, k3=k3, analyzer=analyzer)
            for f, w in self.weights.items()
            if w > 0
        }
        if not self.parts:
            raise ValueError("need at least one field with weight > 0")
        self.k1, self.b = float(k1), float(b)
        if index is not None:
            self.fit_index(index)

    @property
    def name(self) -> str:  # type: ignore[override]
        if self._name:
            return self._name
        w = "-".join(f"{f[0]}{_fmt(v)}" for f, v in self.weights.items())
        return f"bm25f_{w}_k{_fmt(self.k1)}_b{_fmt(self.b)}"

    def set_params(self, k1: float, b: float) -> None:
        self.k1, self.b = float(k1), float(b)
        for p in self.parts.values():
            p.set_params(k1, b)

    def fit(self, phones: list[Phone]) -> None:
        analyzer = self._analyzer if self._analyzer is not None else Analyzer.from_config()
        self.fit_index(InvertedIndex.build(phones, analyzer, fields=tuple(self.parts)))

    def fit_index(self, index: InvertedIndex) -> None:
        self.index = index
        for p in self.parts.values():
            p.fit_index(index)

    def score_all(self, text: str, allowed: set[int] | None = None) -> dict[int, float]:
        acc: dict[int, float] = defaultdict(float)
        for f, part in self.parts.items():
            w = self.weights[f]
            for d, s in part.score_all(text, allowed).items():
                acc[d] += w * s
        return acc

    def score_text(self, text: str, k: int = 100, allowed: set[int] | None = None) -> Ranking:
        if self.index is None:
            raise RuntimeError("call fit() or fit_index() before searching")
        return top_k(self.score_all(text, allowed), self.index, k)

    def search(self, query: Query, k: int = 100) -> Ranking:
        return self.score_text(query.text, k)
