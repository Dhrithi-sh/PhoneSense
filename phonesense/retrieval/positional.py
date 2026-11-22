"""Phrase and proximity search, and a phrase-boosted ranker (T07).

    from phonesense.retrieval.positional import phrase_matches, proximity_matches
    phrase_matches(idx, "wireless charging")          # {internal doc: number of occurrences}
    proximity_matches(idx, "battery", "life", 3)      # within 3 positions, either order
    parse_positional('"wireless charging" battery /3 life samsung')

* **Phrase** ``"t1 t2 ..."``: positional intersection. The phrase is analyzed like documents,
  and the *relative* token offsets of its terms are kept (stopword gaps included, the index
  stores the same gaps), so ``"battery life"`` needs positions p, p+1 and ``"phone with
  camera"`` (-> phone@0, camera@2) needs p, p+2. Returns the number of matching start
  positions per doc.
* **Proximity** ``t1 /k t2``: both terms with ``|pos1 - pos2| <= k`` (k included), in either
  order. Returns the number of matching position pairs per doc.
* Candidate docs come from the doc-id intersection with skip pointers
  (:func:`phonesense.retrieval.boolean.intersect_sorted`), smallest postings list first.
* Every function takes a ``field`` (default ``text``).

**PhraseBoostRetriever** wraps a scorer with ``score_all`` (BM25, BM25F, TF-IDF):

    score(q, d) = base(q, d) + phrase_weight * sum_{phrase p in q} ln(1 + matches(p, d))

The phrases of a query are its quoted phrases plus any auto-phrase from config
(``phrase.auto_phrases``: "wireless charging", "battery life", ...) found in the query text.
The boost only goes to documents the base scorer already scored. ``phrase_weight = 0`` gives
exactly the base ranking. The weight is tuned on dev only (``scripts/tune_phrase.py``).

Adapted from our midterm project Dhvani-lingual (``index/search.py``: ``phrase_search``,
``proximity_search``): same consecutive-offset and windowed checks, now on internal ids, with
stopword-gap offsets, match counts and two-term proximity in either order; the boost is new.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from phonesense.index import Analyzer, InvertedIndex
from phonesense.io import Ranking
from phonesense.retrieval.base import Retriever
from phonesense.retrieval.bm25 import top_k
from phonesense.retrieval.boolean import intersect_sorted
from phonesense.schemas import Phone, Query

_QUOTED = re.compile(r'"([^"]+)"')
_PROX = re.compile(r"(\S+)\s+/(\d+)\s+(\S+)")


def phrase_config() -> dict[str, Any]:
    """The ``phrase:`` section of config.yaml."""
    from phonesense.config import get_config

    return dict(get_config().get("phrase") or {})


def _candidates(index: InvertedIndex, terms: Sequence[str], field: str) -> list[int]:
    lists = sorted(([p.doc for p in index.postings(t, field)] for t in set(terms)), key=len)
    if not lists or not lists[0]:
        return []
    out = lists[0]
    for nxt in lists[1:]:
        out = intersect_sorted(out, nxt)
        if not out:
            break
    return out


def _positions(index: InvertedIndex, term: str, field: str, docs: set[int]) -> dict[int, list[int]]:
    return {p.doc: p.positions for p in index.postings(term, field) if p.doc in docs}


def phrase_matches(index: InvertedIndex, phrase: str, field: str = "text") -> dict[int, int]:
    """``{internal doc: occurrences}`` of the analyzed phrase (relative offsets kept)."""
    analyzed = index.analyzer.analyze_with_positions(phrase)
    if not analyzed:
        return {}
    start = analyzed[0][1]
    parts = [(t, p - start) for t, p in analyzed]
    terms = [t for t, _ in parts]
    docs = _candidates(index, terms, field)
    if not docs:
        return {}
    dset = set(docs)
    pos = {t: _positions(index, t, field, dset) for t in set(terms)}
    out: dict[int, int] = {}
    first = parts[0][0]
    rest = parts[1:]
    for d in docs:
        sets = {t: set(pos[t][d]) for t, _ in rest}
        n = sum(1 for p in pos[first][d] if all(p + off in sets[t] for t, off in rest))
        if n:
            out[d] = n
    return out


def proximity_matches(
    index: InvertedIndex, term1: str, term2: str, k: int, field: str = "text"
) -> dict[int, int]:
    """``{internal doc: pairs}`` with the two (analyzed) terms within ``k`` positions."""
    if k < 0:
        raise ValueError("k must be >= 0")
    a1, a2 = index.analyzer.analyze(term1), index.analyzer.analyze(term2)
    if len(a1) != 1 or len(a2) != 1:
        raise ValueError(f"proximity needs single terms, got {term1!r} -> {a1}, {term2!r} -> {a2}")
    t1, t2 = a1[0], a2[0]
    docs = _candidates(index, [t1, t2], field)
    dset = set(docs)
    p1, p2 = _positions(index, t1, field, dset), _positions(index, t2, field, dset)
    out: dict[int, int] = {}
    for d in docs:
        xs, ys = p1[d], p2[d]
        n = sum(bisect_right(ys, x + k) - bisect_left(ys, x - k) for x in xs)
        if t1 == t2:  # don't pair a position with itself
            n -= len(xs)
        if n > 0:
            out[d] = n
    return out


def parse_positional(text: str) -> tuple[list[str], list[tuple[str, str, int]], str]:
    """Split a query into quoted phrases, ``a /k b`` proximity clauses and the rest."""
    phrases = _QUOTED.findall(text)
    rest = _QUOTED.sub(" ", text)
    prox = [(a, b, int(k)) for a, k, b in _PROX.findall(rest)]
    rest = _PROX.sub(" ", rest)
    return phrases, prox, " ".join(rest.split())


def query_phrases(text: str, auto_phrases: Sequence[str] = ()) -> list[str]:
    """Quoted phrases plus the auto-phrases that occur in the (normalized) query text."""
    phrases = _QUOTED.findall(text)
    norm = " ".join(Analyzer.normalize(_QUOTED.sub(" ", text)).split())
    for p in auto_phrases:
        if re.search(rf"\b{re.escape(p.lower())}\b", norm) and p not in phrases:
            phrases.append(p)
    return phrases


class PhraseBoostRetriever(Retriever):
    """Base score plus ``phrase_weight * sum ln(1 + phrase matches)``; see module docstring."""

    def __init__(
        self,
        base: Retriever,
        phrase_weight: float | None = None,
        auto_phrases: Sequence[str] | None = None,
        field: str = "text",
        name: str | None = None,
    ) -> None:
        cfg = phrase_config()
        tuned = cfg.get("tuned") or {}
        if phrase_weight is None:
            phrase_weight = float(tuned.get("weight", cfg.get("weight", 1.0)))
        if phrase_weight < 0:
            raise ValueError("phrase_weight must be >= 0")
        self.base, self.phrase_weight, self.field = base, float(phrase_weight), field
        self.auto_phrases = list(
            cfg.get("auto_phrases", []) if auto_phrases is None else auto_phrases
        )
        self._name = name

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name or f"{self.base.name}+phrase"

    @property
    def index(self) -> InvertedIndex:
        return self.base.index  # type: ignore[attr-defined]

    def fit(self, phones: list[Phone]) -> None:
        self.base.fit(phones)

    def score_all(self, text: str, allowed: set[int] | None = None) -> dict[int, float]:
        plain = _QUOTED.sub(lambda m: m.group(1), text)
        acc: dict[int, float] = defaultdict(float, self.base.score_all(plain, allowed))  # type: ignore[attr-defined]
        if self.phrase_weight == 0:
            return acc
        for phrase in query_phrases(text, self.auto_phrases):
            for d, n in phrase_matches(self.index, phrase, self.field).items():
                if d in acc:
                    acc[d] += self.phrase_weight * math.log1p(n)
        return acc

    def score_text(self, text: str, k: int = 100, allowed: set[int] | None = None) -> Ranking:
        return top_k(self.score_all(text, allowed), self.index, k)

    def search(self, query: Query, k: int = 100) -> Ranking:
        return self.score_text(query.text, k)
