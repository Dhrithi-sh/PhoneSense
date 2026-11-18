"""Rank fusion: merge the rankings of several retrievers into one (owner: B, T13).

Each retriever (TF-IDF, BM25, dense, Rocchio, HyDE, ...) is good at different queries, and
their raw scores live on different scales, so scores cannot simply be added. This module
offers the standard fixes:

* **Reciprocal Rank Fusion (RRF)**, Cormack, Clarke & Buettcher (SIGIR 2009)::

      score(d) = sum_i  w_i / (k + rank_i(d))

  ``rank_i(d)`` is the 1-based position of ``d`` in ranking ``i``. Only *ranks* are used, so no
  score calibration is needed. ``k`` (default 60, the published value) damps the head: a large
  ``k`` makes the gap between rank 1 and rank 2 small, a small ``k`` rewards a top position
  strongly. A document missing from a ranking contributes 0 for that system.

Every function is deterministic: documents with equal fused scores are ordered by ``doc_id``
ascending, as required of every ranking in CONTRACT.md. Inputs are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from phonesense.config import get_config
from phonesense.io import Ranking
from phonesense.retrieval.base import Retriever
from phonesense.schemas import MAX_RUN_DEPTH, Phone, Query

# Scores that agree to this many decimals count as tied, so float rounding noise from adding
# terms in a different order cannot override the doc_id tie-break.
TIE_DECIMALS = 12


def fusion_defaults() -> dict:
    """The ``fusion:`` block of config.yaml (empty dict if absent)."""
    return dict(get_config().get("fusion", {}))


def _check_weights(weights: Sequence[float] | None, n: int) -> list[float]:
    """Validate per-system weights (default: all 1.0). They must be >= 0 with a positive sum."""
    if weights is None:
        return [1.0] * n
    ws = [float(w) for w in weights]
    if len(ws) != n:
        raise ValueError(f"got {len(ws)} weights for {n} rankings")
    if any(w < 0 for w in ws) or sum(ws) <= 0:
        raise ValueError(f"weights must be >= 0 with a positive sum, got {ws}")
    return ws


def _top(scores: Mapping[str, float], depth: int | None) -> Ranking:
    """Sort by fused score descending, ties by doc_id ascending, and cut to ``depth``."""
    ordered = sorted(scores.items(), key=lambda kv: (-round(kv[1], TIE_DECIMALS), kv[0]))
    return ordered if depth is None else ordered[:depth]


def reciprocal_rank_fusion(
    rankings: Sequence[Ranking],
    k: int = 60,
    weights: Sequence[float] | None = None,
    depth: int | None = None,
) -> Ranking:
    """Fuse ``rankings`` (each ``[(doc_id, score), ...]`` best first) with RRF.

    ``k`` is the RRF constant (> 0); ``weights`` scales each system's vote (default equal);
    ``depth`` cuts the fused list (default: keep every document that appears anywhere). A
    system with weight 0 is ignored completely. The scores inside the input rankings are
    ignored, only their order matters.
    """
    if k <= 0:
        raise ValueError(f"RRF constant k must be > 0, got {k}")
    ws = _check_weights(weights, len(rankings))
    fused: dict[str, float] = {}
    for w, ranking in zip(ws, rankings, strict=True):
        if w == 0:  # weight 0 means "ignore this system", its documents must not enter the result
            continue
        seen: set[str] = set()
        rank = 0
        for doc_id, _ in ranking:
            if doc_id in seen:  # a repeated doc keeps its best (first) position only
                continue
            seen.add(doc_id)
            rank += 1
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return _top(fused, depth)


METHODS = ("rrf",)


class FusionRetriever(Retriever):
    """Runs several retrievers on the same query and fuses their top ``depth`` results.

    ``method``, ``rrf_k``, ``depth`` and ``weights`` default to the ``fusion:`` block of
    config.yaml. Set ``fit_children=False`` when the retrievers are already fitted (for example
    when they share one index) so ``fit`` does not build them a second time.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        method: str | None = None,
        weights: Sequence[float] | None = None,
        rrf_k: int | None = None,
        depth: int | None = None,
        name: str | None = None,
        fit_children: bool = True,
    ) -> None:
        if not retrievers:
            raise ValueError("need at least one retriever to fuse")
        cfg = fusion_defaults()
        self.retrievers = list(retrievers)
        self.method = method or cfg.get("method", "rrf")
        if self.method not in METHODS:
            raise ValueError(f"unknown fusion method {self.method!r}; use one of {METHODS}")
        self.rrf_k = int(rrf_k if rrf_k is not None else cfg.get("rrf_k", 60))
        self.depth = int(depth if depth is not None else cfg.get("depth", MAX_RUN_DEPTH))
        self.weights = list(weights) if weights is not None else cfg.get("weights")
        _check_weights(self.weights, len(self.retrievers))
        self.fit_children = fit_children
        self.name = name or self._default_name()

    def _default_name(self) -> str:
        parts = "+".join(r.name for r in self.retrievers)
        return f"fusion_{self.method}_k{self.rrf_k}_{parts}"

    def fit(self, phones: list[Phone]) -> None:
        if self.fit_children:
            for r in self.retrievers:
                r.fit(phones)

    def fuse(self, rankings: Sequence[Ranking], k: int) -> Ranking:
        """Fuse already-retrieved rankings with this retriever's method and settings."""
        return reciprocal_rank_fusion(rankings, self.rrf_k, self.weights, depth=k)

    def search(self, query: Query, k: int = 100) -> Ranking:
        """Ask every retriever for its top ``depth`` and fuse; returns at most ``k`` pairs."""
        rankings = [r.search(query, self.depth) for r in self.retrievers]
        return self.fuse(rankings, k)
