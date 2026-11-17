"""Prior-based reranking (T18): mix a run's scores with a per-phone prior such as global
PageRank or the reviewer-trust score. Reads and writes runs in the CONTRACT.md 4.6 format.

For each query the run's scores are min-max normalised to [0, 1], then

    new(d) = (1 - lambda) * score_hat(d) + lambda * prior(d)        prior in [0, 1]

lambda = 0 keeps the original order; lambda = 1 ranks by the prior alone. Only phones
already in the run are reordered: reranking never adds documents.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping

from phonesense.schemas import run_name_split

Ranking = list[tuple[str, float]]


def minmax(ranking: Ranking) -> list[float]:
    """Scores scaled to [0, 1] within one query; all ones when every score is equal."""
    scores = [s for _, s in ranking]
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def mix(ranking: Ranking, prior: Mapping[str, float], lam: float, fill: float) -> Ranking:
    """Rerank one query. Ties are broken by doc_id ascending, like rankings_to_entries."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    mixed = [(d, (1 - lam) * s + lam * prior.get(d, fill))
             for (d, _), s in zip(ranking, minmax(ranking), strict=True)]
    return sorted(mixed, key=lambda x: (-x[1], x[0]))


def rerank_with_prior(rankings: Mapping[str, Ranking], prior: Mapping[str, float], lam: float,
                      missing: float | None = None) -> dict[str, Ranking]:
    """Rerank every query with one global prior. Phones missing from the prior get
    ``missing``, by default the prior's median (neutral: neither pushed up nor down)."""
    fill = missing if missing is not None else (statistics.median(prior.values())
                                                if prior else 0.0)
    return {qid: mix(ranking, prior, lam, fill) for qid, ranking in rankings.items()}


def reranked_run_name(base: str, tag: str) -> str:
    """'bm25_k1.2_b0.75_dev' + 'pagerank0.2' -> 'bm25_k1.2_b0.75+pagerank0.2_dev'."""
    split = run_name_split(base)
    return f"{base[:-len(split) - 1]}+{tag}_{split}"
