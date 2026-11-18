"""Tuning the fusion settings on dev queries (owner: B, T13).

Fusion has a few knobs: the RRF constant ``k`` and, for linear fusion, one weight per system.
We choose them by maximising mean nDCG@10 on the **dev** queries only. The ``test`` queries are
never read here: CONTRACT.md allows them once per final configuration, and tuning on them would
make the reported number optimistic. ``scripts/tune_fusion.py`` enforces this by refusing runs
and queries that are not dev.

Everything works on saved rankings (``runs/*.run``), so tuning never re-runs a retriever.

NOTE on the metric: ``ndcg_at_k`` below is a small local implementation so that tuning does not
wait for the evaluation harness (T11, owned by A). It uses gain ``2^grade - 1`` and discount
``log2(rank + 1)``. Once T11 lands, check that its nDCG@10 uses the same definition (or switch
this module to it) so tuned and reported numbers agree.

Tuning on a few dozen queries overfits easily, so ties go to the setting closest to equal
weights, and the report should state how close the tuned score is to the equal-weight one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import product

from phonesense.io import Ranking
from phonesense.retrieval.fusion import reciprocal_rank_fusion, weighted_linear_fusion

Runs = Mapping[str, Ranking]  # {qid: ranking} for one system


def ndcg_at_k(ranking: Ranking, grades: Mapping[str, int], k: int = 10) -> float:
    """nDCG@k with gain ``2^grade - 1``. Unjudged documents count as grade 0.

    The ideal ranking is built from every judged document of the query. Returns 0.0 when the
    query has no document with grade > 0 (nothing to find).
    """
    dcg = sum(
        (2 ** grades.get(doc_id, 0) - 1) / math.log2(i + 2)
        for i, (doc_id, _) in enumerate(ranking[:k])
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def evaluable_qids(qids: Iterable[str], qrels: Mapping[str, Mapping[str, int]]) -> list[str]:
    """Queries that have at least one relevant judgment (others cannot be scored), sorted."""
    return sorted(q for q in set(qids) if any(g > 0 for g in qrels.get(q, {}).values()))


def mean_ndcg(
    rankings: Runs, qrels: Mapping[str, Mapping[str, int]], qids: Sequence[str], k: int = 10
) -> float:
    """Mean nDCG@k over ``qids``; a query missing from ``rankings`` scores 0."""
    if not qids:
        raise ValueError("no queries to evaluate")
    return sum(ndcg_at_k(rankings.get(q, []), qrels.get(q, {}), k) for q in qids) / len(qids)


def weight_grid(n: int, step: float = 0.1) -> Iterator[tuple[float, ...]]:
    """All weight vectors of length ``n`` made of multiples of ``step`` that sum to 1.

    Order is lexicographic and therefore deterministic. ``step`` must divide 1 evenly.
    """
    units = round(1 / step)
    if n < 1 or units < 1 or abs(units * step - 1) > 1e-9:
        raise ValueError(f"step must divide 1 evenly and n >= 1 (n={n}, step={step})")
    for combo in product(range(units + 1), repeat=n):
        if sum(combo) == units:
            yield tuple(c / units for c in combo)


def fuse_runs(
    systems: Sequence[Runs],
    qids: Sequence[str],
    method: str,
    weights: Sequence[float] | None = None,
    rrf_k: int = 60,
    depth: int = 100,
) -> dict[str, Ranking]:
    """Fuse the saved rankings of several systems for every query in ``qids``."""
    fused: dict[str, Ranking] = {}
    for qid in qids:
        parts = [s.get(qid, []) for s in systems]
        if method == "rrf":
            fused[qid] = reciprocal_rank_fusion(parts, rrf_k, weights, depth)
        elif method == "linear":
            fused[qid] = weighted_linear_fusion(parts, weights, depth)
        else:
            raise ValueError(f"unknown fusion method {method!r}")
    return fused


def _distance_from_equal(weights: Sequence[float]) -> float:
    eq = 1 / len(weights)
    return sum((w - eq) ** 2 for w in weights)


def pick_best(trials: Sequence[dict]) -> dict:
    """Highest ``score``; ties go to the weights closest to equal, then to the first listed."""
    if not trials:
        raise ValueError("no trials")
    best_score = max(t["score"] for t in trials)
    tied = [t for t in trials if t["score"] >= best_score - 1e-12]
    return min(tied, key=lambda t: _distance_from_equal(t.get("weights") or [1.0]))


def tune_linear(
    systems: Sequence[Runs],
    qrels: Mapping[str, Mapping[str, int]],
    qids: Sequence[str],
    step: float = 0.1,
    k: int = 10,
    depth: int = 100,
) -> list[dict]:
    """Score every weight vector on the grid: ``[{"weights": [...], "score": nDCG@k}, ...]``."""
    trials = []
    for ws in weight_grid(len(systems), step):
        fused = fuse_runs(systems, qids, "linear", ws, depth=depth)
        trials.append({"weights": list(ws), "score": mean_ndcg(fused, qrels, qids, k)})
    return trials


def tune_rrf_k(
    systems: Sequence[Runs],
    qrels: Mapping[str, Mapping[str, int]],
    qids: Sequence[str],
    k_values: Sequence[int] = (10, 30, 60, 100),
    k: int = 10,
    depth: int = 100,
    weights: Sequence[float] | None = None,
) -> list[dict]:
    """Score each RRF constant: ``[{"rrf_k": c, "score": nDCG@k}, ...]`` (equal weights)."""
    return [
        {
            "rrf_k": c,
            "score": mean_ndcg(fuse_runs(systems, qids, "rrf", weights, c, depth), qrels, qids, k),
        }
        for c in k_values
    ]
