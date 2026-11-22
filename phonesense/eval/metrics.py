"""Ranked-retrieval metrics for one query (T11).

Every function takes ``ranking`` (doc ids, best first) and ``qrels`` (``{doc_id: grade}`` for
that query, grades 0-3 as in CONTRACT.md 4.5).

Conventions (also in the README "Evaluation" section):

* **Binary relevance** = grade >= ``threshold`` (default: ``eval.relevance_threshold`` in
  config.yaml, which is 2 = ``schemas.RELEVANCE_THRESHOLD``). Used by P, R, F1, AP, RR,
  R-precision.
* **Unjudged documents count as non-relevant** (grade 0).
* **nDCG** uses graded gains ``2^grade - 1`` and the discount ``log2(rank + 1)``. The ideal
  DCG sorts *all* judged documents of the query by grade, not just the retrieved ones.
* Queries with no relevant document are not scored here; :mod:`.evaluate` excludes them from
  the averages and lists them in the results ``meta``.
* ``judged@k`` = fraction of the top k that has *any* judgment (grade 0 included). Our qrels
  come from pooling, so a run with low judged@10 (e.g. a new system that wasn't pooled) has
  unreliable scores: always report it next to the other metrics.
* Precision@k divides by k even if the run has fewer than k documents (TREC convention).

    P@k      = |rel in top k| / k
    R@k      = |rel in top k| / |rel|
    F1@k     = 2 P R / (P + R)
    AP       = (1/|rel|) * sum over ranks r of relevant docs of P@r     (MAP = mean AP)
    RR       = 1 / rank of the first relevant doc (0 if none retrieved)
    R-prec   = P@R with R = |rel|
    DCG@k    = sum_{r<=k} (2^g_r - 1) / log2(r + 1),  nDCG@k = DCG@k / IDCG@k
    11-point = interpolated precision at recall 0.0, 0.1, ..., 1.0 (max precision at any
               recall >= level)

Adapted from our midterm project Dhvani-lingual (``dhvani/eval/metrics.py``): same
definitions for P/R/F1/AP/11-point; changed to a configurable threshold (the midterm used
grade >= 1), exponential nDCG gains (the midterm used linear gains), and added RR,
R-precision and judged@k.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from phonesense.schemas import RELEVANCE_THRESHOLD

Ranking = Sequence[str]
Judgments = Mapping[str, int]
RECALL_LEVELS: tuple[float, ...] = tuple(i / 10 for i in range(11))


def default_threshold() -> int:
    """Binary relevance threshold from config.yaml (``eval.relevance_threshold``)."""
    from phonesense.config import get_config

    return int(get_config().get("eval", {}).get("relevance_threshold", RELEVANCE_THRESHOLD))


def _thr(threshold: int | None) -> int:
    return default_threshold() if threshold is None else threshold


def relevant_set(qrels: Judgments, threshold: int | None = None) -> set[str]:
    """Doc ids with grade >= threshold."""
    t = _thr(threshold)
    return {d for d, g in qrels.items() if g >= t}


def _hits(ranking: Ranking, rel: set[str], k: int) -> int:
    return sum(1 for d in ranking[:k] if d in rel)


def precision_at_k(
    ranking: Ranking, qrels: Judgments, k: int, threshold: int | None = None
) -> float:
    """Relevant docs in the top k, divided by k."""
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    return _hits(ranking, relevant_set(qrels, threshold), k) / k


def recall_at_k(ranking: Ranking, qrels: Judgments, k: int, threshold: int | None = None) -> float:
    """Relevant docs in the top k, divided by all relevant docs (0 if there are none)."""
    rel = relevant_set(qrels, threshold)
    return _hits(ranking, rel, k) / len(rel) if rel else 0.0


def f1_at_k(ranking: Ranking, qrels: Judgments, k: int, threshold: int | None = None) -> float:
    """Harmonic mean of P@k and R@k (0 if both are 0)."""
    p = precision_at_k(ranking, qrels, k, threshold)
    r = recall_at_k(ranking, qrels, k, threshold)
    return 2 * p * r / (p + r) if p + r else 0.0


def average_precision(ranking: Ranking, qrels: Judgments, threshold: int | None = None) -> float:
    """Mean of P@r over the ranks r of retrieved relevant docs, divided by |rel|."""
    rel = relevant_set(qrels, threshold)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            hits += 1
            total += hits / i
    return total / len(rel)


def reciprocal_rank(ranking: Ranking, qrels: Judgments, threshold: int | None = None) -> float:
    """1 / rank of the first relevant doc; 0 if none is retrieved."""
    rel = relevant_set(qrels, threshold)
    for i, d in enumerate(ranking, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def r_precision(ranking: Ranking, qrels: Judgments, threshold: int | None = None) -> float:
    """Precision at rank R, where R = number of relevant docs (0 if R = 0)."""
    rel = relevant_set(qrels, threshold)
    return _hits(ranking, rel, len(rel)) / len(rel) if rel else 0.0


def _gain(grade: int) -> float:
    return float(2**grade - 1)


def dcg_at_k(ranking: Ranking, qrels: Judgments, k: int) -> float:
    """sum_{r<=k} (2^grade - 1) / log2(r + 1); unjudged docs have grade 0."""
    return sum(_gain(qrels.get(d, 0)) / math.log2(i + 1) for i, d in enumerate(ranking[:k], 1))


def ndcg_at_k(ranking: Ranking, qrels: Judgments, k: int) -> float:
    """DCG@k divided by the DCG@k of all judged docs sorted by grade (0 if that is 0)."""
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(_gain(g) / math.log2(i + 1) for i, g in enumerate(ideal, 1))
    return dcg_at_k(ranking, qrels, k) / idcg if idcg > 0 else 0.0


def judged_at_k(ranking: Ranking, qrels: Judgments, k: int) -> float:
    """Fraction of the top k (divided by k) that has any judgment, relevant or not."""
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    return sum(1 for d in ranking[:k] if d in qrels) / k


def interpolated_precision_11pt(
    ranking: Ranking, qrels: Judgments, threshold: int | None = None
) -> list[float]:
    """Interpolated precision at recall 0.0, 0.1, ..., 1.0 for one query."""
    rel = relevant_set(qrels, threshold)
    if not rel:
        return [0.0] * len(RECALL_LEVELS)
    points, hits = [], 0
    for i, d in enumerate(ranking, 1):
        if d in rel:
            hits += 1
            points.append((hits / len(rel), hits / i))
    return [max((p for r, p in points if r >= lvl - 1e-12), default=0.0) for lvl in RECALL_LEVELS]


# ------------------------------------------------------------------ metric names


DEFAULT_KS: tuple[int, ...] = (5, 10, 20)


def metric_names(ks: Sequence[int] = DEFAULT_KS, judged_ks: Sequence[int] = (10,)) -> list[str]:
    """Names used in results JSON (CONTRACT.md 4.9), in a stable order."""
    names = [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks] + [f"F1@{k}" for k in ks]
    names += ["MAP"] + [f"nDCG@{k}" for k in ks] + ["RR", "R-prec"]
    names += [f"judged@{k}" for k in judged_ks]
    return names


def canonical_metric(name: str, known: Sequence[str] | None = None) -> str:
    """Map a user-typed name (``ndcg@10``, ``map``, ``p@5``) to its canonical spelling."""
    pool = list(known) if known is not None else metric_names((1, 5, 10, 20, 50, 100), (5, 10, 20))
    for n in pool:
        if n.lower() == name.lower():
            return n
    raise ValueError(f"unknown metric {name!r}; known: {', '.join(pool)}")


def score_query(
    ranking: Ranking,
    qrels: Judgments,
    ks: Sequence[int] = DEFAULT_KS,
    judged_ks: Sequence[int] = (10,),
    threshold: int | None = None,
) -> dict[str, float]:
    """All metrics of :func:`metric_names` for one query (``MAP`` holds this query's AP)."""
    t = _thr(threshold)
    out: dict[str, float] = {}
    for k in ks:
        out[f"P@{k}"] = precision_at_k(ranking, qrels, k, t)
    for k in ks:
        out[f"R@{k}"] = recall_at_k(ranking, qrels, k, t)
    for k in ks:
        out[f"F1@{k}"] = f1_at_k(ranking, qrels, k, t)
    out["MAP"] = average_precision(ranking, qrels, t)
    for k in ks:
        out[f"nDCG@{k}"] = ndcg_at_k(ranking, qrels, k)
    out["RR"] = reciprocal_rank(ranking, qrels, t)
    out["R-prec"] = r_precision(ranking, qrels, t)
    for k in judged_ks:
        out[f"judged@{k}"] = judged_at_k(ranking, qrels, k)
    return out
