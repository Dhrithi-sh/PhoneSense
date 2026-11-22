"""Paired significance tests between runs (T11).

With ~20 queries per split a difference in a mean metric can easily be luck, so systems are
compared query by query on a chosen metric (paired):

* **Paired randomization (permutation) test**, two-sided (Smucker, Allan & Carterette,
  CIKM 2007, recommended for IR). Under H0 the two systems are exchangeable, so each
  per-query difference is equally likely to have either sign. We flip signs at random
  ``permutations`` times (default 10,000; seed from config.yaml) and report
  ``p = (#{|mean flipped diff| >= |observed mean diff|} + 1) / (permutations + 1)``.
* **Paired t-test** on the per-query differences (``scipy.stats.ttest_rel``), for comparison.
* **Holm-Bonferroni** step-down correction when one baseline is compared with m systems:
  sort the p-values ascending, ``p_adj(i) = max_{j<=i} min(1, (m - j) * p(j))`` (0-based j).

Only queries present in both per-query tables are compared.

Adapted from our midterm project Dhvani-lingual (``dhvani/eval/significance.py``): same
randomization test (now vectorized with numpy and seeded from config); the hand-written
incomplete-beta t-test was replaced by scipy; Holm correction and win/loss counts are new.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

PERMUTATIONS = 10_000
PerQuery = Mapping[str, Mapping[str, float]]  # {qid: {metric: value}}


def _seed() -> int:
    from phonesense.config import SEED

    return SEED


def paired_values(
    a: PerQuery, b: PerQuery, metric: str
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Shared qids (sorted) and the metric values of ``a`` and ``b`` on them."""
    qids = sorted(set(a) & set(b))
    missing = [q for q in qids if metric not in a[q] or metric not in b[q]]
    if missing:
        raise KeyError(f"metric {metric!r} missing for queries {missing[:5]}")
    return (
        qids,
        np.array([a[q][metric] for q in qids], dtype=float),
        np.array([b[q][metric] for q in qids], dtype=float),
    )


def randomization_test(
    x: Sequence[float],
    y: Sequence[float],
    permutations: int = PERMUTATIONS,
    seed: int | None = None,
) -> float:
    """Two-sided paired randomization test p-value for mean(x) - mean(y)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if d.size == 0:
        return 1.0
    observed = abs(d.mean())
    rng = np.random.default_rng(_seed() if seed is None else seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, d.size))
    flipped = np.abs((signs * d).mean(axis=1))
    at_least = int(np.count_nonzero(flipped >= observed - 1e-12))
    return (at_least + 1) / (permutations + 1)


def t_test(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Paired t-test ``(t, two-sided p)``; ``(0, 1)`` when undefined (n < 2 or no variance)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0, 1.0
    d = x - y
    if np.allclose(d, d[0]):
        if np.isclose(d[0], 0.0):
            return 0.0, 1.0
        return (float(np.sign(d[0]) * np.inf), 0.0)
    res = stats.ttest_rel(x, y)
    return float(res.statistic), float(res.pvalue)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def compare(
    baseline: PerQuery,
    system: PerQuery,
    metric: str,
    permutations: int = PERMUTATIONS,
    seed: int | None = None,
    tol: float = 1e-12,
) -> dict[str, Any]:
    """Compare ``system`` against ``baseline`` on ``metric`` (difference = system - baseline).

    Returns ``n``, both means, ``mean_diff``, ``p_randomization``, ``t``, ``p_ttest`` and the
    number of queries where the system is ``better`` / ``worse`` / ``tied``.
    """
    qids, b, s = paired_values(baseline, system, metric)
    d = s - b
    t, p_t = t_test(s, b)
    return {
        "metric": metric,
        "n": len(qids),
        "mean_baseline": float(b.mean()) if qids else 0.0,
        "mean_system": float(s.mean()) if qids else 0.0,
        "mean_diff": float(d.mean()) if qids else 0.0,
        "p_randomization": randomization_test(s, b, permutations, seed),
        "t": t,
        "p_ttest": p_t,
        "better": int(np.count_nonzero(d > tol)),
        "worse": int(np.count_nonzero(d < -tol)),
        "tied": int(np.count_nonzero(np.abs(d) <= tol)),
    }


def compare_many(
    baseline: PerQuery,
    systems: Mapping[str, PerQuery],
    metric: str,
    permutations: int = PERMUTATIONS,
    seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """:func:`compare` for each system, plus Holm-adjusted p-values over the systems.

    Adds ``p_randomization_holm`` and ``p_ttest_holm`` to every row.
    """
    rows = {name: compare(baseline, pq, metric, permutations, seed) for name, pq in systems.items()}
    names = list(rows)
    for key in ("p_randomization", "p_ttest"):
        adj = holm([rows[n][key] for n in names])
        for n, p in zip(names, adj, strict=True):
            rows[n][f"{key}_holm"] = p
    return rows
