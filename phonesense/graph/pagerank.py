"""PageRank by power iteration (T17). Our own implementation: no PageRank library is used.

Update rule (lecture form, damping d = 1 - epsilon):

    pi_new[v] = d * (sum over edges u->v of pi[u] * w(u,v) / outweight(u)
                     + dangling_mass * t[v]) + (1 - d) * t[v]

t is the teleport vector (uniform for global PageRank). Rank sitting on dangling nodes
(no out-edges) is spread according to t: the lecture's "replace the zero column" fix.
Iteration stops when the L1 change drops below ``tol``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    """Directed weighted graph over doc_ids, stored as parallel edge arrays."""

    doc_ids: list[str]
    src: np.ndarray
    dst: np.ndarray
    w: np.ndarray

    @classmethod
    def from_edges(cls, doc_ids: Iterable[str], edges: Iterable[Sequence]) -> Graph:
        """edges: (src, dst) or (src, dst, weight) tuples over doc_ids.
        Duplicate edges are merged by summing weights; self-loops are dropped."""
        doc_ids = list(doc_ids)
        index = {d: i for i, d in enumerate(doc_ids)}
        if len(index) != len(doc_ids):
            raise ValueError("duplicate doc_ids")
        merged: dict[tuple[int, int], float] = {}
        for e in edges:
            a, b = index[e[0]], index[e[1]]
            weight = float(e[2]) if len(e) > 2 else 1.0
            if not weight > 0:
                raise ValueError(f"edge weight must be > 0: {tuple(e)}")
            if a != b:
                merged[(a, b)] = merged.get((a, b), 0.0) + weight
        keys = sorted(merged)
        return cls(
            doc_ids=doc_ids,
            src=np.array([k[0] for k in keys], dtype=np.int64),
            dst=np.array([k[1] for k in keys], dtype=np.int64),
            w=np.array([merged[k] for k in keys], dtype=float),
        )


def power_iteration(n, src, dst, w=None, damping=0.85, teleport=None, tol=1e-8, max_iter=500):
    """Return (pi, iterations). pi is a length-n array that sums to 1."""
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be in [0, 1]")
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.ones(len(src)) if w is None else np.asarray(w, dtype=float)
    if teleport is None:
        t = np.full(n, 1.0 / n)
    else:
        t = np.asarray(teleport, dtype=float)
        if t.shape != (n,) or (t < 0).any() or t.sum() <= 0:
            raise ValueError("teleport must be a non-negative length-n vector with a positive sum")
        t = t / t.sum()
    out_w = np.bincount(src, weights=w, minlength=n)
    dangling = out_w == 0
    share = w / out_w[src]  # each edge's fraction of its source's out-weight
    pi = t.copy()
    iterations = 0
    while iterations < max_iter:
        iterations += 1
        flow = np.bincount(dst, weights=pi[src] * share, minlength=n)
        new = damping * (flow + pi[dangling].sum() * t) + (1 - damping) * t
        err = np.abs(new - pi).sum()
        pi = new
        if err < tol:
            break
    return pi, iterations


def pagerank(graph: Graph, damping: float = 0.85, tol: float = 1e-8) -> dict[str, float]:
    """Global PageRank. Values sum to 1."""
    pi, _ = power_iteration(len(graph.doc_ids), graph.src, graph.dst, graph.w,
                            damping=damping, tol=tol)
    return dict(zip(graph.doc_ids, pi.tolist(), strict=True))


def personalized_pagerank(graph: Graph, seeds: dict[str, float], damping: float = 0.85,
                          tol: float = 1e-8) -> dict[str, float]:
    """PageRank whose teleport vector is the normalised seed scores (e.g. the top fused
    hits for a query). Phones unreachable from the seeds score 0. Values sum to 1."""
    index = {d: i for i, d in enumerate(graph.doc_ids)}
    unknown = [d for d in seeds if d not in index]
    if unknown:
        raise KeyError(f"seed doc_ids not in graph: {unknown[:5]}")
    t = np.zeros(len(graph.doc_ids))
    for doc_id, score in seeds.items():
        t[index[doc_id]] = score
    pi, _ = power_iteration(len(graph.doc_ids), graph.src, graph.dst, graph.w,
                            damping=damping, teleport=t, tol=tol)
    return dict(zip(graph.doc_ids, pi.tolist(), strict=True))


def scale_unit(scores: dict[str, float], method: str = "percentile") -> dict[str, float]:
    """Map scores to [0, 1]. 'percentile': share of phones scoring <= this one (ties equal),
    robust to PageRank's heavy tail. 'max': divide by the largest score."""
    doc_ids = list(scores)
    x = np.array([scores[d] for d in doc_ids], dtype=float)
    if method == "percentile":
        y = np.searchsorted(np.sort(x), x, side="right") / len(x)
    elif method == "max":
        y = x / x.max()
    else:
        raise ValueError(f"unknown scaling method: {method!r}")
    return dict(zip(doc_ids, y.tolist(), strict=True))
