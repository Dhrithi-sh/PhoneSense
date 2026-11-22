"""Reviewer-trust score (T19): our own HITS-inspired design.

Phone quality and reviewer reliability reinforce each other across the reviewer-phone
bipartite graph, like hubs and authorities:

    quality      q_d = (C*m + sum_u r_u * rating(u,d)) / (C + sum_u r_u)
    agreement    a_u = 1 - mean_d |rating(u,d) - q_d| / 4
    reliability  r_u = (n_u * a_u + c * r0) / (n_u + c)

m = global mean rating, C = phone prior weight, n_u = number of reviews by u, c and r0 =
reviewer prior strength and prior reliability (a brand-new account starts near r0).
Starting from r_u = 1, the first quality estimate is exactly the Bayesian average.
Convergence is observed, not proven, so the iteration count is returned and capped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phonesense.schemas import Review
from phonesense.trust.baselines import to_unit


@dataclass
class TrustResult:
    quality: dict[str, float]      # doc_id -> 1..5 stars
    reliability: dict[str, float]  # user_id -> [0, 1]
    iterations: int
    converged: bool


def compute_trust(reviews: Sequence[Review], prior_weight: float = 10.0,
                  prior_strength: float = 3.0, prior_reliability: float = 0.5,
                  max_iterations: int = 50, tol: float = 1e-6) -> TrustResult:
    if not reviews:
        raise ValueError("no reviews")
    doc_ids = sorted({r.doc_id for r in reviews})
    user_ids = sorted({r.user_id for r in reviews})
    d_index = {d: i for i, d in enumerate(doc_ids)}
    u_index = {u: i for i, u in enumerate(user_ids)}
    d = np.array([d_index[r.doc_id] for r in reviews], dtype=np.int64)
    u = np.array([u_index[r.user_id] for r in reviews], dtype=np.int64)
    x = np.array([r.rating for r in reviews], dtype=float)
    n_docs, n_users = len(doc_ids), len(user_ids)

    m = x.mean()
    n_u = np.bincount(u, minlength=n_users)
    rel = np.ones(n_users)
    q = np.full(n_docs, np.nan)
    converged = False
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        w_sum = np.bincount(d, weights=rel[u], minlength=n_docs)
        x_sum = np.bincount(d, weights=rel[u] * x, minlength=n_docs)
        q_new = (prior_weight * m + x_sum) / (prior_weight + w_sum)
        deviation = np.bincount(u, weights=np.abs(x - q_new[d]) / 4.0, minlength=n_users)
        agreement = 1.0 - deviation / n_u
        rel = (n_u * agreement + prior_strength * prior_reliability) / (n_u + prior_strength)
        done = np.abs(q_new - q).max() < tol if iterations > 1 else False
        q = q_new
        if done:
            converged = True
            break
    return TrustResult(quality=dict(zip(doc_ids, q.tolist(), strict=True)),
                       reliability=dict(zip(user_ids, rel.tolist(), strict=True)),
                       iterations=iterations, converged=converged)


def trust_scores(reviews: Sequence[Review], **params) -> dict[str, float]:
    """Trust-weighted phone quality in [0, 1], for use as a ranking prior."""
    return to_unit(compute_trust(reviews, **params).quality)
