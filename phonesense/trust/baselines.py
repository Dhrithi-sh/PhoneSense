"""Reputation baselines (T19): raw average, Bayesian average, helpful-vote weighted average,
median. The reviewer-trust score in trust.py is compared against these.

Every function takes a list of ``Review`` (CONTRACT.md 4.2) and returns {doc_id: score} on
the 1..5 star scale. ``to_unit`` maps that to [0, 1].
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence

from phonesense.schemas import Review


def to_unit(scores: dict[str, float]) -> dict[str, float]:
    """Map 1..5 stars to [0, 1]."""
    return {d: (s - 1.0) / 4.0 for d, s in scores.items()}


def _ratings_by_doc(reviews: Sequence[Review]) -> dict[str, list[int]]:
    by_doc: dict[str, list[int]] = defaultdict(list)
    for r in reviews:
        by_doc[r.doc_id].append(r.rating)
    return by_doc


def global_mean(reviews: Sequence[Review]) -> float:
    if not reviews:
        raise ValueError("no reviews")
    return sum(r.rating for r in reviews) / len(reviews)


def raw_average(reviews: Sequence[Review]) -> dict[str, float]:
    return {d: sum(x) / len(x) for d, x in _ratings_by_doc(reviews).items()}


def bayesian_average(reviews: Sequence[Review], prior_weight: float = 10.0) -> dict[str, float]:
    """(C*m + sum of ratings) / (C + n): m = global mean rating, C = prior_weight."""
    m = global_mean(reviews)
    return {d: (prior_weight * m + sum(x)) / (prior_weight + len(x))
            for d, x in _ratings_by_doc(reviews).items()}


def helpful_weighted_average(reviews: Sequence[Review]) -> dict[str, float]:
    """Weighted mean rating, weight = 1 + log(1 + helpful_votes)."""
    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    for r in reviews:
        weight = 1.0 + math.log1p(r.helpful_votes)
        num[r.doc_id] += weight * r.rating
        den[r.doc_id] += weight
    return {d: num[d] / den[d] for d in num}


def median_rating(reviews: Sequence[Review]) -> dict[str, float]:
    return {d: float(statistics.median(x)) for d, x in _ratings_by_doc(reviews).items()}
