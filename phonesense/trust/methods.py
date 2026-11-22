"""The reputation methods compared in the attack experiments and the demo, with the trust
parameters read from the ``trust:`` block of config.yaml.

Each method maps a list of reviews to {doc_id: score in stars}.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from phonesense.config import get_config
from phonesense.schemas import Review
from phonesense.trust.baselines import (
    bayesian_average,
    helpful_weighted_average,
    median_rating,
    raw_average,
)
from phonesense.trust.trust import TrustResult, compute_trust

ReputationFn = Callable[[Sequence[Review]], dict[str, float]]


def trust_params() -> dict:
    """compute_trust keyword arguments from config.yaml."""
    t = get_config()["trust"]
    return {"prior_weight": t["bayesian_prior_weight"], "prior_strength": t["prior_strength"],
            "prior_reliability": t["prior_reliability"], "max_iterations": t["max_iterations"],
            "tol": t["tol"]}


def configured_trust(reviews: Sequence[Review]) -> TrustResult:
    return compute_trust(reviews, **trust_params())


def reputation_methods() -> dict[str, ReputationFn]:
    """name -> method, in the order used in tables and plots."""
    prior_weight = trust_params()["prior_weight"]
    return {
        "raw": raw_average,
        "bayesian": lambda rs: bayesian_average(rs, prior_weight=prior_weight),
        "helpful": helpful_weighted_average,
        "median": median_rating,
        "trust": lambda rs: configured_trust(rs).quality,
    }
