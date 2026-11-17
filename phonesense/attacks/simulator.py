"""Review attack simulator (T20), after Lecture 9's reputation-system attacks.

bad_mouth     fake 1-star reviews on a good phone (a competitor's attack)
ballot_stuff  fake 5-star reviews on a weak phone (self-promotion)

Every fake review comes from a brand-new account (a sybil) whose user_id starts with
``SYN_`` and has ``is_synthetic=True`` (CONTRACT.md 4.2), so injected reviews can never be
mistaken for real data. ``inject`` never modifies its input.

With ``camouflage=k`` each sybil first posts k reviews on other phones that copy the current
consensus (rounded raw average), to look reliable before it attacks. This is the attacker
that a reviewer-trust score should struggle against.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from phonesense.schemas import Review, make_review_id
from phonesense.trust.baselines import raw_average

ATTACK_RATING = {"bad_mouth": 1, "ballot_stuff": 5}


def _fake_review(user_id: str, doc_id: str, rating: int, timestamp: int) -> Review:
    return Review(review_id=make_review_id(user_id, doc_id, timestamp), doc_id=doc_id,
                  user_id=user_id, rating=rating, title="", text="", timestamp=timestamp,
                  helpful_votes=0, verified=False, is_synthetic=True)


def inject(reviews: Sequence[Review], target: str, kind: str, n: int, seed: int,
           camouflage: int = 0) -> list[Review]:
    """Return a NEW list: the original reviews followed by the fake ones. ``n`` sybils each
    post one attack review on ``target`` (plus ``camouflage`` cover reviews elsewhere)."""
    if kind not in ATTACK_RATING:
        raise ValueError(f"unknown attack kind {kind!r}; use 'bad_mouth' or 'ballot_stuff'")
    if n < 0 or camouflage < 0:
        raise ValueError("n and camouflage must be >= 0")
    if not any(r.doc_id == target for r in reviews):
        raise ValueError(f"target {target!r} has no reviews")
    out = list(reviews)
    ts = max(r.timestamp for r in reviews)
    if camouflage and n:
        rng = np.random.default_rng(seed)
        consensus = raw_average(reviews)
        others = sorted(d for d in consensus if d != target)
        k = min(camouflage, len(others))
    for i in range(n):
        user_id = f"SYN_{kind}_{target}_{seed}_{i}"
        if camouflage:
            for doc_id in rng.choice(others, size=k, replace=False):
                ts += 1000
                rating = int(min(5, max(1, round(consensus[str(doc_id)]))))
                out.append(_fake_review(user_id, str(doc_id), rating, ts))
        ts += 1000
        out.append(_fake_review(user_id, target, ATTACK_RATING[kind], ts))
    return out
