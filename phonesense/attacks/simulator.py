"""Review attack simulator (T20), after Lecture 9's reputation-system attacks.

bad_mouth     fake 1-star reviews on a good phone (a competitor's attack)
ballot_stuff  fake 5-star reviews on a weak phone (self-promotion)

Every fake review comes from a brand-new account (a sybil) whose user_id starts with
``SYN_`` and has ``is_synthetic=True`` (CONTRACT.md 4.2), so injected reviews can never be
mistaken for real data. ``inject`` never modifies its input.
"""

from __future__ import annotations

from collections.abc import Sequence

from phonesense.schemas import Review, make_review_id

ATTACK_RATING = {"bad_mouth": 1, "ballot_stuff": 5}


def _fake_review(user_id: str, doc_id: str, rating: int, timestamp: int) -> Review:
    return Review(review_id=make_review_id(user_id, doc_id, timestamp), doc_id=doc_id,
                  user_id=user_id, rating=rating, title="", text="", timestamp=timestamp,
                  helpful_votes=0, verified=False, is_synthetic=True)


def inject(reviews: Sequence[Review], target: str, kind: str, n: int, seed: int) -> list[Review]:
    """Return a NEW list: the original reviews followed by ``n`` fake ones on ``target``."""
    if kind not in ATTACK_RATING:
        raise ValueError(f"unknown attack kind {kind!r}; use 'bad_mouth' or 'ballot_stuff'")
    if n < 0:
        raise ValueError("n must be >= 0")
    if not any(r.doc_id == target for r in reviews):
        raise ValueError(f"target {target!r} has no reviews")
    out = list(reviews)
    ts = max(r.timestamp for r in reviews)
    for i in range(n):
        ts += 1000
        out.append(_fake_review(f"SYN_{kind}_{target}_{seed}_{i}", target, ATTACK_RATING[kind],
                                ts))
    return out
