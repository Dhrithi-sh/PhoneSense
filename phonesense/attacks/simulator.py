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

import math
from collections import Counter
from collections.abc import Callable, Sequence

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


def attack_size(reviews: Sequence[Review], target: str, percent: float) -> int:
    """Fake reviews for an attack of ``percent`` of the target's review count, rounded up."""
    return math.ceil(sum(r.doc_id == target for r in reviews) * percent / 100.0)


def pick_targets(reviews: Sequence[Review], kind: str, k: int, seed: int,
                 min_reviews: int = 20) -> list[str]:
    """bad_mouth: sample from the top quarter of phones by raw average; ballot_stuff: from the
    bottom quarter. Only phones with at least ``min_reviews`` reviews are eligible."""
    if kind not in ATTACK_RATING:
        raise ValueError(f"unknown attack kind {kind!r}")
    counts = Counter(r.doc_id for r in reviews)
    raw = raw_average(reviews)
    eligible = sorted((d for d in raw if counts[d] >= min_reviews), key=lambda d: (-raw[d], d))
    if not eligible:
        raise ValueError(f"no phone has at least {min_reviews} reviews")
    quarter = max(1, len(eligible) // 4)
    pool = eligible[:quarter] if kind == "bad_mouth" else eligible[-quarter:]
    rng = np.random.default_rng(seed)
    return sorted(str(d) for d in rng.choice(pool, size=min(k, len(pool)), replace=False))


def rank_of(scores: dict[str, float], doc_id: str) -> int:
    """1 = best. Ties are broken by doc_id ascending."""
    s = scores[doc_id]
    return 1 + sum(1 for d, v in scores.items() if v > s or (v == s and d < doc_id))


ReputationFn = Callable[[Sequence[Review]], dict[str, float]]


def measure(reviews: Sequence[Review], attacked: Sequence[Review], target: str,
            methods: dict[str, ReputationFn],
            before: dict[str, dict[str, float]] | None = None) -> dict[str, dict]:
    """Score and rank of ``target`` before and after the attack, per reputation method.
    Pass ``before`` ({method: scores on the clean reviews}) to avoid recomputing them."""
    out = {}
    for name, fn in methods.items():
        b = before[name] if before is not None else fn(reviews)
        a = fn(attacked)
        out[name] = {"score_before": b[target], "score_after": a[target],
                     "rank_before": rank_of(b, target), "rank_after": rank_of(a, target)}
    return out
