"""One attack, explained (T25): the logic behind the demo's attack tab, with no UI code so it
can be tested and reused.

``attack_report`` injects fake reviews on one phone and returns, for every reputation method,
the phone's score and rank before and after, plus each method's top phones after the attack
and how reliable the trust score judges the attacking accounts to be.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from phonesense.attacks.simulator import attack_size, inject, rank_of
from phonesense.schemas import Review
from phonesense.trust.methods import configured_trust, reputation_methods


@dataclass
class AttackReport:
    target: str
    kind: str
    n_fake: int                      # attack reviews (one per sybil account)
    n_cover: int                     # camouflage reviews posted on other phones
    n_phones: int
    rows: list[dict]                 # one per method, in method order
    leaderboards: dict[str, list[tuple[str, float]]]  # after the attack, best first
    sybil_reliability: float | None  # mean trust reliability of the attacking accounts


def clean_scores(reviews: Sequence[Review]) -> dict[str, dict[str, float]]:
    """Every method's scores on the unattacked reviews (worth caching in a UI)."""
    return {name: fn(reviews) for name, fn in reputation_methods().items()}


def attack_report(reviews: Sequence[Review], target: str, kind: str, percent: float,
                  seed: int = 1, camouflage: int = 0,
                  before: Mapping[str, Mapping[str, float]] | None = None,
                  top_n: int = 10) -> AttackReport:
    """Attack ``target`` with ``percent``% of its review count in fake reviews.

    ``rank_gain`` is signed so that a positive value means the attacker succeeded: the phone
    dropped (bad-mouthing) or rose (ballot stuffing) by that many places."""
    before = before if before is not None else clean_scores(reviews)
    n = attack_size(reviews, target, percent)
    attacked = inject(reviews, target, kind, n, seed, camouflage=camouflage)
    sign = 1 if kind == "bad_mouth" else -1

    trust = configured_trust(attacked)
    after = {name: (trust.quality if name == "trust" else fn(attacked))
             for name, fn in reputation_methods().items()}
    rows = []
    for name in after:
        rb, ra = rank_of(before[name], target), rank_of(after[name], target)
        rows.append({
            "method": name,
            "score_before": before[name][target], "score_after": after[name][target],
            "rank_before": rb, "rank_after": ra,
            "rank_gain": sign * (ra - rb),
            "score_gain": sign * (before[name][target] - after[name][target]),
        })
    leaderboards = {name: sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]
                    for name, scores in after.items()}
    sybils = {r.user_id for r in attacked[len(reviews):]}
    sybil_rel = (sum(trust.reliability[u] for u in sybils) / len(sybils)) if sybils else None
    return AttackReport(target=target, kind=kind, n_fake=n,
                        n_cover=len(attacked) - len(reviews) - n, n_phones=len(after["raw"]),
                        rows=rows, leaderboards=leaderboards, sybil_reliability=sybil_rel)
