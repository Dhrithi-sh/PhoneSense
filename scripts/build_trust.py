"""Compute reviewer trust (T19) and write data/processed/reviewer_trust.tsv.

    python scripts/build_trust.py           # reads data/processed/reviews.jsonl
    python scripts/build_trust.py --toy     # seeded toy reviews, writes nothing to data/

Also writes results/trust_stats.json: how many reviewers have a single review (the trust
score can only learn from repeat reviewers), iterations, and how far trust moves phones
away from the Bayesian average.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from phonesense import io
from phonesense.config import get_path
from phonesense.trust.baselines import bayesian_average
from phonesense.trust.methods import trust_params
from phonesense.trust.toy import make_toy_reviews
from phonesense.trust.trust import compute_trust


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--toy", action="store_true", help="use seeded toy reviews; write no TSV")
    args = ap.parse_args()

    if args.toy:
        reviews = make_toy_reviews(n_phones=300, n_users=4000)
    else:
        path = get_path("reviews")
        if not path.exists():
            raise SystemExit(f"{path} not found. Run scripts/build_data.py first (or --toy).")
        reviews = io.read_reviews(path)

    params = trust_params()
    res = compute_trust(reviews, **params)
    per_user = Counter(r.user_id for r in reviews)
    bay = bayesian_average(reviews, prior_weight=params["prior_weight"])
    shift = np.array([res.quality[d] - bay[d] for d in bay])
    rel = np.array(list(res.reliability.values()))
    stats = {
        "source": "toy" if args.toy else str(get_path("reviews")),
        "reviews": len(reviews),
        "phones": len(res.quality),
        "reviewers": len(per_user),
        "share_single_review_reviewers": round(sum(c == 1 for c in per_user.values())
                                               / len(per_user), 4),
        "iterations": res.iterations,
        "converged": res.converged,
        "reliability_quantiles": {q: round(float(np.quantile(rel, q)), 4)
                                  for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "quality_minus_bayesian_abs_mean": round(float(np.abs(shift).mean()), 4),
        "quality_minus_bayesian_abs_max": round(float(np.abs(shift).max()), 4),
        "params": params,
    }
    print(json.dumps(stats, indent=2))
    if args.toy:
        return
    io.write_reviewer_trust(get_path("reviewer_trust"), res.reliability)
    out = get_path("results_dir") / "trust_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {get_path('reviewer_trust')} and {out}")


if __name__ == "__main__":
    main()
