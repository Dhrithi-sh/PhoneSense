"""Reputation-only attack experiment (T20 sanity run; T21 adds full-pipeline nDCG).

For each attack kind, seed, target and attack size, inject fake reviews and record how far
the target's score and rank move under each reputation method. Positive shifts always mean
"the attacker gained".

    python scripts/run_attacks.py                   # reads data/processed/reviews.jsonl
    python scripts/run_attacks.py --camouflage 5    # sybils copy the consensus first
    python scripts/run_attacks.py --toy --seeds 1   # seeded toy reviews, quick dry run

Writes results/attacks/<name>.csv and report/figures/<name>.png, where <name> is
attack_rank_shift (plus _camo<k>). --toy writes both files to results/attacks/ instead.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from phonesense import io  # noqa: E402
from phonesense.attacks.simulator import attack_size, inject, measure, pick_targets  # noqa: E402
from phonesense.config import get_config, get_path  # noqa: E402
from phonesense.trust.baselines import (  # noqa: E402
    bayesian_average,
    helpful_weighted_average,
    median_rating,
    raw_average,
)
from phonesense.trust.toy import make_toy_reviews  # noqa: E402
from phonesense.trust.trust import compute_trust  # noqa: E402

KINDS = ("bad_mouth", "ballot_stuff")


def reputation_methods() -> dict:
    t = get_config()["trust"]
    return {
        "raw": raw_average,
        "bayesian": lambda rs: bayesian_average(rs, prior_weight=t["bayesian_prior_weight"]),
        "helpful": helpful_weighted_average,
        "median": median_rating,
        "trust": lambda rs: compute_trust(
            rs, prior_weight=t["bayesian_prior_weight"], prior_strength=t["prior_strength"],
            prior_reliability=t["prior_reliability"], max_iterations=t["max_iterations"],
            tol=t["tol"]).quality,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--toy", action="store_true", help="use seeded toy reviews")
    ap.add_argument("--seeds", type=int, default=None, help="use only the first N seeds")
    ap.add_argument("--camouflage", type=int, default=0)
    args = ap.parse_args()

    cfg = get_config()["attacks"]
    sizes = cfg["sizes_percent"]
    seeds = cfg["seeds"][:args.seeds] if args.seeds else cfg["seeds"]
    if args.toy:
        reviews = make_toy_reviews(n_phones=300, n_users=4000)
    else:
        path = get_path("reviews")
        if not path.exists():
            raise SystemExit(f"{path} not found. Run scripts/build_data.py first (or --toy).")
        reviews = io.read_reviews(path)

    methods = reputation_methods()
    before = {name: fn(reviews) for name, fn in methods.items()}
    print(f"{len(reviews)} reviews, {len(before['raw'])} phones, sizes {sizes}, seeds {seeds}")

    rows = []
    for kind in KINDS:
        sign = 1 if kind == "bad_mouth" else -1
        for seed in seeds:
            for target in pick_targets(reviews, kind, cfg["targets_per_kind"], seed,
                                       min_reviews=cfg["min_target_reviews"]):
                for pct in sizes:
                    n = attack_size(reviews, target, pct)
                    attacked = inject(reviews, target, kind, n, seed, camouflage=args.camouflage)
                    for name, m in measure(reviews, attacked, target, methods, before).items():
                        rows.append({
                            "kind": kind, "seed": seed, "target": target, "percent": pct,
                            "n_fake": n, "camouflage": args.camouflage, "method": name,
                            "score_shift": sign * (m["score_before"] - m["score_after"]),
                            "rank_shift": sign * (m["rank_after"] - m["rank_before"]),
                        })

    name = ("toy_" if args.toy else "") + "attack_rank_shift"
    name += f"_camo{args.camouflage}" if args.camouflage else ""
    csv_path = get_path("results_dir") / "attacks" / f"{name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cells: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        cells[(r["kind"], r["method"], r["percent"])].append((r["score_shift"], r["rank_shift"]))

    def mean(kind, method, pct, i):
        xs = cells[(kind, method, pct)]
        return sum(x[i] for x in xs) / len(xs)

    for kind in KINDS:
        print(f"\n{kind}: mean rank positions gained by the attacker (mean score shift, stars)")
        print("percent " + "".join(f"{m:>17}" for m in methods))
        for pct in sizes:
            print(f"{pct:>7} " + "".join(
                f"{mean(kind, m, pct, 1):>9.1f} ({mean(kind, m, pct, 0):.2f})" for m in methods))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kind in zip(axes, KINDS, strict=True):
        for m in methods:
            ax.plot(sizes, [mean(kind, m, pct, 1) for pct in sizes], marker="o", label=m)
        title = kind + (f" (camouflage {args.camouflage})" if args.camouflage else "")
        ax.set_title(title + (" [TOY DATA]" if args.toy else ""))
        ax.set_xlabel("attack size (% of the target's reviews)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean rank positions gained by attacker")
    axes[1].legend(title="reputation method")
    fig.tight_layout()
    # toy figures stay in git-ignored results/, so they never reach the report
    fig_dir = csv_path.parent if args.toy else get_path("figures_dir")
    fig_path = fig_dir / f"{name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"\nwrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
