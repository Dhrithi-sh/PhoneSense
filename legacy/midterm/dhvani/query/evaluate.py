"""Word-level evaluation of the four matchers on the Aksharantar test split.

For each test pair ``(native word, human romanisation)`` we treat the human
romanisation as the query and ask: does the matcher rank the correct Devanagari
``native word`` at the top of the vocabulary? We report **accuracy@1** and
**MRR** (mean reciprocal rank) for each of the four matchers — this is the
Phase-5 comparison table.

Run after exporting the data and learning the costs::

    python -m dhvani.query.aksharantar --max-train 50000 --max-test 5000
    python -m dhvani.query.editdist data/aksharantar/train_pairs.tsv data/edit_costs.json
    python -m dhvani.query.evaluate data/aksharantar/test_pairs.tsv data/edit_costs.json
"""

import argparse

from dhvani.query import match as M
from dhvani.query.aksharantar import load_pairs
from dhvani.query.editdist import load_costs
from dhvani.query.kgram import KGramIndex


def evaluate(test_pairs, index, costs, pool=50, k=10):
    """Return ``{matcher: (accuracy@1, mrr)}`` over ``test_pairs``."""
    results = {}
    for matcher in M.MATCHERS:
        hits, rr, n = 0, 0.0, 0
        for native, roman in test_pairs:
            ranked = M._ranked(roman, index, matcher, costs=costs, pool=pool)[:k]
            n += 1
            for rank_pos, (term, _score) in enumerate(ranked, start=1):
                if term == native:
                    if rank_pos == 1:
                        hits += 1
                    rr += 1.0 / rank_pos
                    break
        results[matcher] = (hits / n if n else 0.0, rr / n if n else 0.0)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate matchers on Aksharantar test split.")
    ap.add_argument("test_pairs", help="data/aksharantar/test_pairs.tsv")
    ap.add_argument("costs", help="data/edit_costs.json from editdist")
    ap.add_argument("--pool", type=int, default=50, help="k-gram candidate pool size")
    ap.add_argument("--k", type=int, default=10, help="cutoff for MRR / top-k")
    ap.add_argument("--limit", type=int, default=2000, help="max test queries (speed)")
    args = ap.parse_args(argv)

    pairs = load_pairs(args.test_pairs)[: args.limit]
    vocab = sorted({native for native, _ in pairs})
    index = KGramIndex(vocab, k=2)
    costs = load_costs(args.costs)

    print(f"{len(pairs)} test queries over a {len(vocab)}-word vocabulary\n")
    results = evaluate(pairs, index, costs, pool=args.pool, k=args.k)
    print(f"{'matcher':<14}{'accuracy@1':>12}{'MRR':>8}")
    print("-" * 34)
    for matcher in M.MATCHERS:
        acc, mrr = results[matcher]
        print(f"{matcher:<14}{acc:>12.3f}{mrr:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
