"""The 50-name phonetic test set and its evaluation.

Names are where phonetic matching earns its keep: one person's "Lakshmi" is
another's "Laxmi". We keep 50 Indian names, each with real spelling variants
(``names_testset.tsv``), and measure how well each matcher retrieves the
canonical name from a variant — accuracy@1 and MRR, the names counterpart to the
Aksharantar word-level test.

    python -m dhvani.query.names               # uses the shipped edit_costs.json
    python -m dhvani.query.names <costs.json>
"""

import argparse
import os

from dhvani.query import match as M
from dhvani.query.kgram import KGramIndex

NAMES_TSV = os.path.join(os.path.dirname(__file__), "names_testset.tsv")
EDIT_COSTS = os.path.join(os.path.dirname(__file__), "edit_costs.json")


def load_names(path=NAMES_TSV):
    """Return ``[(canonical, [variant, ...]), ...]`` (variants exclude the canonical)."""
    names = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            canonical, rest = line.split("\t", 1)
            variants = [v.strip() for v in rest.split(",") if v.strip() and v.strip() != canonical]
            if canonical and variants:
                names.append((canonical, variants))
    return names


def evaluate(names, costs, pool=50, k=10):
    """Return ``{matcher: (accuracy@1, MRR)}`` over every variant query."""
    vocab = [canonical for canonical, _ in names]
    index = KGramIndex(vocab, k=2)
    results = {}
    for matcher in M.MATCHERS:
        hits, rr, n = 0, 0.0, 0
        for canonical, variants in names:
            for variant in variants:
                ranked = M._ranked(variant, index, matcher, costs=costs, pool=pool)[:k]
                n += 1
                for pos, (term, _score) in enumerate(ranked, start=1):
                    if term == canonical:
                        if pos == 1:
                            hits += 1
                        rr += 1.0 / pos
                        break
        results[matcher] = (hits / n if n else 0.0, rr / n if n else 0.0)
    return results


def main(argv=None):
    from dhvani.query.editdist import load_costs

    ap = argparse.ArgumentParser(description="Evaluate matchers on the 50-name set.")
    ap.add_argument("costs", nargs="?", default=EDIT_COSTS)
    args = ap.parse_args(argv)

    names = load_names()
    costs = load_costs(args.costs)
    total = sum(len(v) for _, v in names)
    print(f"{len(names)} names, {total} variant queries\n")
    results = evaluate(names, costs)
    print(f"{'matcher':<14}{'accuracy@1':>12}{'MRR':>8}")
    print("-" * 34)
    for matcher in M.MATCHERS:
        acc, mrr = results[matcher]
        print(f"{matcher:<14}{acc:>12.3f}{mrr:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
