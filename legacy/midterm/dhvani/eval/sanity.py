"""Sanity checks that need no relevance judgments.

The rubric allows "a clear sanity check for a partial system". These numbers
come straight from the run files on the frozen corpus:

1. Cross-form agreement. Every information need is written four ways
   (Hindi, Hinglish, messy Hinglish, English). If the system really handles
   Hinglish and English, all four should find much the same articles as the
   Hindi form. For each need we take the Hindi form's top k as the reference
   and measure how much of it each other form also returns (overlap@k). The
   same is shown with translation switched off, which isolates what the
   cross-lingual layer adds for English queries.

2. System agreement. For each pair of rankers (and of stemming modes) on the
   same queries: overlap@k of their top k, and Kendall's tau on the order of
   the articles both returned. High agreement means the systems mostly differ
   in order; low agreement means they really find different things, so the
   judgments will be needed to say which is better.

Run it after `python -m dhvani.eval.experiments --out data/eval/full`:

    python -m dhvani.eval.sanity --runs data/eval/full/runs
"""

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from dhvani.eval.metrics import read_run

REFERENCE_FORM = "hindi"


def overlap(a, b, k=10):
    """Share of a's top k that is also in b's top k."""
    top = a[:k]
    if not top:
        return None
    return len(set(top) & set(b[:k])) / len(top)


def kendall_tau(a, b):
    """Kendall's tau between the orders of the items both lists contain."""
    shared = [d for d in a if d in set(b)]
    n = len(shared)
    if n < 2:
        return None
    pos = {d: i for i, d in enumerate(b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos[shared[i]] < pos[shared[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (n * (n - 1) / 2)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def cross_form_agreement(run, queries, k=10, reference=REFERENCE_FORM):
    """{form: mean overlap@k with the reference form of the same need}."""
    by_need = defaultdict(dict)
    for qid, need, form, _text in queries:
        by_need[need][form] = qid
    scores = defaultdict(list)
    for forms in by_need.values():
        ref = forms.get(reference)
        if not ref or not run.get(ref):
            continue
        for form, qid in forms.items():
            if form != reference:
                scores[form].append(overlap(run[ref], run.get(qid, []), k))
    return {form: _mean(vals) for form, vals in scores.items()}


def system_agreement(runs, k=10):
    """[(name_a, name_b, mean overlap@k, mean Kendall tau)] for every pair of runs."""
    rows = []
    for a, b in combinations(sorted(runs), 2):
        qids = set(runs[a]) & set(runs[b])
        ov = _mean(overlap(runs[a][q], runs[b][q], k) for q in qids)
        tau = _mean(kendall_tau(runs[a][q][:k], runs[b][q][:k]) for q in qids)
        rows.append((a, b, ov, tau))
    return rows


def load_runs(runs_dir):
    """{"mode_ranker": {qid: [doc_ids]}} from a run directory."""
    return {p.stem: read_run(p) for p in sorted(Path(runs_dir).glob("*.txt"))}


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def main(argv=None):
    from dhvani.eval.experiments import build_query_no_xling, read_needs, run_one
    from dhvani.rank.real_index import load_index

    parser = argparse.ArgumentParser(description="Sanity checks without judgments.")
    parser.add_argument("--runs", default="data/eval/full/runs")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args(argv)

    queries = read_needs()
    runs = load_runs(args.runs)
    if not runs:
        print(f"No run files in {args.runs}; run python -m dhvani.eval.experiments --out data/eval/full first.")
        return

    print(f"1. Cross-form agreement: overlap@{args.k} with the Hindi form of the same need")
    forms = ["hinglish", "messy", "english"]
    print(f"{'run':<14}" + "".join(f"{f:>10}" for f in forms))
    for name in sorted(runs):
        agree = cross_form_agreement(runs[name], queries, args.k)
        print(f"{name:<14}" + "".join(f"{_fmt(agree.get(f)):>10}" for f in forms))

    english = [q for q in queries if q[2] in (REFERENCE_FORM, "english")]
    off = run_one(load_index("none"), english, "net", k=args.k, query_builder=build_query_no_xling, mode="none")
    off = {q: [d for d, _ in r] for q, r in off.items()}
    on = cross_form_agreement(runs.get("none_net", {}), queries, args.k).get("english")
    without = cross_form_agreement(off, queries, args.k).get("english")
    print(f"\nEnglish vs Hindi form (none_net): translation on {_fmt(on)}, translation off {_fmt(without)}")

    print(f"\n2. System agreement (overlap@{args.k}, Kendall tau on shared articles)")
    for group, pick in (("rankers, no stemming", lambda n: n.startswith("none_")),
                        ("stemming modes, net score", lambda n: n.endswith("_net"))):
        print(f"\n{group}")
        subset = {n: r for n, r in runs.items() if pick(n)}
        for a, b, ov, tau in system_agreement(subset, args.k):
            print(f"  {a:<12} vs {b:<12} overlap {_fmt(ov)}  tau {_fmt(tau)}")


if __name__ == "__main__":
    main()
