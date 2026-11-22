"""Rerank a run with a prior (T18) and write one new run per lambda into runs/.

    python scripts/run_rerank.py --run runs/bm25_k1.2_b0.75_dev.run --prior pagerank
    python scripts/run_rerank.py --run runs/rrf_bm25+dense_dev.run --prior ppr --lam 0.2
    python scripts/run_rerank.py --run runs/X_dev.run --prior trust

Priors:
    pagerank  global PageRank of graph_edges.tsv, as a percentile in [0, 1]
    ppr       personalized PageRank per query, seeded from that query's top results
    trust     reviewer-trust phone quality from reviews.jsonl, in [0, 1]

Output names follow CONTRACT.md 4.6, e.g. bm25_k1.2_b0.75+pagerank0.2_dev. Evaluate them
like any other run. Tune lambda on dev runs, then apply the chosen value to test once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phonesense import io
from phonesense.config import get_config, get_path
from phonesense.graph.pagerank import load_graph, pagerank_scores
from phonesense.graph.rerank import ppr_rerank, rerank_with_prior, reranked_run_name
from phonesense.trust.trust import trust_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--prior", choices=("pagerank", "ppr", "trust"), required=True)
    ap.add_argument("--lam", type=float, nargs="+", default=None,
                    help="one or more lambdas (default: rerank.lambdas in config.yaml)")
    ap.add_argument("--graph", type=Path, default=None, help="default: graph_edges path")
    ap.add_argument("--reviews", type=Path, default=None, help="default: reviews path")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: runs_dir")
    ap.add_argument("--damping", type=float, default=None,
                    help="PageRank damping for pagerank/ppr (default: graph.damping)")
    ap.add_argument("--tag", default=None,
                    help="run-name tag before lambda (default: the prior name), e.g. knnpr")
    args = ap.parse_args()

    cfg = get_config()
    g, t, r = cfg["graph"], cfg["trust"], cfg["rerank"]
    lambdas = args.lam if args.lam is not None else r["lambdas"]
    damping = args.damping if args.damping is not None else g["damping"]
    entries = io.read_run_entries(args.run)
    if not entries:
        raise SystemExit(f"{args.run} is empty")
    base_name = entries[0].run_name
    rankings = io.read_run(args.run)
    candidates = {d for ranking in rankings.values() for d, _ in ranking}
    out_dir = args.out_dir or get_path("runs_dir")

    prior = graph = None
    if args.prior == "pagerank":
        phones = get_path("phones")
        doc_ids = {p.doc_id for p in io.read_phones(phones)} if phones.exists() else candidates
        prior = pagerank_scores(doc_ids, damping=damping, tol=g["tol"], path=args.graph)
    elif args.prior == "ppr":
        graph = load_graph(args.graph)
    else:
        reviews = io.read_reviews(args.reviews or get_path("reviews"))
        prior = trust_scores(reviews, prior_weight=t["bayesian_prior_weight"],
                             prior_strength=t["prior_strength"],
                             prior_reliability=t["prior_reliability"],
                             max_iterations=t["max_iterations"], tol=t["tol"])
    if prior is not None:
        covered = len(candidates & set(prior)) / len(candidates)
        print(f"{args.prior} prior covers {covered:.1%} of the run's phones "
              "(the rest get the prior's median)")

    for lam in lambdas:
        if args.prior == "ppr":
            out = ppr_rerank(rankings, graph, lam, n_seeds=r["ppr_seeds"],
                             damping=damping, tol=g["tol"])
        else:
            out = rerank_with_prior(rankings, prior, lam)
        name = reranked_run_name(base_name, f"{args.tag or args.prior}{lam:g}")
        path = io.write_rankings(out_dir / f"{name}.run", out, name)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
