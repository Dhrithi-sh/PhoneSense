"""Rocchio pseudo-relevance-feedback runs on top of TF-IDF (T14).

    python scripts/run_rocchio.py --split dev                 # -> runs/rocchio_..._dev.run
    python scripts/run_rocchio.py --split dev --m 5 --beta 0.5 --n-terms 30
    python scripts/run_rocchio.py --query "phone with great night camera"   # top 10 + terms
    python scripts/run_rocchio.py --split dev --phones tests/fixtures/phones.jsonl  # no index

By default it uses the saved index from scripts/build_index.py (``--sample`` for the sample
index). With ``--phones`` it builds a private in-memory index from that file instead.
Parameters are tuned on dev queries only (CONTRACT.md, evaluation rules).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phonesense.config import get_path
from phonesense.index import InvertedIndex, index_path
from phonesense.io import read_phones, read_queries
from phonesense.retrieval.base import run_retriever
from phonesense.retrieval.rocchio import RocchioRetriever
from phonesense.retrieval.tfidf import TfidfRetriever


def build_retriever(args: argparse.Namespace) -> RocchioRetriever | None:
    base = TfidfRetriever()
    if args.phones:
        base.fit(read_phones(args.phones))
    else:
        path = index_path(args.sample)
        if not (path / "meta.json").exists():
            flag = " --sample" if args.sample else ""
            print(f"no index at {path}. Build it first: python scripts/build_index.py{flag}",
                  file=sys.stderr)
            return None
        base.fit_index(InvertedIndex.load(path))
    return RocchioRetriever(
        base, m=args.m, alpha=args.alpha, beta=args.beta, gamma=args.gamma, n_terms=args.n_terms
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--sample", action="store_true", help="use the sample index")
    ap.add_argument("--phones", type=Path, help="build an in-memory index from this file")
    ap.add_argument("--m", type=int, default=10, help="pseudo-relevant documents")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--n-terms", type=int, default=20, help="terms kept in the updated query")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--query", help="print the top 10 for this text instead of writing a run")
    args = ap.parse_args(argv)

    retriever = build_retriever(args)
    if retriever is None:
        return 1

    if args.query:
        expanded = retriever.expanded_query(args.query)
        shown = sorted(expanded.items(), key=lambda kv: -kv[1])[:10]
        print("query terms:", ", ".join(f"{t}:{w:.3f}" for t, w in shown))
        for rank, (doc_id, score) in enumerate(retriever.score_vector(expanded, 10), 1):
            print(f"{rank:2d}  {score:.4f}  {doc_id}")
        return 0

    if not get_path("queries").exists():
        print(f"no queries file at {get_path('queries')}", file=sys.stderr)
        return 1
    queries = read_queries(get_path("queries"), None if args.split == "all" else args.split)
    out = run_retriever(retriever, queries, k=args.k, run_name=f"{retriever.name}_{args.split}")
    print(f"wrote {out} ({len(queries)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
