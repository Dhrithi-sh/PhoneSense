"""HyDE + Rocchio runs on top of TF-IDF (T15).

    python scripts/run_hyde.py --split dev                    # -> runs/hyde_..._dev.run
    python scripts/run_hyde.py --split dev --delta 0.5 --m 5
    python scripts/run_hyde.py --split dev --offline          # replay logged LLM answers only
    python scripts/run_hyde.py --query "phone with great night camera"   # top 10 + terms

Every LLM call is logged to runs/llm_logs/hyde.jsonl (committed) and repeated calls are
replayed, so a second run costs nothing and is identical. ``--offline`` forbids new calls, which
lets anyone reproduce the numbers without an API key. Needs the provider key in ``.env``
otherwise (see .env.example). Parameters are tuned on dev queries only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phonesense.config import get_path
from phonesense.index import InvertedIndex, index_path
from phonesense.io import read_phones, read_queries
from phonesense.llm.client import LLMClient, LLMError
from phonesense.retrieval.base import run_retriever
from phonesense.retrieval.hyde import HydeGenerator, HydeRetriever
from phonesense.retrieval.tfidf import TfidfRetriever


def build_retriever(args: argparse.Namespace, client: LLMClient) -> HydeRetriever | None:
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
    return HydeRetriever(
        HydeGenerator(client), base, m=args.m, alpha=args.alpha, beta=args.beta,
        gamma=args.gamma, n_terms=args.n_terms, delta=args.delta,
    )


def main(argv: list[str] | None = None, client: LLMClient | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--sample", action="store_true", help="use the sample index")
    ap.add_argument("--phones", type=Path, help="build an in-memory index from this file")
    ap.add_argument("--provider", help="gemini | openai (default: llm.provider in config.yaml)")
    ap.add_argument("--model", help="default: llm.models.<provider> in config.yaml")
    ap.add_argument("--offline", action="store_true", help="replay logged answers, no new calls")
    ap.add_argument("--m", type=int, default=10, help="pseudo-relevant documents")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.75)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--delta", type=float, default=0.75, help="weight of the hypothetical listing")
    ap.add_argument("--n-terms", type=int, default=30, help="terms kept in the updated query")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--query", help="print the top 10 for this text instead of writing a run")
    args = ap.parse_args(argv)

    if client is None:
        client = LLMClient(args.provider, args.model, offline=args.offline)
    retriever = build_retriever(args, client)
    if retriever is None:
        return 1

    try:
        if args.query:
            hypothetical = retriever.generator.generate(args.query)
            print("hypothetical listing:\n  " + hypothetical.replace("\n", "\n  "))
            expanded = retriever.expanded_query(args.query, hypothetical)
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
    except LLMError as exc:
        print(f"LLM error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {out} ({len(queries)} queries); LLM calls made {client.calls_made}, "
          f"replayed {client.calls_replayed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
