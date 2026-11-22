"""Dense retrieval runs (T12).

    python scripts/run_dense.py --split dev               # -> runs/dense_<model>_dev.run
    python scripts/run_dense.py --split all --phones data/processed/sample/phones.jsonl
    python scripts/run_dense.py --query "phone with great night camera"   # print the top 10
    python scripts/run_dense.py --split dev --encoder hashing             # offline smoke test

Embeds every phone once (cached under ``.cache/dense``), then writes the top-100 per query in
TREC run format (CONTRACT.md 4.6). Needs ``pip install -r requirements-dense.txt`` unless
``--encoder hashing`` is used.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from phonesense.config import get_path
from phonesense.io import read_phones, read_queries
from phonesense.retrieval.base import run_retriever
from phonesense.retrieval.dense import DenseRetriever, HashingEncoder, encoder_from_config
from phonesense.schemas import Query


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--phones", type=Path, default=None, help="default: paths.phones")
    ap.add_argument("--encoder", choices=("sentence-transformer", "hashing"),
                    default="sentence-transformer")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--query", help="print the top 10 for this text instead of writing a run")
    args = ap.parse_args(argv)

    phones_path = args.phones or get_path("phones")
    if not phones_path.exists():
        print(f"no phones file at {phones_path}: run scripts/build_data.py first", file=sys.stderr)
        return 1
    encoder = HashingEncoder() if args.encoder == "hashing" else encoder_from_config()
    retriever = DenseRetriever(encoder)

    t0 = time.perf_counter()
    retriever.fit(read_phones(phones_path))
    state = "cache hit" if retriever.cache_hit else "embedded"
    print(f"{len(retriever.doc_ids)} phones, {state}, {time.perf_counter() - t0:.1f}s")

    if args.query:
        titles = {p.doc_id: p.title for p in read_phones(phones_path)}
        for rank, (doc_id, score) in enumerate(retriever.score_text(args.query, 10), 1):
            print(f"{rank:2d}  {score:.4f}  {doc_id}  {titles[doc_id][:70]}")
        return 0

    queries: list[Query] = read_queries(
        get_path("queries"), None if args.split == "all" else args.split
    )
    out = run_retriever(retriever, queries, k=args.k, run_name=f"{retriever.name}_{args.split}")
    print(f"wrote {out} ({len(queries)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
