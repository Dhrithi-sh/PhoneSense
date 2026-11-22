"""Build the phone graph (T16) and write data/processed/graph_edges.tsv.

    python scripts/build_graph.py                    # co-review edges from reviews.jsonl
    python scripts/build_graph.py --embeddings .cache/dense/<model>_<key>.npy   # + kNN edges
    python scripts/build_graph.py --toy              # seeded toy reviews, writes nothing to data/

``--embeddings`` takes the dense retriever's cache: a .npy matrix with a sibling .json whose
"doc_ids" list gives the row order. Also writes results/graph_stats.json (edges and coverage
per type, dangling and isolated phones, PageRank top 10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phonesense import io
from phonesense.config import get_config, get_path
from phonesense.graph.build_graph import build_graph
from phonesense.graph.pagerank import pagerank
from phonesense.trust.toy import make_toy_reviews


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    matrix = np.load(path)
    if len(meta["doc_ids"]) != len(matrix):
        raise SystemExit(f"{path}: {len(matrix)} rows but {len(meta['doc_ids'])} doc_ids")
    return meta["doc_ids"], matrix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--toy", action="store_true", help="use seeded toy reviews; write nothing")
    ap.add_argument("--knn-k", type=int, default=None,
                    help="kNN edges per phone (default: graph.knn_k)")
    ap.add_argument("--out", type=Path, default=None,
                    help="edges file to write (default: paths.graph_edges)")
    ap.add_argument("--stats-out", type=Path, default=None,
                    help="stats file to write (default: results/graph_stats.json)")
    args = ap.parse_args()
    cfg = get_config()["graph"]

    if args.toy:
        reviews = make_toy_reviews(n_phones=300, n_users=4000)
        doc_ids = sorted({r.doc_id for r in reviews})
    else:
        phones_path, reviews_path = get_path("phones"), get_path("reviews")
        for p in (phones_path, reviews_path):
            if not p.exists():
                raise SystemExit(f"{p} not found. Run scripts/build_data.py first (or --toy).")
        doc_ids = [p.doc_id for p in io.read_phones(phones_path)]
        reviews = io.read_reviews(reviews_path)
    embeddings = load_embeddings(args.embeddings) if args.embeddings else None

    knn_k = args.knn_k or cfg["knn_k"]
    edges, graph, stats = build_graph(doc_ids, reviews, embeddings, knn_k=knn_k,
                                      co_review_min_shared=cfg["co_review_min_shared"])
    pr = pagerank(graph, damping=cfg["damping"], tol=cfg["tol"])
    stats["pagerank_top10"] = sorted(pr, key=lambda d: (-pr[d], d))[:10]
    stats["source"] = "toy" if args.toy else str(get_path("reviews"))
    print(json.dumps(stats, indent=2))
    if args.toy:
        return
    edges_out = args.out or get_path("graph_edges")
    io.write_graph_edges(edges_out, edges)
    out = args.stats_out or get_path("results_dir") / "graph_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {edges_out} and {out}")


if __name__ == "__main__":
    main()
