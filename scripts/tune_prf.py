"""Tune Rocchio and HyDE on DEV queries only (T14, T15).

    python scripts/tune_prf.py --method rocchio
    python scripts/tune_prf.py --method hyde --provider groq --model openai/gpt-oss-20b

Grid-searches the settings that matter (how many feedback documents ``m``, the feedback weight
``beta``, the number of kept terms, and for HyDE the weight ``delta`` of the hypothetical
listing) and scores each by mean nDCG@10 with the same harness code the tables use (queries with
no relevant phone are excluded, like in the tables). Only dev queries are read. HyDE's
hypothetical listings are replayed from the logged LLM calls (``offline``), so tuning costs no
API calls and uses exactly the texts of the real runs.

Writes results/prf_<method>_dev.json and prints the best settings and the default's score. Ties
go to the setting closest to the default (a smaller change is less likely to be overfitting 27
queries). Then produce the final runs with scripts/run_rocchio.py / run_hyde.py: dev, and test
once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from phonesense.eval.evaluate import evaluate_rankings

from phonesense.config import get_path
from phonesense.index import InvertedIndex, index_path
from phonesense.io import read_qrels, read_queries
from phonesense.llm.client import LLMClient
from phonesense.retrieval.hyde import HydeGenerator, HydeRetriever
from phonesense.retrieval.rocchio import RocchioRetriever
from phonesense.retrieval.tfidf import TfidfRetriever

ROCCHIO_DEFAULT = {"m": 10, "beta": 0.75, "n_terms": 20}
ROCCHIO_GRID = {"m": [3, 5, 10, 15], "beta": [0.25, 0.5, 0.75, 1.0, 1.5],
                "n_terms": [10, 20, 30, 50]}
HYDE_DEFAULT = {"m": 10, "beta": 0.75, "n_terms": 30, "delta": 0.75}
HYDE_GRID = {"m": [5, 10], "beta": [0.0, 0.25, 0.5, 0.75], "n_terms": [20, 30, 50],
             "delta": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]}


def configs(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*grid.values())]


def distance(cfg: dict, default: dict, grid: dict[str, list]) -> float:
    """How far a setting is from the default, in grid steps (used only to break ties)."""
    return sum(abs(grid[k].index(cfg[k]) - grid[k].index(default[k])) for k in cfg)


def score(retriever, queries, qrels) -> float:
    rankings = {q.qid: [d for d, _ in retriever.search(q, 100)] for q in queries}
    _, means, _, _ = evaluate_rankings(rankings, qrels, {q.qid for q in queries})
    return means["nDCG@10"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--method", choices=("rocchio", "hyde"), required=True)
    ap.add_argument("--qrels", type=Path, help="default: qrels/qrels.txt, else qrels/llm.txt")
    ap.add_argument("--provider", default="groq", help="provider of the logged HyDE calls")
    ap.add_argument("--model", default="openai/gpt-oss-20b", help="model of the logged HyDE calls")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    qrels_path = args.qrels or next(
        (p for p in (get_path("qrels"), get_path("qrels_llm")) if p.exists()), None
    )
    if qrels_path is None:
        print("no qrels found: run scripts/judge_pool.py first", file=sys.stderr)
        return 1
    qrels = read_qrels(qrels_path)
    queries = read_queries(get_path("queries"), "dev")  # dev only, by construction
    path = index_path(False)
    if not (path / "meta.json").exists():
        print(f"no index at {path}: python scripts/build_index.py", file=sys.stderr)
        return 1
    base = TfidfRetriever()
    base.fit_index(InvertedIndex.load(path))

    if args.method == "rocchio":
        grid, default = ROCCHIO_GRID, ROCCHIO_DEFAULT
    else:
        grid, default = HYDE_GRID, HYDE_DEFAULT
    generator = None
    if args.method == "hyde":
        generator = HydeGenerator(LLMClient(args.provider, args.model, offline=True))

    def build(cfg: dict):
        if args.method == "rocchio":
            return RocchioRetriever(base, **cfg)
        return HydeRetriever(generator, base, **cfg)

    trials = []
    for cfg in configs(grid):
        trials.append({**cfg, "ndcg10": score(build(cfg), queries, qrels)})
    default_score = next(t["ndcg10"] for t in trials if all(t[k] == v for k, v in default.items()))
    best = max(t["ndcg10"] for t in trials)
    tied = [t for t in trials if t["ndcg10"] >= best - 1e-9]
    chosen = min(tied, key=lambda t: distance({k: t[k] for k in default}, default, grid))

    print(f"{args.method}: {len(trials)} settings on {len(queries)} dev queries, qrels {qrels_path}")
    print(f"  default {default}: nDCG@10 {default_score:.4f}")
    for t in sorted(trials, key=lambda t: -t["ndcg10"])[:6]:
        print(f"  {t['ndcg10']:.4f}  " + ", ".join(f"{k}={t[k]}" for k in default))
    gain = chosen["ndcg10"] - default_score
    picked = ", ".join(f"{k}={chosen[k]}" for k in default)
    print(f"chosen: {picked}  ({chosen['ndcg10']:.4f}, default {default_score:.4f}, gain {gain:+.4f})")

    out = args.out or get_path("results_dir") / f"prf_{args.method}_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"method": args.method, "split": "dev", "qrels": str(qrels_path),
               "default": default, "default_ndcg10": default_score,
               "chosen": {k: chosen[k] for k in default},
               "chosen_ndcg10": chosen["ndcg10"], "trials": trials}
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
