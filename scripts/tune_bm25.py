"""Tune BM25 k1/b and the BM25F field weights on the DEV queries only (T05).

    python scripts/tune_bm25.py                    # dev, objective from config (nDCG@10)
    python scripts/tune_bm25.py --objective MAP
    python scripts/tune_bm25.py --no-write-config  # don't touch config.yaml

Protocol:
1. Grid k1 x b (``bm25.tuning.k1_grid`` / ``b_grid`` in config.yaml) with plain BM25 on the
   ``text`` field; pick the best mean objective over dev queries (ties: closest to the
   default k1 = 1.2, b = 0.75).
2. At that k1/b, grid the BM25F-lite weights title x features (``title_grid`` /
   ``features_grid``; description fixed at 1).
3. Write ``results/bm25_tuning.json`` (full grids, best params, objective),
   ``report/figures/bm25_tuning_heatmap.{png,pdf}`` (+ ``bm25f_tuning_heatmap``), and the best
   params into config.yaml under ``bm25.tuned`` (between the BEGIN/END marker comments, so the
   rest of the file and its comments stay untouched).

``--split test`` is refused on purpose: tuning on the test queries would leak them into the
parameters and invalidate the final test numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from phonesense.config import get_config, get_path, write_tuned_block
from phonesense.eval.evaluate import evaluate_rankings, resolve_qrels_path
from phonesense.eval.metrics import canonical_metric, metric_names
from phonesense.index import StaleIndexError, open_index
from phonesense.io import read_qrels, read_queries
from phonesense.retrieval.bm25 import BM25FRetriever, BM25Retriever
from phonesense.schemas import Query

REFUSE_TEST = (
    "error: tune_bm25.py only tunes on the dev queries. Choosing k1/b or field weights on the "
    "{split} queries would leak test information into the parameters and invalidate our final "
    "test results. Tune on dev (the default), then run "
    "`python scripts/run_baselines.py --split test` once with the tuned parameters."
)
def score(retriever: Any, queries: list[Query], qrels: dict, k: int) -> dict[str, float]:
    """Mean metrics of ``retriever`` over ``queries`` (via the T11 harness)."""
    rankings = {q.qid: [d for d, _ in retriever.search(q, k)] for q in queries}
    _, means, _, _ = evaluate_rankings(rankings, qrels, {q.qid for q in queries})
    return means


def write_tuned(tuned: dict[str, Any], path: Path | None = None) -> Path:
    """Replace the ``bm25.tuned`` block in config.yaml (between the marker comments)."""
    return write_tuned_block("bm25", tuned, path)


def _best(rows: list[dict], key: str, prefer: dict[str, float]) -> dict:
    def dist(r: dict) -> float:
        return sum((r[p] - v) ** 2 for p, v in prefer.items())

    return min(rows, key=lambda r: (-round(r[key], 12), dist(r)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--split",
        default="dev",
        choices=("dev", "test", "all"),
        help="must be dev (test/all are refused)",
    )
    ap.add_argument("--objective", help="metric to maximize (default bm25.tuning.objective)")
    ap.add_argument("--qrels", help="qrels file (default qrels/qrels.txt, else qrels/llm.txt)")
    ap.add_argument("--queries", type=Path, help="queries file (default queries/queries.jsonl)")
    ap.add_argument("--sample", action="store_true", help="use the sample index (smoke tests)")
    ap.add_argument("--k", type=int, default=100, help="ranking depth per query")
    ap.add_argument("--no-write-config", action="store_true")
    ap.add_argument("--out", type=Path, help="tuning JSON (default results/bm25_tuning.json)")
    ap.add_argument("--figures-dir", type=Path, help="default report/figures/")
    args = ap.parse_args(argv)

    if args.split != "dev":
        print(REFUSE_TEST.format(split=args.split), file=sys.stderr)
        return 2

    tcfg = (get_config().get("bm25") or {}).get("tuning") or {}
    queries_path = args.queries or get_path("queries")
    if not queries_path.exists():
        print(f"error: {queries_path} does not exist yet (queries are T08).", file=sys.stderr)
        return 1
    queries = read_queries(queries_path, split="dev")
    if not queries:
        print(f"error: no dev queries in {queries_path}", file=sys.stderr)
        return 1
    try:
        qrels_path = resolve_qrels_path(args.qrels)
        index, is_sample = open_index(args.sample)
    except (FileNotFoundError, StaleIndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    qrels = read_qrels(qrels_path)
    if not any(qrels.get(q.qid) for q in queries):
        print(f"error: {qrels_path} has no judgments for the dev queries", file=sys.stderr)
        return 1

    objective = canonical_metric(args.objective or tcfg.get("objective", "nDCG@10"), metric_names())
    k1_grid = [float(x) for x in tcfg.get("k1_grid", [0.4 + 0.2 * i for i in range(9)])]
    b_grid = [float(x) for x in tcfg.get("b_grid", [i / 10 for i in range(11)])]
    t0 = time.perf_counter()

    bm25 = BM25Retriever(index=index)
    grid = []
    for k1 in k1_grid:
        for b in b_grid:
            bm25.set_params(k1, b)
            means = score(bm25, queries, qrels, args.k)
            grid.append({"k1": k1, "b": b, objective: means[objective], "metrics": means})
        print(
            f"  k1={k1:g}: best b so far "
            f"{max((r for r in grid if r['k1'] == k1), key=lambda r: r[objective])['b']:g}"
        )
    best = _best(grid, objective, {"k1": 1.2, "b": 0.75})
    default = next((r for r in grid if r["k1"] == 1.2 and r["b"] == 0.75), None)
    if default is None:  # b = 0.75 is not on the 0.1-step grid: score the default separately
        bm25.set_params(1.2, 0.75)
        means = score(bm25, queries, qrels, args.k)
        default = {"k1": 1.2, "b": 0.75, objective: means[objective], "metrics": means}
    print(
        f"best BM25: k1={best['k1']:g} b={best['b']:g} {objective}={best[objective]:.4f}"
        + (f" (default 1.2/0.75: {default[objective]:.4f})" if default else "")
    )

    title_grid = [float(x) for x in tcfg.get("title_grid", [1, 2, 3, 5])]
    feat_grid = [float(x) for x in tcfg.get("features_grid", [1, 1.5, 2])]
    fgrid = []
    for tw in title_grid:
        for fw in feat_grid:
            r = BM25FRetriever(
                {"title": tw, "features": fw, "description": 1.0},
                best["k1"],
                best["b"],
                index=index,
            )
            means = score(r, queries, qrels, args.k)
            fgrid.append(
                {
                    "title": tw,
                    "features": fw,
                    "description": 1.0,
                    objective: means[objective],
                    "metrics": means,
                }
            )
    fbest = _best(fgrid, objective, {"title": 3.0, "features": 1.5})
    print(
        f"best BM25F weights: title={fbest['title']:g} features={fbest['features']:g} "
        f"description=1 {objective}={fbest[objective]:.4f}"
    )

    out = args.out or get_path("results_dir") / "bm25_tuning.json"
    report = {
        "split": "dev",
        "objective": objective,
        "qrels": str(qrels_path),
        "index": "sample" if is_sample else "full",
        "n_queries": len(queries),
        "k": args.k,
        "seconds": round(time.perf_counter() - t0, 2),
        "best": {"k1": best["k1"], "b": best["b"], objective: best[objective]},
        "default": {"k1": 1.2, "b": 0.75, objective: default[objective]} if default else None,
        "grid": grid,
        "bm25f": {
            "k1": best["k1"],
            "b": best["b"],
            "best": {
                "title": fbest["title"],
                "features": fbest["features"],
                "description": 1.0,
                objective: fbest[objective],
            },
            "grid": fgrid,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")

    from phonesense.eval.plots import heatmap

    mat = [
        [next(r[objective] for r in grid if r["k1"] == k1 and r["b"] == b) for b in b_grid]
        for k1 in k1_grid
    ]
    figs = heatmap(
        mat,
        k1_grid,
        b_grid,
        "bm25_tuning_heatmap",
        "k1",
        "b",
        f"BM25 {objective} on dev ({len(queries)} queries)",
        objective,
        args.figures_dir,
        best=(k1_grid.index(best["k1"]), b_grid.index(best["b"])),
    )
    fmat = [
        [
            next(r[objective] for r in fgrid if r["title"] == tw and r["features"] == fw)
            for fw in feat_grid
        ]
        for tw in title_grid
    ]
    figs += heatmap(
        fmat,
        title_grid,
        feat_grid,
        "bm25f_tuning_heatmap",
        "title weight",
        "features weight",
        f"BM25F {objective} on dev (k1={best['k1']:g}, b={best['b']:g})",
        objective,
        args.figures_dir,
        best=(title_grid.index(fbest["title"]), feat_grid.index(fbest["features"])),
    )
    print("wrote " + ", ".join(str(p) for p in figs))

    if is_sample:
        print("note: tuned on the sample index; not writing config.yaml")
    elif not args.no_write_config:
        tuned = {
            "k1": best["k1"],
            "b": best["b"],
            "field_weights": {
                "title": fbest["title"],
                "features": fbest["features"],
                "description": 1.0,
            },
            "objective": objective,
            "dev_bm25": round(best[objective], 4),
            "dev_bm25f": round(fbest[objective], 4),
            "n_dev_queries": len(queries),
        }
        print(f"wrote bm25.tuned to {write_tuned(tuned)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
