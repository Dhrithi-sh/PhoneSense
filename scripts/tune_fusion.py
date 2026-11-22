"""Tune fusion settings on DEV queries only (T13).

    python scripts/tune_fusion.py --runs tfidf_lnc.ltc_dev dense_all-minilm-l6-v2_dev \\
        rocchio_tfidf_lnc.ltc_m10_b0.75_g0_t20_dev
    python scripts/tune_fusion.py --runs a_dev b_dev --method linear --step 0.05

Reads runs/<name>.run for each system, fuses them offline, and picks by mean nDCG@10:
  * RRF:    the constant k from --k-values (equal weights)
  * linear: the weight vector on a grid of --step (weights sum to 1)
Writes results/fusion_dev.json and prints the values to copy into the ``fusion:`` block of
config.yaml. It refuses non-dev runs and never reads test queries. Judgments come from --qrels
(default qrels/llm.txt, the round-1 dev labels). The scores are on a small query set: compare
the tuned score with the equal-weight one before trusting the gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phonesense.config import get_path
from phonesense.io import read_qrels, read_queries, read_run
from phonesense.retrieval.fusion_tune import (
    evaluable_qids,
    fuse_runs,
    mean_ndcg,
    pick_best,
    tune_linear,
    tune_rrf_k,
)
from phonesense.schemas import run_name_split


def main(argv: list[str] | None = None, runs_dir: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="dev run names, without .run")
    ap.add_argument("--method", choices=("rrf", "linear", "both"), default="both")
    ap.add_argument("--step", type=float, default=0.1, help="weight grid step for linear")
    ap.add_argument("--k-values", nargs="+", type=int, default=[10, 30, 60, 100])
    ap.add_argument("--qrels", type=Path, help="default: paths.qrels_llm")
    ap.add_argument("--out", type=Path, help="default: results/fusion_dev.json")
    args = ap.parse_args(argv)

    runs_dir = runs_dir or get_path("runs_dir")
    for name in args.runs:
        if run_name_split(name) != "dev":
            print(f"refusing {name!r}: tuning uses dev runs only", file=sys.stderr)
            return 1
        if not (runs_dir / f"{name}.run").exists():
            print(f"missing {runs_dir / (name + '.run')}", file=sys.stderr)
            return 1
    qrels_path = args.qrels or get_path("qrels_llm")
    if not qrels_path.exists():
        print(f"missing {qrels_path}: judge the dev pool first (scripts/judge_pool.py)",
              file=sys.stderr)
        return 1

    dev_qids = [q.qid for q in read_queries(get_path("queries"), "dev")]
    qrels = read_qrels(qrels_path)
    qids = evaluable_qids(dev_qids, qrels)
    if not qids:
        print("no dev query has a relevant judgment", file=sys.stderr)
        return 1
    systems = [read_run(runs_dir / f"{name}.run") for name in args.runs]

    result: dict = {
        "split": "dev",
        "metric": "ndcg@10",
        "systems": args.runs,
        "qrels": str(qrels_path),
        "queries_evaluated": len(qids),
        "queries_skipped": sorted(set(dev_qids) - set(qids)),
        "single_systems": {
            name: mean_ndcg(run, qrels, qids) for name, run in zip(args.runs, systems, strict=True)
        },
    }
    print(f"{len(qids)}/{len(dev_qids)} dev queries evaluable")
    for name, score in result["single_systems"].items():
        print(f"  alone  {score:.4f}  {name}")

    if args.method in ("rrf", "both"):
        trials = tune_rrf_k(systems, qrels, qids, args.k_values)
        best = pick_best([{**t, "weights": None} for t in trials])
        result["rrf"] = {"trials": trials, "best_rrf_k": best["rrf_k"], "best_score": best["score"]}
        print(f"  RRF    {best['score']:.4f}  best k = {best['rrf_k']}  "
              f"(k=60: {next((t['score'] for t in trials if t['rrf_k'] == 60), float('nan')):.4f})")

    if args.method in ("linear", "both"):
        trials = tune_linear(systems, qrels, qids, args.step)
        best = pick_best(trials)
        equal = mean_ndcg(fuse_runs(systems, qids, "linear"), qrels, qids)
        result["linear"] = {
            "step": args.step,
            "trials": trials,
            "best_weights": best["weights"],
            "best_score": best["score"],
            "equal_weight_score": equal,
        }
        print(f"  linear {best['score']:.4f}  weights {best['weights']}  (equal: {equal:.4f})")

    out = args.out or get_path("results_dir") / "fusion_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("copy into config.yaml (fusion:) once you are happy, and keep the system order:")
    print(f"  systems: {args.runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
