"""Write fused run files from saved runs (T13): RRF and weighted linear fusion.

    python scripts/run_fusion.py --method rrf --runs bm25_tuned dense_X --rrf-k 5 --split dev
    python scripts/run_fusion.py --method linear --runs bm25_tuned dense_X \\
        --weights 0.55 0.45 --split test

``--runs`` are run base names without the split suffix (the files runs/<name>_<split>.run must
exist); the settings come from scripts/tune_fusion.py on DEV. The output name starts with
``rrf_`` or ``fusion_linear_`` (what ablation.yaml looks for) and carries the setting:
``rrf_k5_<a>+<b>_<split>.run`` and ``fusion_linear_w55-45_<a>+<b>_<split>.run``.

Nothing is tuned here. For ``--split test`` the script refuses to overwrite an existing run
unless ``--force``: test is run once per final configuration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from phonesense.config import get_path
from phonesense.io import read_queries, read_run, write_rankings
from phonesense.retrieval.fusion_tune import fuse_runs
from phonesense.schemas import MAX_RUN_DEPTH


def fused_run_name(method: str, runs: list[str], rrf_k: int | None, weights: list[float] | None,
                   split: str) -> str:
    """``rrf_k5_<a>+<b>_<split>`` or ``fusion_linear_w55-45_<a>+<b>_<split>``."""
    parts = "+".join(runs)
    if method == "rrf":
        return f"rrf_k{rrf_k}_{parts}_{split}"
    tag = "-".join(f"{round(w * 100):d}" for w in (weights or [1.0 / len(runs)] * len(runs)))
    return f"fusion_linear_w{tag}_{parts}_{split}"


def main(argv: list[str] | None = None, runs_dir: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--method", choices=("rrf", "linear"), required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="run base names, no split suffix")
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant (method rrf)")
    ap.add_argument("--weights", nargs="+", type=float, help="one weight per run (method linear)")
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--depth", type=int, default=MAX_RUN_DEPTH)
    ap.add_argument("--force", action="store_true", help="overwrite an existing test run")
    args = ap.parse_args(argv)

    if args.method == "linear" and args.weights and len(args.weights) != len(args.runs):
        print(f"got {len(args.weights)} weights for {len(args.runs)} runs", file=sys.stderr)
        return 1
    runs_dir = runs_dir or get_path("runs_dir")
    sources = []
    for name in args.runs:
        path = runs_dir / f"{name}_{args.split}.run"
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1
        sources.append(read_run(path))
    qids = [q.qid for q in read_queries(get_path("queries"), args.split)]

    out_name = fused_run_name(args.method, args.runs, args.rrf_k, args.weights, args.split)
    out = runs_dir / f"{out_name}.run"
    if args.split == "test" and out.exists() and not args.force:
        print(f"{out} exists: test is run once per configuration (use --force to overwrite)",
              file=sys.stderr)
        return 1

    fused = fuse_runs(sources, qids, args.method, args.weights, args.rrf_k, args.depth)
    write_rankings(out, fused, out_name, args.depth)
    setting = f"k={args.rrf_k}" if args.method == "rrf" else f"weights={args.weights or 'equal'}"
    print(f"wrote {out} ({len(fused)} queries, {args.method}, {setting})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
