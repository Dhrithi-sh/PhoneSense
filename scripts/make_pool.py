"""Pool the top documents of several runs for judging (T09a / T09b).

    python scripts/make_pool.py --split dev                      # every runs/*_dev.run + *_all.run
    python scripts/make_pool.py --split dev --depth 20
    python scripts/make_pool.py --split dev --runs tfidf_lnc.ltc_dev dense_all-minilm-l6-v2_dev

Reads runs/<name>.run (CONTRACT 4.6; ``*_all.run`` runs cover both splits and only the queries of
``--split`` are pooled), takes the top ``--depth`` documents of each run per query
and writes the union to runs/pool_<split>.tsv (qid, doc_id, systems). Judge it next with
scripts/judge_pool.py. Pool every system you will evaluate, baselines included, before judging:
documents outside the pool count as not relevant.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from phonesense.config import get_path
from phonesense.io import read_queries, read_run
from phonesense.llm.pooling import pool_runs, write_pool
from phonesense.schemas import run_name_split


def main(argv: list[str] | None = None, runs_dir: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--runs", nargs="*", help="run names without .run (default: all for the split)")
    ap.add_argument("--depth", type=int, default=15, help="top documents taken from each run")
    ap.add_argument("--out", type=Path, help="default: runs/pool_<split>.tsv")
    args = ap.parse_args(argv)

    runs_dir = runs_dir or get_path("runs_dir")
    names = args.runs or sorted(
        p.stem
        for pattern in (f"*_{args.split}.run", "*_all.run")
        for p in runs_dir.glob(pattern)
        if not p.stem.startswith("pool_")
    )
    if not names:
        print(f"no runs for split {args.split!r} in {runs_dir}", file=sys.stderr)
        return 1
    for name in names:
        if run_name_split(name) not in (args.split, "all"):
            print(f"run {name!r} is not a {args.split} or all run", file=sys.stderr)
            return 1
        if not (runs_dir / f"{name}.run").exists():
            print(f"missing {runs_dir / (name + '.run')}", file=sys.stderr)
            return 1

    qids = [q.qid for q in read_queries(get_path("queries"), args.split)]
    runs = {name: read_run(runs_dir / f"{name}.run") for name in names}
    pool = pool_runs(runs, args.depth, qids)
    out = write_pool(args.out or runs_dir / f"pool_{args.split}.tsv", pool)

    sizes = [len(d) for d in pool.values()]
    unique_by = Counter(s[0] for docs in pool.values() for s in docs.values() if len(s) == 1)
    print(f"{len(names)} runs, depth {args.depth}: {len(pool)}/{len(qids)} queries, "
          f"{sum(sizes)} pairs to judge (avg {sum(sizes) / max(len(sizes), 1):.1f} per query)")
    for name in names:
        print(f"  {name}: {unique_by.get(name, 0)} documents found only by this run")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
