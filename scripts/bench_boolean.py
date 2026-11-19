"""Boolean AND efficiency with vs. without skip pointers (T07) -> results/efficiency.json.

    python scripts/bench_boolean.py              # full index, 200 queries
    python scripts/bench_boolean.py --sample

Draws ``--n`` random AND queries of 2-3 terms (seed from config.yaml) from the ``text``
vocabulary, using terms with df >= ``--min-df`` so intersections are non-trivial, and runs
each with skip-pointer intersection and with a plain linear merge. Each query is timed
``--repeats`` times per mode (best time kept). Reports mean / p95 latency, mean comparisons,
the speedup, and checks both modes return identical results. Merged into
results/efficiency.json under ``boolean_and`` (other keys in that file are kept).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np

from phonesense.config import SEED, get_path
from phonesense.index import StaleIndexError, open_index
from phonesense.retrieval.boolean import BooleanSearcher


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-df", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)
    try:
        index, is_sample = open_index(args.sample)
    except (FileNotFoundError, StaleIndexError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    vocab = [t for t in index.vocabulary("text") if index.df(t) >= args.min_df]
    rng = random.Random(SEED)
    queries = [rng.sample(vocab, rng.choice([2, 3])) for _ in range(args.n)]

    report = {
        "index": "sample" if is_sample else "full",
        "N": index.N,
        "n_queries": args.n,
        "min_df": args.min_df,
        "repeats": args.repeats,
        "seed": SEED,
        "vocab_candidates": len(vocab),
    }
    results = {}
    for mode, skips in (("with_skips", True), ("without_skips", False)):
        bs = BooleanSearcher(index, skips=skips)
        times, comps, sizes = [], [], []
        for terms in queries:
            best = float("inf")
            for _ in range(args.repeats):
                bs.stats.comparisons = 0
                t0 = time.perf_counter()
                res = bs.and_query(terms)
                best = min(best, time.perf_counter() - t0)
            times.append(best * 1e6)
            comps.append(bs.stats.comparisons)
            sizes.append(len(res))
            results.setdefault(tuple(terms), []).append(res)
        t = np.array(times)
        report[mode] = {
            "mean_us": round(float(t.mean()), 2),
            "p95_us": round(float(np.percentile(t, 95)), 2),
            "mean_comparisons": round(float(np.mean(comps)), 1),
            "mean_result_size": round(float(np.mean(sizes)), 2),
        }
    if any(a != b for a, b in results.values()):
        print("error: skip and non-skip intersections differ", file=sys.stderr)
        return 1
    w, wo = report["with_skips"], report["without_skips"]
    report["speedup_mean_latency"] = round(wo["mean_us"] / w["mean_us"], 3)
    report["comparison_reduction"] = round(1 - w["mean_comparisons"] / wo["mean_comparisons"], 3)

    out = get_path("results_dir") / "efficiency.json"
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    data["boolean_and"] = report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
