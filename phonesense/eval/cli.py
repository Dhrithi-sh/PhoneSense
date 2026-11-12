"""Evaluation CLI (stub until T11).

    python -m phonesense.eval.cli --run runs/bm25_dev.run --qrels qrels/qrels.txt

Today it parses and validates both files against CONTRACT.md and reports coverage.
Metric computation (P@k, R@k, F1, MAP, nDCG@k) arrives in T11.
Exit codes: 0 = inputs valid, 1 = invalid input.
"""

from __future__ import annotations

import argparse
import sys

from phonesense.io import read_qrels, read_run_entries
from phonesense.schemas import SchemaError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m phonesense.eval.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="TREC run file (runs/<run_name>.run)")
    parser.add_argument("--qrels", required=True, help="TREC qrels file (qrels/qrels.txt)")
    args = parser.parse_args(argv)

    try:
        entries = read_run_entries(args.run)
        qrels = read_qrels(args.qrels)
    except (SchemaError, OSError) as e:
        print(f"INVALID INPUT: {e}", file=sys.stderr)
        return 1

    run_qids = {e.qid for e in entries}
    run_name = entries[0].run_name if entries else "(empty)"
    n_judg = sum(len(d) for d in qrels.values())
    print(f"run   OK: {args.run} (run_name={run_name}, {len(run_qids)} queries, "
          f"{len(entries)} lines)")
    print(f"qrels OK: {args.qrels} ({len(qrels)} queries, {n_judg} judgments)")
    missing = sorted(run_qids - qrels.keys())
    if missing:
        print(f"warning: {len(missing)} run queries have no judgments: {', '.join(missing)}")
    unretrieved = sorted(qrels.keys() - run_qids)
    if unretrieved:
        print(f"warning: {len(unretrieved)} judged queries absent from run: "
              f"{', '.join(unretrieved)}")
    print("Inputs are valid.")
    print("Metrics: not implemented yet (T11).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
