"""One-command reproduction from processed data (= make reproduce; FULL=1 adds the raw build).

    python scripts/reproduce.py           # from data/processed/phones.jsonl
    python scripts/reproduce.py --full    # first download + rebuild the data (slow, ~1 h)

Steps: [data] -> index -> baselines dev/test -> T06/T07 runs dev/test -> eval -> ablation ->
report numbers. Tuned parameters are read from config.yaml (``bm25.tuned``, ``phrase.tuned``,
written once by the dev-only tuning scripts), so this does not re-tune; test runs are
regenerated with ``--force`` because they are deterministic re-runs of the same settings.
Teammates' runs (dense, Rocchio, HyDE, fusion, graph reranks) are used if present in
``runs/``; they come from their own scripts (README "Reproduce").

A step that fails because its inputs don't exist yet (e.g. no qrels) is reported and the
remaining steps still run, so the ablation shows those rows as pending.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from phonesense.config import repo_root


def steps(full: bool) -> list[tuple[str, list[str]]]:
    s = [("data", ["scripts/build_data.py"])] if full else []
    s += [
        ("index", ["scripts/build_index.py"]),
        ("baselines dev", ["scripts/run_baselines.py", "--split", "dev"]),
        ("baselines test", ["scripts/run_baselines.py", "--split", "test", "--force"]),
        ("T06/T07 dev", ["scripts/run_t06_t07.py", "--split", "dev"]),
        ("T06/T07 test", ["scripts/run_t06_t07.py", "--split", "test", "--force"]),
        ("efficiency", ["scripts/bench_boolean.py"]),
        ("eval", ["scripts/run_eval.py"]),
        ("ablation", ["scripts/ablation.py"]),
        ("report numbers", ["scripts/report_numbers.py"]),
    ]
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also rebuild data from raw")
    args = ap.parse_args(argv)
    summary = []
    for name, cmd in steps(args.full):
        print(f"\n=== {name}: python {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run([sys.executable, *cmd], cwd=repo_root()).returncode
        summary.append((name, rc, time.perf_counter() - t0))
        if rc != 0 and name in ("data", "index"):
            print(f"step {name} failed; stopping", file=sys.stderr)
            break
    print("\n=== summary")
    for name, rc, sec in summary:
        print(f"  {'ok    ' if rc == 0 else 'FAILED'}  {name:16s} {sec:7.1f}s")
    return 0 if all(rc == 0 for name, rc, _ in summary if name in ("data", "index")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
