"""Run everything in Rishit's part in one go: tests, evaluations and a demo.

    .venv/bin/python scripts/rishit_results.py            # everything
    .venv/bin/python scripts/rishit_results.py --quick    # tests and the demo only

Steps:
1. Tests: partwise-tests/rishit (and the whole partwise-tests folder).
2. Evaluations on the frozen corpus: the experiment runner (runs, metrics,
   significance, speed-ups), the sanity check, learning to rank, and the
   stop word / idf / Zipf statistics. Outputs go to data/eval/full.
3. Demo: the queries that show the system best, the same need in Hindi,
   Hinglish and English, plus kal, an English query that needs translation,
   and one --explain run.

Each step's output is printed with a heading, and a summary at the end says
which steps passed. Steps that need the real index are skipped (with a note)
if it isn't built.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

DEMO = [
    ("Same need, three ways: Hindi", ["भूकंप के झटके"]),
    ("Same need, three ways: Hinglish", ["bhukamp ke jhatke"]),
    ("Same need, three ways: English", ["earthquake tremors delhi"]),
    ("English query, Hindi articles (translation + phonetic)", ["rahul gandhi detained protest"]),
    ("Hinglish names", ["shreyas iyer shatak"]),
    ("Messy spelling", ["smriti mandana captain bani"]),
    ("The assignment's own example: kal ka mausam", ["kal ka mausam"]),
    ("Date-aware kal: tomorrow's weather", ["kal mausam kaisa rahega"]),
    ("Fusion of all rankers", ["chardham yatra record", "--ranker", "rrf"]),
    ("A vague query gets a low-confidence warning", ["aaj ki khabar"]),
    ("Every step of the pipeline (--explain)", ["भूकंप के झटके", "--k", "2", "--explain"]),
]


def run(title, cmd, results, timeout=1800):
    print(f"\n{'=' * 78}\n{title}\n$ {' '.join(str(c) for c in cmd)}\n{'=' * 78}", flush=True)
    start = time.time()
    try:
        done = subprocess.run(cmd, cwd=ROOT, timeout=timeout)
        ok = done.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"timed out after {timeout} s")
        ok = False
    results.append((title, ok, time.time() - start))
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tests, evaluations and a demo for Rishit's part.")
    parser.add_argument("--quick", action="store_true", help="tests and the demo only, no evaluations")
    parser.add_argument("--dense", action="store_true", help="include the e5 feature in learning to rank")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from dhvani.rank.real_index import real_index_available
    real = real_index_available("none")

    results = []
    run("1a. Rishit's tests", [PY, "-m", "pytest", "-q", "partwise-tests/rishit"], results)
    run("1b. All partwise tests", [PY, "-m", "pytest", "-q", "partwise-tests"], results)

    if not args.quick:
        if not real:
            print("\nThe real index isn't built, so the evaluations are skipped. Build it with:\n"
                  "  .venv/bin/python -m index.build --input data/news.jsonl")
        else:
            run("2a. Experiment runner: runs, metrics, significance, speed-ups",
                [PY, "-m", "dhvani.eval.experiments", "--out", "data/eval/full"], results)
            run("2b. Sanity check without judgments", [PY, "-m", "dhvani.eval.sanity"], results)
            ltr = [PY, "-m", "dhvani.eval.ltr"] + (["--dense"] if args.dense else [])
            run("2c. Learning to rank (leave-one-need-out)", ltr, results)
            run("2d. Stop words, idf and Zipf", [PY, "-m", "dhvani.eval.corpus_stats", "--out", "data/eval/full"], results)

    for i, (title, query) in enumerate(DEMO, start=1):
        run(f"3.{i} Demo: {title}", [PY, "app/cli.py", *query, *([] if "--k" in query else ["--k", "3"])], results)

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    for title, ok, secs in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {secs:6.1f} s  {title}")
    failed = [t for t, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)} of {len(results)} steps passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
